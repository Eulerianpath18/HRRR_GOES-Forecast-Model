"""PyTorch tensors and sequential acquisition. Policy observations never contain labels."""
from __future__ import annotations
import json
from pathlib import Path
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .core import write_json, digest
from .satellite import crop


class Standardizer:
    def __init__(self, values=None, fingerprint=None):
        self.values = values or {}
        self.fingerprint = fingerprint

    @classmethod
    def fit(cls, root, path=None, sample_ids=None):
        root = Path(root)
        manifest = json.loads((root/'manifest.json').read_text())
        df = pd.read_parquet(root/'samples.parquet')
        opt = pd.read_parquet(root/'optional.parquet')
        df = df[df.split == 'train'].merge(opt,on='sample_id',validate='one_to_one')
        if df.empty: raise ValueError('No training rows: configure chronological splits first')
        if sample_ids is not None:
            wanted = set(sample_ids)
            if not wanted <= set(df.sample_id):
                raise ValueError('Normalizer IDs must belong to the training split')
            df = df[df.sample_id.isin(wanted)]
            if df.empty:
                raise ValueError('No normalizer fitting rows')
        columns = manifest['core_features']+manifest['mos_extra']
        columns += [col for a in manifest['actions'] if a['kind']=='tabular' for col in a['columns']]
        values = {}
        for col in dict.fromkeys(columns):
            x = df[col].to_numpy(dtype=float); x = x[np.isfinite(x)]
            values[col] = [float(x.mean()) if len(x) else 0.,max(float(x.std()),1e-6) if len(x) else 1.]
        obj = cls(values,manifest['fingerprint'])
        if path: write_json(path,{'values':values,'fingerprint':manifest['fingerprint'],
                                  'train_ids_digest':digest(sorted(df.sample_id.tolist())),
                                  'n_train':len(df)})
        return obj

    @classmethod
    def load(cls,path):
        data = json.loads(Path(path).read_text())
        return cls(data['values'],data['fingerprint'])

    def apply(self, values, columns):
        x = np.asarray(values,dtype='float32')
        mask = np.isfinite(x)
        for i,col in enumerate(columns):
            mean,sd = self.values.get(col,[0.,1.])
            x[i] = (x[i]-mean)/sd if mask[i] else 0.
        return torch.from_numpy(x),torch.from_numpy(mask)


def image_tensors(x):
    # Fixed physical scaling, no statistics from validation/test. Units remain interpretable.
    mask = np.isfinite(x)
    return torch.from_numpy(np.where(mask,(x-250.)/50.,0).astype('float32')),torch.from_numpy(mask)


class FeatureStore:
    def __init__(self,root,mode='direct',normalizer=None):
        if mode not in ['direct','mos']: raise ValueError('mode must be direct or mos')
        self.root,self.mode = Path(root),mode
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.rows = pd.read_parquet(self.root/'samples.parquet').set_index('sample_id',drop=False)
        self.optional = pd.read_parquet(self.root/'optional.parquet').set_index('sample_id')
        self.normalizer = normalizer or Standardizer()
        if self.normalizer.fingerprint not in [None,self.manifest['fingerprint']]:
            raise ValueError('Normalizer belongs to another dataset')
        self.columns = self.manifest['core_features']+(self.manifest['mos_extra'] if mode=='mos' else [])
        self.actions = {a['id']:a for a in self.manifest['actions'] if mode in a['modes']}
        self.g = self.manifest['satellite']

    def resources(self,sample_id,action):
        meta = self.manifest['samples'][sample_id]
        if action['kind']=='satellite':
            info = meta['satellite'].get(str(action['band']))
            return {info['resource_id']} if info else set()
        return {r for col in action['columns'] for r in meta['tabular_resources'].get(col,[])}

    def empty_action(self,action):
        shape = (len(action['columns']),) if action['kind']=='tabular' else (1,action['pixels'],action['pixels'])
        return {'value':torch.zeros(shape),'valid':torch.zeros(shape,dtype=torch.bool),
                'acquired':torch.tensor(False)}

    def load_action(self,sample_id,action):
        if action['kind']=='tabular':
            values = self.optional.loc[sample_id,action['columns']].to_numpy(dtype=float)
            value,valid = self.normalizer.apply(values,action['columns'])
        else:
            metadata = self.manifest['samples'][sample_id]
            prepared = metadata.get('prepared_actions', {}).get(action['id'])
            if prepared:
                with np.load(self.root/prepared, allow_pickle=False) as data:
                    value,valid = image_tensors(data['image_k'])
                return {'value':value,'valid':valid,'acquired':torch.tensor(True)}
            info = metadata['satellite'].get(str(action['band']))
            if not info:
                blank = self.empty_action(action); blank['acquired'] = torch.tensor(True)
                return blank
            lead = int(self.rows.loc[sample_id,'lead_h'])
            footprints = self.g['lead_footprints_km']
            width = footprints.get(str(lead),footprints.get(lead))*action.get('scale',1.)
            with np.load(self.root/info['path'],allow_pickle=False) as d:
                x = crop(d['image_k'],float(d['footprint_km']),width,action['pixels'],
                         action.get('east_km',0),action.get('north_km',0))
            value,valid = image_tensors(x)
        return {'value':value,'valid':valid,'acquired':torch.tensor(True)}

    def core(self,sample_id):
        row = self.rows.loc[sample_id]
        values,mask = self.normalizer.apply(row[self.columns].to_numpy(dtype=float),self.columns)
        p = self.g['overview_pixels']
        overview,overview_mask = torch.zeros((1,p,p)),torch.zeros((1,p,p),dtype=torch.bool)
        if row.overview_path:
            with np.load(self.root/row.overview_path,allow_pickle=False) as d:
                overview,overview_mask = image_tensors(d['image_k'])
        return {'core':values,'core_mask':mask,'overview':overview,'overview_mask':overview_mask}


class AcquisitionSession:
    """Offline replay environment. Budget covers OPTIONAL compute only.

    Source byte accounting is separate and counts distinct GOES/GRIB resources;
    costs are supplied engineering proxies, not measured inference latency.
    No label or hidden feature value is returned by observation()/available_actions().
    """
    def __init__(self,store,sample_id,budget,paid_resources=None):
        if math.isnan(budget) or budget < 0: raise ValueError('budget must be nonnegative')
        self.store,self.sample_id = store,sample_id
        self.budget,self.spent = float(budget),0.
        self.load_seconds = {}
        self.source_bytes = 0; self.unknown_size_resources = set()
        self.paid = paid_resources if paid_resources is not None else set()
        self.acquired = set()
        self.state = store.core(sample_id)
        self.state['optional'] = {k:store.empty_action(a) for k,a in store.actions.items()}
        meta = store.manifest['samples'][sample_id]
        # Mandatory overview has already paid for its parent raw scene.
        image = meta['satellite'].get(str(store.g['overview_band']))
        if image: self.paid.add(image['resource_id'])
        if store.mode=='mos': self.paid.update(meta['tabular_resources'].get('hrrr_t2m_c',[]))

    def available_actions(self):
        # Selection uses metadata and costs only, never undisclosed values or DQF.
        return {k:a['cost_units'] for k,a in self.store.actions.items()
                if k not in self.acquired and a['cost_units'] <= self.budget-self.spent+1e-9
                and set(a.get('requires',[])) <= self.acquired}

    def acquire(self,action_id):
        if action_id in self.acquired: raise ValueError('Action already acquired')
        if action_id not in self.available_actions(): raise ValueError('Action unavailable or over budget')
        action = self.store.actions[action_id]
        started = time.perf_counter()
        result = self.store.load_action(self.sample_id,action)
        self.load_seconds[action_id] = time.perf_counter()-started
        self.state['optional'][action_id] = result
        self.acquired.add(action_id); self.spent += action['cost_units']
        for rid in self.store.resources(self.sample_id,action)-self.paid:
            n = self.store.manifest['resources'][rid].get('bytes')
            if n is None: self.unknown_size_resources.add(rid)
            else: self.source_bytes += n
            self.paid.add(rid)
        return self.observation()

    def observation(self):
        # Clone so a learner cannot mutate cached state or earlier observations.
        def clone(x): return {k:clone(v) for k,v in x.items()} if isinstance(x,dict) else x.clone()
        result = clone(self.state)
        result['remaining_budget'] = torch.tensor(self.budget-self.spent,dtype=torch.float32)
        return result


class ForecastDataset(Dataset):
    """CPU tensors suitable for DataLoader; fixed action selection or core-only.

    Random/sequential policies should use AcquisitionSession. Targets are attached
    here only for supervised training, never included in session observations.
    """
    def __init__(self,root,mode='direct',split=None,normalizer=None,actions=(),budget=float('inf')):
        self.store = FeatureStore(root,mode,normalizer)
        f = self.store.rows
        if split is not None: f = f[f.split==split]
        target = 'observed_t2m_c' if mode=='direct' else 'error_c'
        self.ids = f.index[np.isfinite(f[target])].tolist()
        self.target,self.actions,self.budget = target,tuple(actions),budget
        if set(self.actions)-set(self.store.actions): raise ValueError('Action is not permitted for this model mode')
        if sum(self.store.actions[a]['cost_units'] for a in self.actions)>budget+1e-9:
            raise ValueError('Requested fixed actions exceed budget')

    def __len__(self): return len(self.ids)

    def __getitem__(self,index):
        sid = self.ids[index]
        session = AcquisitionSession(self.store,sid,self.budget)
        for a in self.actions: session.acquire(a)
        return {'inputs':session.observation(),'target':torch.tensor(float(self.store.rows.loc[sid,self.target]),dtype=torch.float32),
                'sample_id':sid}


class RandomAcquisitionDataset(ForecastDataset):
    """Random valid masks for training a predictor on partially acquired inputs.

    Uses PyTorch's worker RNG. Seed torch/DataLoader for reproducibility. Budgets are
    sampled uniformly from the supplied list; this is mask augmentation, not a policy.
    """
    def __init__(self,root,mode='direct',split='train',normalizer=None,
                 budgets=(0.,1.,2.),stop_probability=0.2):
        super().__init__(root,mode,split,normalizer)
        if not budgets or any(not math.isfinite(b) or b<0 for b in budgets):
            raise ValueError('Mask budgets must be finite and nonnegative')
        if not 0<=stop_probability<=1: raise ValueError('Invalid stopping probability')
        self.budgets,self.stop_probability = tuple(budgets),stop_probability

    def __getitem__(self,index):
        sid = self.ids[index]
        budget = self.budgets[torch.randint(len(self.budgets),()).item()]
        session = AcquisitionSession(self.store,sid,budget)
        while session.available_actions():
            if torch.rand(()).item()<self.stop_probability: break
            actions = list(session.available_actions())
            session.acquire(actions[torch.randint(len(actions),()).item()])
        return {'inputs':session.observation(),
                'target':torch.tensor(float(self.store.rows.loc[sid,self.target]),dtype=torch.float32),
                'sample_id':sid}
