"""Portable inference bundles and causally aligned live desktop inputs."""
import copy
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .core import load_config, stations, utc, digest, write_json
from .runtime import atomic_torch_save, sha256, locked, source_id
from .torch_data import FeatureStore, Standardizer, AcquisitionSession
from .training import load_predictor, make_model
from .policy import acquire, predict


def export_bundle(root, checkpoint, output, policy_path=None):
    store, model, saved = load_predictor(root, checkpoint, 'cpu')
    policy = None
    if saved['config']['acquisition'] == 'random':
        import joblib
        if not policy_path or not Path(policy_path).exists():
            raise ValueError('Masked model requires its fitted policy before GUI export')
        policy = joblib.load(policy_path)
        if policy['checkpoint_sha256'] != sha256(checkpoint) or policy['data_id'] != saved['identity']['data_id']:
            raise ValueError('Policy and predictor identities differ')
    schema = {key:store.manifest[key] for key in ['core_features','mos_extra','actions','satellite','config']}
    atomic_torch_save(output, {'format':'forecastprep-gui-v1', 'source_id':source_id(),
        'model':saved['model'], 'normalizer':saved['normalizer'], 'config':saved['config'],
        'schema':schema, 'stations':stations(Path(root)/'stations.csv'), 'policy':policy,
        'training_data_id':saved['identity']['data_id'], 'validation_mse':saved['validation_mse'],
        'checkpoint_sha256':sha256(checkpoint)})
    return str(output)


def export_selected(root, runs, output):
    runs, output = Path(runs), Path(output)
    selected=json.loads((runs/'selected.json').read_text())['selected_ids']
    files=[]
    for name in selected:
        files.append(export_bundle(root,runs/'training'/name/'best.pt',output/(name+'.pt'),runs/'policies'/name/'policy.joblib'))
    write_json(output/'index.json',{'bundles':[Path(p).name for p in files]})
    return {'bundles':len(files),'directory':str(output)}


def load_bundle(path):
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    if bundle.get('format')!='forecastprep-gui-v1':
        raise ValueError('Choose an exported GUI bundle, not last.pt or best.pt')
    if bundle['source_id']!=source_id():
        raise ValueError('This model bundle requires the project code version used to export it')
    return bundle


def latest_cycle(config, now=None):
    now=utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    deadline=now-pd.Timedelta(minutes=config['issue_delay_minutes'])
    choices=[day+pd.Timedelta(hours=h) for day in pd.date_range(deadline.normalize()-pd.Timedelta(days=2),deadline.normalize(),freq='D') for h in config['cycles']]
    return max(t for t in choices if t<=deadline)


def validate_live_config(config, bundle):
    """Only acquisition time, site list, and satellite platform may differ."""
    trained=bundle['schema']['config']
    for key in ['leads','cycles','issue_delay_minutes','observation_latency_minutes','goes_latency_minutes',
                'label_tolerance_minutes','input_tolerance_minutes','history_hours','hrrr_fields','actions']:
        if digest(config[key])!=digest(trained[key]):
            raise ValueError('Configuration is incompatible with the model: '+key)
    left,right=copy.deepcopy(config['satellite']),copy.deepcopy(trained['satellite'])
    for item in [left,right]:
        item.pop('number',None);item.pop('prepare_action_crops',None)
    if digest(left)!=digest(right):
        raise ValueError('Satellite geometry/preprocessing differs from training')


def fetch_inputs(config, station, cache, now=None, providers=None):
    """Build one cycle; request station observations only through initialization."""
    from .pipeline import _build
    from .providers import IEM
    now=utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    init=latest_cycle(config,now)
    c=copy.deepcopy(config)
    c.update(start=init.isoformat(),end=(init+pd.Timedelta(hours=1)).isoformat(),cycles=[init.hour],splits=None,strict=True)
    output=Path(cache)/digest({'config':c,'station':station})
    output.mkdir(parents=True,exist_ok=True)
    station_path=output/'station.csv'
    pd.DataFrame([station]).to_csv(station_path,index=False)
    c['stations']=str(station_path.resolve())
    class InputIEM(IEM):
        def observations(self, station, start, end):
            return super().observations(station,start,min(utc(end),init))
    with locked(output):
        provided=providers(output,c) if providers else {'iem':InputIEM(output/'raw/iem')}
        report=_build(c,output,**provided)
        rows=pd.read_parquet(output/'samples.parquet')
        if report['violations'] or report['failures'] or not np.isfinite(rows[['obs_current_c','hrrr_t2m_c']]).all().all():
            raise ValueError('Required current inputs are incomplete; see '+str(output/'report.json'))
        write_json(output/'live.json',{'retrieved_utc':now.isoformat(),'init_utc':init.isoformat(),
                   'station':station,'label':'Forecast-cycle inputs; no verification observations required'})
    return output


def inspect_inputs(root):
    store=FeatureStore(root,'mos')
    frame=store.rows.reset_index(drop=True).merge(store.optional.reset_index(),on='sample_id')
    return frame


def forecast(bundle, root, leads, overrides=None, budget=3.15, strategy='learned', station_id=None):
    """Use training scalers, never fit a normalizer on current observations."""
    if not math.isfinite(budget) or budget<0:raise ValueError('Budget must be finite and nonnegative')
    store=FeatureStore(root,bundle['config']['mode'])
    schema=bundle['schema']
    # Single-cycle snapshots have different dates/station paths and fingerprints.
    # Validate the input feature schema explicitly before attaching the saved scaler.
    for key in ['core_features','mos_extra','actions']:
        if digest(store.manifest[key])!=digest(schema[key]):raise ValueError('Input feature schema differs from the model')
    current=copy.deepcopy(store.manifest['config'])
    current['cycles']=schema['config']['cycles']
    validate_live_config(current,bundle)
    store.normalizer=Standardizer(bundle['normalizer'])
    if station_id is None and store.rows.station_id.nunique()!=1:
        raise ValueError('Choose a station for this snapshot')
    if station_id is not None and station_id not in set(store.rows.station_id):
        raise ValueError('Selected station is absent from this snapshot; import its inputs first')
    if not leads or set(leads)-set(schema['config']['leads']):raise ValueError('Select supported forecast leads')
    allowed=set(store.columns+list(store.optional.columns))-{'station_lat','station_lon','lead_h','hour_sin','hour_cos','year_sin','year_cos'}
    for sid, edits in (overrides or {}).items():
        if sid not in store.rows.index:raise ValueError('Unknown input sample')
        for name,value in edits.items():
            if name not in allowed or not math.isfinite(value):raise ValueError('Invalid edited feature: '+name)
            if name.endswith('_c') and not -100<=value<=65:raise ValueError('Temperature outside -100 to 65 C')
            if name=='obs_age_minutes' and value<0:raise ValueError('Observation age cannot be negative')
            target=store.rows if name in store.rows.columns else store.optional
            target.loc[sid,name]=value
    model=make_model(store,bundle['config']).cpu().eval()
    model.load_state_dict(bundle['model'])
    answers=[]
    for sid,row in store.rows.iterrows():
        if station_id is not None and row.station_id!=station_id:continue
        if row.lead_h not in leads:continue
        if not np.isfinite(row.obs_current_c) or (store.mode=='mos' and not np.isfinite(row.hrrr_t2m_c)):
            raise ValueError('Current temperature/HRRR is missing')
        cfg=bundle['config']
        maximum=sum(store.actions[a]['cost_units'] for a in cfg['actions'])
        session=AcquisitionSession(store,sid,budget if cfg['acquisition']=='random' else maximum)
        if cfg['acquisition']=='random':
            state=acquire(session,model,'cpu',strategy,bundle['policy'],np.random.default_rng(cfg['seed']))
        else:
            for action in cfg['actions']:session.acquire(action)
            state=session.observation()
        value=predict(model,state,'cpu')
        corrected=value+float(row.hrrr_t2m_c) if store.mode=='mos' else value
        if not math.isfinite(corrected):raise ValueError('Nonfinite forecast')
        answers.append({'sample_id':sid,'station_id':row.station_id,'init_utc':str(row.init_utc),
          'valid_utc':str(row.valid_utc),'lead_h':int(row.lead_h),'prediction_c':corrected,
          'hrrr_t2m_c':float(row.hrrr_t2m_c),'correction_c':value if store.mode=='mos' else None,
          'optional_cost':session.spent,'edited':bool(overrides),'mode':store.mode})
    if not answers:raise ValueError('No samples match the selected leads')
    return answers
