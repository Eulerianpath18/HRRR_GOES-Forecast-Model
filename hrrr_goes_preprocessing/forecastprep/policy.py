"""Learn marginal error reductions on a chronological policy-training holdout."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from .core import digest, atomic_bytes, write_json
from .runtime import locked, atomic_torch_save, StopRequest, status, sha256
from .training import load_predictor
from .experiment_data import training_partitions
from .torch_data import AcquisitionSession
from .models import to_device


def batch_state(state, device):
    return to_device(default_collate([state]), device)


@torch.no_grad()
def predict(model, state, device):
    return float(model(batch_state(state, device)).item())


@torch.no_grad()
def features(model, state, device):
    vector = model.representation(batch_state(state, device))[0].cpu().numpy()
    return np.r_[vector, float(state['remaining_budget'])].astype('float64')


def fit(root, checkpoint, directory, settings, device='cpu'):
    directory = Path(directory)
    with locked(directory), StopRequest() as stop:
        store, model, saved = load_predictor(root, checkpoint, device)
        if saved['config']['acquisition'] != 'random':
            raise ValueError('Policy requires a predictor trained with random acquisition masks')
        _, ids, _ = training_partitions(store, saved['config'].get('policy_fraction', .2))
        rng = np.random.default_rng(settings.get('seed', 42))
        if len(ids) > settings['max_samples']:
            ids = sorted(rng.choice(ids, settings['max_samples'], replace=False).tolist())
        identity = digest({'checkpoint': sha256(checkpoint), 'settings': settings, 'ids': ids,
                           'data_id': saved['identity']['data_id']})
        final = directory / 'policy.joblib'
        done = directory / 'complete.json'
        if done.exists():
            result = json.loads(done.read_text())
            if result['identity'] != identity:
                raise ValueError('Policy training configuration changed')
            if not final.exists() or sha256(final) != result['sha256']:
                raise ValueError('Policy artifact changed or missing')
            return final
        progress = directory / 'progress.pt'
        offset = 0
        records = {name: {'x': [], 'y': []} for name in store.actions}
        if progress.exists():
            value = torch.load(progress, map_location='cpu', weights_only=False)
            if value['identity'] != identity:
                raise ValueError('Policy progress belongs to different predictor or settings')
            offset, records = value['offset'], value['records']
        status(directory, 'policy_training', offset=offset)
        for index in range(offset, len(ids)):
            sid = ids[index]
            local_rng = np.random.default_rng(settings.get('seed', 42) + index)
            budget = float(local_rng.choice([b for b in settings['budgets'] if b > 0]))
            session = AcquisitionSession(store, sid, budget)
            target = float(store.rows.loc[sid, 'error_c' if store.mode == 'mos' else 'observed_t2m_c'])
            for _ in range(settings.get('states_per_sample', 3)):
                state = session.observation()
                available = session.available_actions()
                if not available:
                    break
                baseline_loss = (predict(model, state, device) - target) ** 2
                vector = features(model, state, device)
                for name, cost in available.items():
                    candidate = session.observation()
                    candidate['optional'][name] = store.load_action(sid, store.actions[name])
                    candidate['remaining_budget'] -= cost
                    loss = (predict(model, candidate, device) - target) ** 2
                    records[name]['x'].append(vector)
                    records[name]['y'].append(baseline_loss - loss)
                session.acquire(str(local_rng.choice(list(available))))
            offset = index + 1
            if offset % 50 == 0 or stop.requested:
                atomic_torch_save(progress, {'identity': identity, 'offset': offset, 'records': records})
                status(directory, 'policy_training', offset=offset, total=len(ids))
            if stop.requested:
                raise SystemExit(75)
        estimators, mean_gain = {}, {}
        for name, values in records.items():
            if len(values['y']) < 2:
                raise ValueError('Insufficient policy-training acquisitions for ' + name)
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=settings.get('ridge_alpha', 10.)))
            estimator.fit(np.asarray(values['x']), np.asarray(values['y']))
            estimators[name] = estimator
            mean_gain[name] = float(np.mean(values['y']))
        import io
        stream = io.BytesIO()
        joblib.dump({'identity': identity, 'checkpoint_sha256': sha256(checkpoint),
                     'estimators': estimators, 'mean_gain': mean_gain,
                     'settings': settings, 'data_id': saved['identity']['data_id']}, stream)
        atomic_bytes(final, stream.getvalue())
        write_json(done, {'identity': identity, 'sha256': sha256(final),
                          'training_samples': len(ids), 'action_examples': {k:len(v['y']) for k,v in records.items()}})
        status(directory, 'complete')
        return final


def acquire(session, model, device, strategy, policy=None, rng=None):
    rng = rng or np.random.default_rng(0)
    while session.available_actions():
        available = session.available_actions()
        if strategy == 'random':
            chosen = str(rng.choice(list(available)))
        elif strategy == 'static':
            scores = {name: policy['mean_gain'][name] / cost for name, cost in available.items()}
            chosen = max(scores, key=scores.get)
            if scores[chosen] <= 0:
                break
        elif strategy == 'learned':
            vector = features(model, session.observation(), device)
            scores = {name: float(policy['estimators'][name].predict(vector[None])[0]) / cost
                      for name, cost in available.items()}
            chosen = max(scores, key=scores.get)
            if scores[chosen] <= 0:
                break
        else:
            raise ValueError('Unknown acquisition strategy')
        session.acquire(chosen)
    return session.observation()
