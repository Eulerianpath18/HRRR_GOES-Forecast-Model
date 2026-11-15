"""Persistence, seasonal climatology, linear models, and subset Gaussian processes."""
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from .archive import verify_ready
from .core import atomic_bytes, write_json, digest
from .runtime import locked, sha256, status, source_id, StopRequest
from .torch_data import FeatureStore, Standardizer
from .experiment_data import cohort, training_partitions


def matrix(store, ids):
    frame = store.rows.loc[ids]
    columns = list(store.columns)
    parts = [frame[columns]]
    for action in store.actions.values():
        if action['kind'] == 'tabular':
            parts.append(store.optional.loc[ids, action['columns']])
    table = pd.concat(parts, axis=1)
    table = table.loc[:, ~table.columns.duplicated()]
    values = table.to_numpy(dtype=float)
    mask = np.isfinite(values)
    for index, column in enumerate(table.columns):
        mean, scale = store.normalizer.values[column]
        values[:, index] = np.where(mask[:, index], (values[:, index] - mean) / scale, 0.)
    return np.c_[values, mask.astype(float)]


def fit(root, output, settings, stop_after_models=None):
    root, output = Path(root), Path(output)
    ready = verify_ready(root)
    identity = digest({'data': ready['data_id'], 'settings': settings, 'source_id': source_id()})
    with locked(output), StopRequest() as stop:
        marker = output / 'complete.json'
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous['identity'] != identity:
                raise ValueError('Baseline configuration changed')
            if sha256(output / 'baselines.joblib') != previous['sha256']:
                raise ValueError('Baseline artifact changed')
            return
        models = {}
        selection = []
        progress = output / 'progress.joblib'
        if progress.exists():
            saved = joblib.load(progress)
            if saved['identity'] != identity:
                raise ValueError('Baseline progress belongs to another configuration or dataset')
            models, selection = saved['models'], saved['selection']
        completed = 0

        def save_progress():
            buffer = io.BytesIO()
            joblib.dump({'identity': identity, 'models': models, 'selection': selection}, buffer)
            atomic_bytes(progress, buffer.getvalue())
            status(output, 'fitting', completed_models=list(models))

        for mode in ['direct', 'mos']:
            store = FeatureStore(root, mode)
            train_ids, _, boundary = training_partitions(store, settings['policy_fraction'])
            val_ids = cohort(store, 'val')
            scaler = Standardizer.fit(root, sample_ids=train_ids)
            store.normalizer = scaler
            x, vx = matrix(store, train_ids), matrix(store, val_ids)
            target = 'observed_t2m_c' if mode == 'direct' else 'error_c'
            y, vy = store.rows.loc[train_ids, target].to_numpy(), store.rows.loc[val_ids, target].to_numpy()
            candidates = [('linear', [LinearRegression()]),
                          ('ridge', [Ridge(alpha=alpha) for alpha in settings['ridge_alphas']])]
            rng = np.random.default_rng(settings['seed'])
            subset = np.sort(rng.choice(len(y), min(len(y), settings['gp_max_train']), replace=False))
            gps = [GaussianProcessRegressor(kernel=RBF(scale) + WhiteKernel(settings['gp_noise']),
                       normalize_y=True, optimizer=None, alpha=1e-6)
                   for scale in settings['gp_length_scales']]
            candidates.append(('gp_subset', gps))
            for name, estimators in candidates:
                key = mode + '_' + name
                if key in models:
                    continue
                best_score, best = float('inf'), None
                for estimator in estimators:
                    indices = subset if name == 'gp_subset' else np.arange(len(y))
                    estimator.fit(x[indices], y[indices])
                    prediction = estimator.predict(vx)
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError('Nonfinite baseline prediction')
                    score = float(np.mean((prediction - vy) ** 2))
                    if score < best_score:
                        best_score, best = score, estimator
                models[key] = {'mode': mode, 'model': best, 'normalizer': scaler.values,
                               'fingerprint': scaler.fingerprint}
                selection.append({'model': key, 'validation_mse': best_score,
                                  'training_rows': len(subset) if name == 'gp_subset' else len(y),
                                  'predictor_policy_boundary': boundary})
                save_progress()
                completed += 1
                if stop.requested or (stop_after_models and completed >= stop_after_models):
                    status(output, 'interrupted', completed_models=list(models))
                    raise SystemExit(75)
            frame = store.rows.loc[train_ids].copy()
            frame['month'] = frame.valid_utc.dt.month
            frame['hour'] = frame.valid_utc.dt.hour
            if mode == 'direct':
                models['climatology'] = {'station': frame.groupby(['station_id','month','hour']).observed_t2m_c.mean().to_dict(),
                    'regional': frame.groupby(['month','hour']).observed_t2m_c.mean().to_dict(),
                    'global': float(frame.observed_t2m_c.mean())}
            else:
                models['bias_mos'] = {'station': frame.groupby(['station_id','lead_h']).error_c.mean().to_dict(),
                                     'regional': frame.groupby('lead_h').error_c.mean().to_dict()}
            save_progress()
        stream = io.BytesIO()
        joblib.dump({'models': models, 'identity': identity, 'data_id': ready['data_id'], 'source_id': source_id()}, stream)
        atomic_bytes(output / 'baselines.joblib', stream.getvalue())
        write_json(output / 'selection.json', selection)
        write_json(marker, {'identity': identity, 'sha256': sha256(output / 'baselines.joblib')})
        status(output, 'complete')


def predictions(root, artifact, split):
    ready = verify_ready(root)
    bundle = joblib.load(artifact)
    if bundle['data_id'] != ready['data_id']:
        raise ValueError('Baselines belong to another dataset')
    if bundle['source_id'] != source_id():
        raise ValueError('Baselines belong to a different implementation')
    store = FeatureStore(root, 'direct')
    ids = cohort(store, split)
    frame = store.rows.loc[ids]
    result = {'persistence': frame.obs_current_c.to_numpy(), 'raw_hrrr': frame.hrrr_t2m_c.to_numpy()}
    climate, bias = bundle['models']['climatology'], bundle['models']['bias_mos']
    result['climatology'] = np.array([climate['station'].get((r.station_id, r.valid_utc.month, r.valid_utc.hour),
        climate['regional'].get((r.valid_utc.month, r.valid_utc.hour), climate['global'])) for r in frame.itertuples()])
    result['bias_mos'] = np.array([r.hrrr_t2m_c + bias['station'].get((r.station_id, r.lead_h),
        bias['regional'][r.lead_h]) for r in frame.itertuples()])
    for name, entry in bundle['models'].items():
        if 'model' not in entry:
            continue
        current = FeatureStore(root, entry['mode'], Standardizer(entry['normalizer'], entry['fingerprint']))
        values = entry['model'].predict(matrix(current, ids))
        result[name] = values + (frame.hrrr_t2m_c.to_numpy() if entry['mode'] == 'mos' else 0.)
    return frame, result
