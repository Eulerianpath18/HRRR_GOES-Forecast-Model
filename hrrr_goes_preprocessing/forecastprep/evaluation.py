"""Resumable inference and paired skill summaries on untouched test samples."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .core import digest, write_parquet, write_json
from .runtime import locked, sha256, StopRequest, status
from .training import load_predictor
from .experiment_data import cohort, ExperimentDataset
from .torch_data import AcquisitionSession
from .models import to_device
from .policy import acquire, predict


def evaluate(root, checkpoint, output, split, strategy='fixed', budget=None, policy_path=None, device='cpu'):
    output = Path(output)
    with locked(output), StopRequest() as stop:
        store, model, saved = load_predictor(root, checkpoint, device)
        config = saved['config']
        ids = cohort(store, split)
        if not ids:
            raise ValueError('No common-cohort evaluation samples in ' + split)
        identity = digest({'data': saved['identity']['data_id'], 'model': sha256(checkpoint),
                           'policy': sha256(policy_path) if policy_path else None,
                           'strategy': strategy, 'budget': budget, 'split': split})
        identity_path = output / 'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())['identity'] != identity:
            raise ValueError('Evaluation inputs changed. Choose another output directory.')
        write_json(identity_path, {'identity': identity})
        marker = output / 'complete.json'
        final = output / 'predictions.parquet'
        if marker.exists():
            done = json.loads(marker.read_text())
            if final.exists() and sha256(final) == done['sha256']:
                return final
            raise ValueError('Completed predictions are missing or corrupt')
        policy = joblib.load(policy_path) if policy_path else None
        if policy and (policy['checkpoint_sha256'] != sha256(checkpoint) or policy['data_id'] != saved['identity']['data_id']):
            raise ValueError('Policy belongs to another predictor or dataset')
        parts = []
        for offset in range(0, len(ids), 128):
            file = output / 'parts' / f'{offset:08d}.parquet'
            selected = ids[offset:offset + 128]
            if file.exists():
                previous = pd.read_parquet(file)
                if previous.sample_id.tolist() != selected or not np.isfinite(previous.prediction_c).all():
                    raise ValueError('Invalid evaluation progress file')
                parts.append(file)
                continue
            if stop.requested:
                status(output, 'interrupted', offset=offset)
                raise SystemExit(75)
            values, costs, seconds, bytes_used, unknown = [], [], [], [], []
            if strategy == 'fixed':
                data = ExperimentDataset(store, selected, config['actions'])
                loader = DataLoader(data, batch_size=config['batch_size'], num_workers=0)
                with torch.no_grad():
                    for batch in loader:
                        start = time.perf_counter()
                        estimates = model(to_device(batch['inputs'], device)).cpu().numpy()
                        elapsed = time.perf_counter() - start
                        values.extend(estimates.tolist())
                        seconds.extend([elapsed / len(estimates)] * len(estimates))
                costs = [data.maximum] * len(selected)
                bytes_used, unknown = [np.nan] * len(selected), [True] * len(selected)
            else:
                for sid in selected:
                    started = time.perf_counter()
                    session = AcquisitionSession(store, sid, float(budget))
                    rng = np.random.default_rng(int(digest({'sid': sid, 'seed': config['seed']})[:16], 16))
                    state = acquire(session, model, device, strategy, policy, rng)
                    values.append(predict(model, state, device))
                    costs.append(session.spent)
                    seconds.append(time.perf_counter() - started)
                    bytes_used.append(session.source_bytes)
                    unknown.append(bool(session.unknown_size_resources))
                    if session.spent > float(budget) + 1e-6:
                        raise AssertionError('Acquisition budget exceeded')
            frame = store.rows.loc[selected, ['sample_id','station_id','init_utc','valid_utc','lead_h',
                                              'split','observed_t2m_c','hrrr_t2m_c']].copy()
            estimates = np.asarray(values)
            if not np.isfinite(estimates).all():
                raise FloatingPointError('Nonfinite evaluation predictions')
            frame['prediction_c'] = estimates + (frame.hrrr_t2m_c.to_numpy() if store.mode == 'mos' else 0.)
            frame['optional_cost'] = costs
            frame['budget'] = float(budget) if budget is not None else max(costs, default=0.)
            frame['seconds'] = seconds
            frame['source_bytes'] = bytes_used
            frame['unknown_bytes'] = unknown
            frame['timing_scope'] = 'model_forward_per_batch' if strategy == 'fixed' else 'full_sequential_policy_and_local_io'
            write_parquet(file, frame.reset_index(drop=True))
            parts.append(file)
            status(output, 'evaluating', completed=min(offset+128, len(ids)), total=len(ids))
        result = pd.concat([pd.read_parquet(file) for file in parts], ignore_index=True)
        write_parquet(final, result)
        write_json(marker, {'identity': identity, 'sha256': sha256(final), 'rows': len(result)})
        status(output, 'complete', rows=len(result))
        return final


def summary(frame, block_replicates=500, seed=42):
    error = frame.prediction_c.to_numpy() - frame.observed_t2m_c.to_numpy()
    raw_error = frame.hrrr_t2m_c.to_numpy() - frame.observed_t2m_c.to_numpy()
    mse, raw_mse = np.mean(error ** 2), np.mean(raw_error ** 2)
    out = {'n': len(frame), 'mae_c': float(np.mean(abs(error))), 'rmse_c': float(np.sqrt(mse)),
           'bias_c': float(np.mean(error)), 'raw_hrrr_rmse_c': float(np.sqrt(raw_mse)),
           'mse_skill_vs_hrrr': float(1-mse/raw_mse) if raw_mse else None,
           'mean_optional_cost': float(frame.optional_cost.mean())}
    # Resample whole seven-day time blocks, retaining all stations/leads together.
    # pandas may store datetimes in ns or us; calendar flooring is unit independent.
    block = pd.to_datetime(frame.init_utc, utc=True).dt.floor('7D').to_numpy()
    groups = pd.DataFrame({'block': block, 'se': error**2, 'raw_se': raw_error**2, 'n': 1}).groupby('block').sum()
    out['time_blocks'] = len(groups)
    if len(groups) >= 8 and block_replicates:
        rng = np.random.default_rng(seed)
        stats = groups.to_numpy()
        deltas = []
        for _ in range(block_replicates):
            draw = stats[rng.integers(len(stats), size=len(stats))].sum(axis=0)
            deltas.append(np.sqrt(draw[1]/draw[2])-np.sqrt(draw[0]/draw[2]))
        out['rmse_improvement_ci_low_c'], out['rmse_improvement_ci_high_c'] = map(float, np.quantile(deltas, [.025,.975]))
    else:
        out['rmse_improvement_ci_low_c'] = out['rmse_improvement_ci_high_c'] = None
    return out


def make_report(predictions_path, output, replicates=500):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(predictions_path)
    tables = []
    for keys in [['model','seed','split','budget'], ['model','seed','split','budget','lead_h'],
                 ['model','seed','split','budget','station_id'], ['model','seed','split','budget','season']]:
        if 'season' in keys:
            frame['season'] = frame.valid_utc.dt.month.map({12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'})
        rows = []
        for labels, part in frame.groupby(keys):
            rows.append({**dict(zip(keys, labels)), **summary(part, replicates if len(keys)==4 else 0)})
        name = 'overall' if len(keys)==4 else keys[-1]
        table = pd.DataFrame(rows)
        table.to_csv(output / f'metrics_{name}.csv', index=False)
        tables.append(table)
    means = tables[0].groupby(['model','split','budget']).agg(mean_rmse_c=('rmse_c','mean'),
            seed_std_rmse_c=('rmse_c','std'), seed_count=('seed','nunique'), mean_cost=('mean_optional_cost','mean')).reset_index()
    means.to_csv(output / 'metrics_across_seeds.csv', index=False)
    for split in frame.split.unique():
        data = tables[1][tables[1].split == split]
        # Full fixed-input models and numerical baselines provide a readable lead plot.
        data = data[~data.model.str.contains('/learned|/random|/static', regex=True)]
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, part in data.groupby('model'):
            line = part.groupby('lead_h').rmse_c.mean()
            ax.plot(line.index, line.values, marker='o', label=name)
        ax.set(xlabel='Forecast lead (hours)', ylabel='Temperature RMSE (C)', title=split)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output / f'rmse_by_lead_{split}.png', dpi=160)
        plt.close(fig)
        data = tables[0][(tables[0].split == split) & tables[0].model.str.contains('/learned|/random|/static', regex=True)]
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, part in data.groupby('model'):
            line = part.groupby('budget').agg(cost=('mean_optional_cost','mean'), rmse=('rmse_c','mean'))
            ax.plot(line.cost, line.rmse, marker='o', label=name)
        ax.set(xlabel='Mean acquired optional cost (proxy units)', ylabel='RMSE (C)', title=split)
        if len(data):
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f'budget_curve_{split}.png', dpi=160)
        plt.close(fig)
    write_json(output / 'report.json', {'prediction_rows': len(frame), 'replicates': replicates,
        'uncertainty': 'Paired seven-day block bootstrap over all stations within a block; intervals require at least eight blocks.',
        'seed_summary': 'Means and sample standard deviations across independent training seeds, no best-seed selection.',
        'timing': 'Fixed forward timings exclude input loading. Sequential policy timings include local IO; do not compare these as equal latency scopes.'})
    (output / 'RESULTS.md').write_text('# Experiment outputs\n\n'
        'Open metrics_overall.csv and metrics_across_seeds.csv for accuracy and optional cost. '
        'The lead, station, and season tables show where errors occur. '
        'Positive RMSE-improvement intervals indicate gains over raw HRRR on the common sample cohort. '
        'Training seeds and paired time-block uncertainty measure different sources of variation. '
        'Feature costs are configured proxies and require hardware calibration before claims about operational compute savings.\n')
    return means
