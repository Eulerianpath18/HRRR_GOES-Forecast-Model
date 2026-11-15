"""Versioned experiment plans and validation-selected comparisons."""
import copy
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from .core import digest, write_json, write_parquet
from .archive import verify_ready
from .runtime import locked, sha256, source_id
from .torch_data import FeatureStore
from . import training, baselines, policy, evaluation


def prepare(root, config_path, output):
    output = Path(output).resolve()
    ready = verify_ready(root)
    settings = yaml.safe_load(Path(config_path).read_text())
    validate_settings(settings)
    identifier = digest({'data': ready['data_id'], 'settings': settings, 'source_id': source_id()})
    with locked(output):
        path = output / 'experiments.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['identity'] != identifier:
                raise ValueError('Experiment plan changed. Choose a new run directory.')
            return previous
        runs = []
        for mode in settings['modes']:
            store = FeatureStore(root, mode)
            numeric = [name for name, action in store.actions.items() if action['kind'] == 'tabular']
            all_actions = list(store.actions)
            for profile, definition in settings['profiles'].items():
                for rate in settings['learning_rates']:
                    for seed in settings['seeds']:
                        config = copy.deepcopy(settings['training'])
                        config.update(mode=mode, kind=definition['kind'], acquisition=definition['acquisition'],
                            actions=all_actions if definition['features']=='all' else numeric,
                            learning_rate=rate, seed=seed, policy_fraction=settings['policy_fraction'],
                            budgets=settings['budgets'])
                        run_id = f'{mode}_{profile}_lr{rate:g}_seed{seed}'
                        runs.append({'id': run_id, 'profile': profile, 'config': config})
        plan = {'identity': identifier, 'data_id': ready['data_id'], 'source_id': source_id(), 'settings': settings, 'runs': runs}
        write_json(path, plan)
        return plan


def validate_settings(settings):
    for name in ['seeds', 'modes', 'learning_rates', 'budgets', 'test_splits']:
        values = settings[name]
        if not values or len(set(values)) != len(values):
            raise ValueError('Empty or repeated experiment setting: ' + name)
    if set(settings['modes']) - {'direct', 'mos'} or set(settings['test_splits']) - {'test', 'test_spatial'}:
        raise ValueError('Unsupported model mode or test split')
    if not 0 < settings['policy_fraction'] < .5:
        raise ValueError('policy_fraction must be between 0 and .5')
    if any(not np.isfinite(x) or x <= 0 for x in settings['learning_rates']):
        raise ValueError('Learning rates must be finite and positive')
    if any(not np.isfinite(x) or x < 0 for x in settings['budgets']) or max(settings['budgets']) <= 0:
        raise ValueError('Budgets must be finite, nonnegative, and include a positive value')
    train = settings['training']
    for name in ['batch_size', 'epochs', 'patience', 'checkpoint_batches', 'threads', 'embedding']:
        if type(train[name]) is not int or train[name] < 1:
            raise ValueError('Training setting must be a positive integer: ' + name)
    if type(train['workers']) is not int or train['workers'] < 0:
        raise ValueError('workers must be a nonnegative integer')
    if not settings['profiles']:
        raise ValueError('No experiment profiles')
    for profile in settings['profiles'].values():
        if profile['kind'] not in ('mlp', 'hybrid') or profile['features'] not in ('numeric', 'all') or profile['acquisition'] not in ('fixed', 'random'):
            raise ValueError('Invalid experiment profile')


def read_experiments(output):
    plan = json.loads((Path(output) / 'experiments.json').read_text())
    if plan['source_id'] != source_id():
        raise ValueError('Implementation changed. Restore the original code or use a new experiment directory.')
    return plan


def train_task(root, output, index, device):
    plan = read_experiments(output)
    item = plan['runs'][index]
    return training.run(root, Path(output) / 'training' / item['id'], item['config'], device)


def select(output):
    output = Path(output)
    plan = read_experiments(output)
    measurements = []
    for item in plan['runs']:
        directory = output / 'training' / item['id']
        path = directory / 'status.json'
        if not path.exists() or json.loads(path.read_text())['state'] != 'complete':
            raise ValueError('Training task incomplete: ' + item['id'])
        status = json.loads(path.read_text())
        if status.get('data_id') != plan['data_id'] or status.get('config_id') != digest(item['config']):
            raise ValueError('Training task identity mismatch: ' + item['id'])
        if not (directory / 'best.pt').exists() or sha256(directory / 'best.pt') != status.get('best_sha256'):
            raise ValueError('Missing or changed best checkpoint: ' + item['id'])
        measurements.append({'id': item['id'], 'mode': item['config']['mode'], 'profile': item['profile'],
                             'rate': item['config']['learning_rate'], 'seed': item['config']['seed'],
                             'mse': status['best_validation_mse']})
    frame = pd.DataFrame(measurements)
    selected = []
    for (mode, profile), group in frame.groupby(['mode','profile']):
        means = group.groupby('rate').mse.mean()
        rate = float(means.idxmin())
        selected.extend(group[group.rate==rate].id.tolist())
    frame.to_csv(output / 'validation_selection.csv', index=False)
    result = {'selected_ids': selected, 'criterion': 'Lowest mean validation MSE across all configured seeds',
              'data_id': plan['data_id'], 'identity': plan['identity']}
    path = output / 'selected.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Frozen model selection differs. Use a new experiment directory.')
    write_json(path, result)
    return result


def fit_baselines(root, output):
    plan = read_experiments(output)
    settings = {**plan['settings']['baselines'], 'policy_fraction': plan['settings']['policy_fraction']}
    baselines.fit(root, Path(output) / 'baselines', settings)


def evaluate_task(root, output, index, device):
    output = Path(output)
    plan = read_experiments(output)
    chosen = json.loads((output / 'selected.json').read_text())
    run_id = chosen['selected_ids'][index]
    item = next(r for r in plan['runs'] if r['id']==run_id)
    checkpoint = output / 'training' / run_id / 'best.pt'
    settings = plan['settings']
    policy_path = None
    if item['config']['acquisition'] == 'random':
        options = {**settings['policy'], 'budgets': settings['budgets']}
        policy_path = policy.fit(root, checkpoint, output / 'policies' / run_id, options, device)
    records = []
    for split in settings['test_splits']:
        if policy_path:
            for strategy in ['static','random','learned']:
                for budget in settings['budgets']:
                    task = f'{split}_{strategy}_{budget:g}'
                    file = evaluation.evaluate(root, checkpoint, output/'evaluation'/run_id/task, split,
                                                strategy, budget, policy_path, device)
                    records.append({'path': str(file.relative_to(output)), 'split': split,
                                    'model': item['config']['mode']+'/'+item['profile']+'/'+strategy,
                                    'seed': item['config']['seed'], 'sha256': sha256(file)})
        else:
            file = evaluation.evaluate(root, checkpoint, output/'evaluation'/run_id/split, split, device=device)
            records.append({'path': str(file.relative_to(output)), 'split': split,
                            'model': item['config']['mode']+'/'+item['profile'],
                            'seed': item['config']['seed'], 'sha256': sha256(file)})
    write_json(output/'evaluation'/run_id/'index.json', records)


def report(root, output):
    output = Path(output)
    plan = read_experiments(output)
    chosen = json.loads((output / 'selected.json').read_text())
    frames = []
    schema = json.loads((Path(root)/'manifest.json').read_text())
    numeric_cost = {mode: sum(a['cost_units'] for a in schema['actions']
                    if a['kind']=='tabular' and mode in a['modes']) for mode in ('direct','mos')}
    for run_id in chosen['selected_ids']:
        records = json.loads((output/'evaluation'/run_id/'index.json').read_text())
        for record in records:
            path = output/record['path']
            if sha256(path) != record['sha256']:
                raise ValueError('Evaluation artifact changed: '+str(path))
            frame = pd.read_parquet(path)
            frame['model'], frame['seed'] = record['model'], record['seed']
            frames.append(frame)
    for split in plan['settings']['test_splits']:
        frame, predictions = baselines.predictions(root, output/'baselines'/'baselines.joblib', split)
        for name, values in predictions.items():
            part = frame[['sample_id','station_id','init_utc','valid_utc','lead_h','split','observed_t2m_c','hrrr_t2m_c']].copy()
            part['prediction_c'], part['model'], part['seed'] = values, name, -1
            mode = 'mos' if name.startswith('mos_') else 'direct' if name.startswith('direct_') else None
            part['optional_cost'] = numeric_cost[mode] if mode else 0.
            part['budget'] = part['optional_cost']
            frames.append(part.reset_index(drop=True))
    results = pd.concat(frames, ignore_index=True)
    # Every comparison must use the identical paired cohort within a test split.
    for split, all_rows in results.groupby('split'):
        reference = set(all_rows[all_rows.model=='raw_hrrr'].sample_id)
        for key, group in all_rows.groupby(['model','seed','budget']):
            if group.sample_id.duplicated().any() or set(group.sample_id) != reference:
                raise ValueError(f'Inconsistent comparison cohort: {split}/{key}')
    destination = output/'results'
    write_parquet(destination/'predictions.parquet', results)
    evaluation.make_report(destination/'predictions.parquet', destination, plan['settings']['bootstrap_replicates'])
    return {'rows': len(results), 'results': str(destination)}
