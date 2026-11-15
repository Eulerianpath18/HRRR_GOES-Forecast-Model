"""Monthly Slurm tasks, durable initialization progress, and dataset acceptance."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .core import (load_config, stations, initializations, utc, digest, write_json,
                   write_parquet, atomic_bytes)
from .pipeline import build, audit
from .runtime import locked, sha256, status


FILES = ('samples.parquet', 'optional.parquet', 'manifest.json')


def plan(config_path, output):
    output = Path(output).resolve()
    config = load_config(config_path)
    station_rows = stations(config['stations'])
    identity = copy.deepcopy(config)
    identity['stations'] = station_rows
    fingerprint = digest({'schema': 3, 'config': identity})
    with locked(output):
        destination = output / 'plan.json'
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing['fingerprint'] != fingerprint:
                raise ValueError('Archive plan differs. Choose another output directory.')
            return existing
        atomic_bytes(output / 'stations.csv', Path(config['stations']).read_bytes())
        config['stations'] = 'stations.csv'
        start, end = utc(config['start']), utc(config['end'])
        months = pd.date_range(start.normalize().replace(day=1), end, freq='MS')
        shards = []
        for month in months:
            left, right = max(start, month), min(end, month + pd.offsets.MonthBegin())
            if left < right:
                shards.append({'id': month.strftime('%Y-%m'), 'start': left.isoformat(), 'end': right.isoformat()})
        result = {'version': 1, 'fingerprint': fingerprint, 'config': config, 'shards': shards}
        write_json(destination, result)
        status(output, 'planned', tasks=len(shards))
        return result


def read_plan(root):
    root = Path(root).resolve()
    result = json.loads((root / 'plan.json').read_text())
    config = copy.deepcopy(result['config'])
    config['stations'] = str(root / 'stations.csv')
    return result, config


def build_part(root, index, retry_failed=False, provider_factory=None, rebuild=False):
    schedule, _ = read_plan(root)
    if not 0 <= index < len(schedule['shards']):
        raise ValueError('Import task index is out of range')
    with locked(Path(root) / 'task_locks' / schedule['shards'][index]['id']):
        return _build_part(root, index, retry_failed, provider_factory, rebuild)


def _build_part(root, index, retry_failed=False, provider_factory=None, rebuild=False):
    root = Path(root).resolve()
    schedule, config = read_plan(root)
    shard = schedule['shards'][index]
    destination = root / 'shards' / shard['id']
    marker = destination / 'complete.json'
    if rebuild:
        marker.unlink(missing_ok=True)
        for checkpoint in (destination / 'progress').glob('*/complete.json'):
            checkpoint.unlink()
    if marker.exists():
        saved = json.loads(marker.read_text())
        intact = saved['plan'] == schedule['fingerprint'] and all(
            (destination / name).exists() and sha256(destination / name) == value
            for name, value in saved['hashes'].items())
        if intact and not (retry_failed and saved['report']['failures']):
            return saved['report']
    # Re-import invalidates acceptance; re-finalize before starting new experiments.
    (root / 'READY.json').unlink(missing_ok=True)
    config['sample_anchor'] = config['start']
    config.update(start=shard['start'], end=shard['end'])
    providers = provider_factory(destination, config) if provider_factory else {}
    # build() owns the exclusive shard lock and initialization checkpoints.
    report = build(config, destination, **providers)
    with locked(destination):
        write_json(marker, {'plan': schedule['fingerprint'], 'report': report,
                            'hashes': {name: sha256(destination / name) for name in FILES}})
    return report


def quality(root, policy=None, deep=False):
    root = Path(root)
    policy = policy or {}
    report = audit(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    config = manifest['config']
    frame = pd.read_parquet(root / 'samples.parquet')
    optional = pd.read_parquet(root / 'optional.parquet')
    violations = list(report['violations'])
    expected = pd.MultiIndex.from_product([
        [s['station_id'] for s in stations(root / 'stations.csv')], initializations(config), config['leads']],
        names=['station_id', 'init_utc', 'lead_h']) if (root / 'stations.csv').exists() else None
    if expected is not None:
        actual = pd.MultiIndex.from_frame(frame[['station_id', 'init_utc', 'lead_h']])
        if actual.duplicated().any() or len(actual.difference(expected)) or len(expected.difference(actual)):
            violations.append('missing, unexpected, or duplicate planned sample keys')
    for name in ['observed_t2m_c', 'hrrr_t2m_c', 'obs_current_c']:
        values = frame[name].dropna()
        if not np.isfinite(values).all() or not values.between(-100, 65).all():
            violations.append('invalid temperature range: ' + name)
    frame = frame.merge(optional, on='sample_id', validate='one_to_one')
    frame['label_present'] = np.isfinite(frame.observed_t2m_c)
    frame['hrrr_present'] = np.isfinite(frame.hrrr_t2m_c)
    frame['current_present'] = np.isfinite(frame.obs_current_c)
    frame['overview_present'] = frame.overview_path.ne('')
    required = policy.get('required_splits', ['train', 'val', 'test', 'test_spatial'])
    required_frame = frame[frame.split.isin(required)]
    groups = []
    for key, part in required_frame.groupby(['split', 'station_id', 'lead_h']):
        entry = dict(zip(['split', 'station_id', 'lead_h'], key))
        entry['rows'] = len(part)
        for name, default in [('label', .9), ('hrrr', .99), ('current', .9), ('overview', .95)]:
            value = float(part[name + '_present'].mean())
            entry[name + '_fraction'] = value
            if value < policy.get('min_' + name + '_fraction', default):
                violations.append(f'low {name} coverage: {key} ({value:.3f})')
        groups.append(entry)
    for split in required:
        if not len(required_frame[required_frame.split == split]):
            violations.append('empty required split: ' + split)
    failures = json.loads((root / 'failures.json').read_text()) if (root / 'failures.json').exists() else []
    if len(failures) > policy.get('max_source_failures', 0):
        violations.append(f'{len(failures)} recorded source failures exceed acceptance policy')
    image_paths = set(frame.overview_path[frame.overview_path.ne('')])
    for sample in manifest['samples'].values():
        image_paths.update(image['path'] for image in sample['satellite'].values())
        image_paths.update(sample.get('prepared_actions', {}).values())
    files = {}
    for relative in sorted(image_paths):
        path = root / relative
        if not path.is_file():
            violations.append('missing image: ' + relative)
            continue
        files[relative] = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                           'sha256': sha256(path)}
        if deep:
            try:
                with np.load(path, allow_pickle=False) as data:
                    array = data['image_k']
                    if array.ndim != 3 or array.shape[0] != 1 or not np.isfinite(array).any():
                        raise ValueError('invalid image shape or no finite pixels')
                    if np.isinf(array).any():
                        raise ValueError('infinite image values')
                    if 'footprint_km' in data and np.isfinite(array).mean() < config['satellite']['min_valid_fraction']:
                        raise ValueError('insufficient valid pixels')
            except Exception as exc:
                violations.append(f'corrupt image: {relative}: {exc}')
    report.update(violations=sorted(set(violations)), groups=groups, source_failures=len(failures),
                  checked_image_files=len(files), deep=deep, policy=policy)
    write_json(root / 'quality.json', report)
    pd.DataFrame(groups).to_csv(root / 'coverage.csv', index=False)
    return report, files


def finalize(root, policy_path, deep=True):
    root = Path(root).resolve()
    schedule, config = read_plan(root)
    policy = yaml.safe_load(Path(policy_path).read_text())
    with locked(root):
        (root / 'READY.json').unlink(missing_ok=True)
        frames, extras, samples, resources, failures = [], [], {}, {}, []
        schema = None
        for shard in schedule['shards']:
            directory = root / 'shards' / shard['id']
            marker = directory / 'complete.json'
            if not marker.exists():
                raise ValueError('Incomplete import task: ' + shard['id'])
            saved = json.loads(marker.read_text())
            if saved['plan'] != schedule['fingerprint'] or any(
                    sha256(directory / name) != saved['hashes'][name] for name in FILES):
                raise ValueError('Changed or corrupt import task: ' + shard['id'])
            frame = pd.read_parquet(directory / 'samples.parquet')
            prefix = 'shards/' + shard['id'] + '/'
            frame['overview_path'] = frame.overview_path.map(lambda value: prefix + value if value else '')
            frames.append(frame)
            extras.append(pd.read_parquet(directory / 'optional.parquet'))
            part = json.loads((directory / 'manifest.json').read_text())
            if schema is None:
                schema = {k: v for k, v in part.items() if k not in ['samples', 'resources']}
            for sid, metadata in part['samples'].items():
                if sid in samples:
                    raise ValueError('Duplicate sample between shards: ' + sid)
                for image in metadata['satellite'].values():
                    image['path'] = prefix + image['path']
                metadata['prepared_actions'] = {name: prefix + path for name,path in metadata.get('prepared_actions',{}).items()}
                samples[sid] = metadata
            resources.update(part['resources'])
            failures.extend(json.loads((directory / 'failures.json').read_text()))
        schema.update(config=config, fingerprint=schedule['fingerprint'], samples=samples, resources=resources)
        write_parquet(root / 'samples.parquet', pd.concat(frames, ignore_index=True))
        write_parquet(root / 'optional.parquet', pd.concat(extras, ignore_index=True))
        write_json(root / 'manifest.json', schema)
        write_json(root / 'failures.json', failures)
        write_json(root / 'build_config.json', {'fingerprint': schedule['fingerprint'], 'config': config,
                                              'stations': stations(config['stations'])})
        report, image_files = quality(root, policy, deep)
        if report['violations']:
            status(root, 'rejected', violations=report['violations'])
            raise ValueError('Dataset rejected. Read quality.json and retry failed import tasks.')
        hashes = {name: sha256(root / name) for name in FILES}
        write_json(root / 'READY.json', {'fingerprint': schedule['fingerprint'], 'hashes': hashes,
                                        'data_id': digest({'tables': hashes, 'images': {k:v['sha256'] for k,v in image_files.items()}}), 'images': image_files, 'quality': report})
        status(root, 'ready', rows=report['rows'])
        return report


def verify_ready(root):
    root = Path(root)
    path = root / 'READY.json'
    if not path.exists():
        raise ValueError('Dataset is not accepted. Run finalize first.')
    ready = json.loads(path.read_text())
    for name, value in ready['hashes'].items():
        if sha256(root / name) != value:
            raise ValueError('Dataset changed after acceptance: ' + name)
    for name, signature in ready['images'].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != signature['size']:
            raise ValueError('Image changed after acceptance: ' + name)
        if path.stat().st_mtime_ns != signature['mtime_ns'] and sha256(path) != signature['sha256']:
            raise ValueError('Image content changed after acceptance: ' + name)
    return ready
