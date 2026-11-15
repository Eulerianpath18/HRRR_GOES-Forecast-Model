from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import yaml


def utc(value):
    t = pd.Timestamp(value)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.partial')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_json(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, default=str, allow_nan=False)+'\n').encode())


def write_parquet(path, frame):
    import io
    b = io.BytesIO()
    frame.to_parquet(b, index=False)
    atomic_bytes(path, b.getvalue())


def write_npz(path, **arrays):
    import io
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    atomic_bytes(path, b.getvalue())


def load_config(path):
    path = Path(path)
    c = yaml.safe_load(path.read_text())
    c['stations'] = str((path.parent / c['stations']).resolve())
    validate_config(c)
    return c


def validate_config(c):
    start, end = utc(c['start']), utc(c['end'])
    if not start < end: raise ValueError('start must precede end (end exclusive)')
    if start < utc('2020-12-03'): raise ValueError('This implementation targets HRRR v4 (2020-12-03 onward)')
    if not c['cycles'] or any(type(x) is not int or x not in range(24) for x in c['cycles']):
        raise ValueError('cycles must be integer UTC hours 0..23')
    if not c['leads'] or any(type(x) is not int or not 1 <= x <= 48 for x in c['leads']):
        raise ValueError('leads must be integers 1..48')
    if max(c['leads']) > 18 and not set(c['cycles']) <= {0,6,12,18}:
        raise ValueError('leads >18 require 00/06/12/18 UTC cycles')
    if len(set(c['leads'])) != len(c['leads']) or len(set(c['cycles'])) != len(c['cycles']):
        raise ValueError('Duplicate cycles/leads')
    stride = c.get('sample_every_n_days', 1)
    if type(stride) is not int or stride < 1:
        raise ValueError('sample_every_n_days must be a positive integer')
    for key in ['issue_delay_minutes','observation_latency_minutes','goes_latency_minutes','label_tolerance_minutes']:
        if c[key] < 0: raise ValueError(f'{key} must be nonnegative')
    if c['issue_delay_minutes'] >= min(c['leads'])*60:
        raise ValueError('Issue deadline must precede every valid time')
    g = c['satellite']
    if g['number'] not in [16,18,19]: raise ValueError('Explicit GOES satellite must be 16,18,19')
    if not g['bands'] or not set(g['bands']) <= set(range(7,17)) or len(set(g['bands'])) != len(g['bands']):
        raise ValueError('Initial pipeline supports unique infrared bands 7..16 (kelvin)')
    if g['pixels'] < 16 or g['footprint_km'] <= 0: raise ValueError('Invalid satellite cache geometry')
    if g['overview_band'] not in g['bands']: raise ValueError('overview_band must be downloaded')
    if not 0 < g['min_valid_fraction'] <= 1: raise ValueError('Invalid valid fraction')
    for h in c['leads']:
        if str(h) not in g['lead_footprints_km'] and h not in g['lead_footprints_km']:
            raise ValueError(f'Missing footprint for lead {h}')
        w = g['lead_footprints_km'].get(h, g['lead_footprints_km'].get(str(h)))
        if not 0 < w <= g['footprint_km']: raise ValueError('Lead footprint exceeds cached extent')
    fields = [f['name'] for f in c['hrrr_fields']]
    if 'hrrr_t2m_c' not in fields or len(set(fields)) != len(fields):
        raise ValueError('Unique HRRR field names including hrrr_t2m_c are required')
    if any(not name.startswith('hrrr_') for name in fields):
        raise ValueError('HRRR fields must use hrrr_ prefix')
    if not c['history_hours'] or any(type(h) is not int or h <= 0 for h in c['history_hours']):
        raise ValueError('history_hours must contain positive integer hours')
    if len(set(c['history_hours'])) != len(c['history_hours']):
        raise ValueError('Duplicate history lags')
    optional_columns = (set(fields)-{'hrrr_t2m_c'}) | {f'obs_lag{h}_c' for h in c['history_hours']}
    ids = [a['id'] for a in c['actions']]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate action IDs')
    for a in c['actions']:
        if a['kind'] not in ['tabular','satellite'] or a['cost_units'] <= 0:
            raise ValueError('Invalid action kind/cost')
        if not a['modes'] or not set(a['modes']) <= {'direct','mos'}: raise ValueError('Invalid action modes')
        if a['kind'] == 'tabular':
            if not a['columns'] or not set(a['columns']) <= optional_columns:
                raise ValueError('Tabular action columns must name optional predictors')
            if any(x.startswith('hrrr_') for x in a['columns']) and 'direct' in a['modes']:
                raise ValueError('Future HRRR fields cannot enter the direct model')
        else:
            if a['band'] not in g['bands']: raise ValueError('Action band must be downloaded')
            if a['pixels'] < 2 or a.get('scale',1) <= 0: raise ValueError('Invalid action patch')
            for w in g['lead_footprints_km'].values():
                half = w*a.get('scale',1)/2
                if max(abs(a.get('east_km',0)),abs(a.get('north_km',0)))+half > g['footprint_km']/2:
                    raise ValueError('Action extends beyond cached footprint')
    # Prerequisite graph must be acyclic and available in every applicable mode.
    by_id = {a['id']:a for a in c['actions']}
    def visit(key, stack):
        if key in stack: raise ValueError('Cyclic action prerequisites')
        for req in by_id[key].get('requires',[]):
            if req not in by_id: raise ValueError('Unknown action prerequisite')
            if not set(by_id[key]['modes']) <= set(by_id[req]['modes']):
                raise ValueError('Prerequisite unavailable in action mode')
            visit(req,stack | {key})
    for key in by_id: visit(key,set())
    s = c.get('splits')
    if s and not utc(s['train_end']) < utc(s['val_end']): raise ValueError('Invalid split boundaries')


def stations(path):
    df = pd.read_csv(path, dtype={'station_id':str})
    required = {'station_id','latitude','longitude','iem_network'}
    if not required <= set(df): raise ValueError(f'Stations require {required}')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in df.station_id):
        raise ValueError('Station IDs must be nonempty alphanumeric identifiers')
    if df.station_id.duplicated().any() or df[list(required)].isna().any().any():
        raise ValueError('Station IDs must be unique and metadata complete')
    if not df.latitude.between(-90,90).all() or not df.longitude.between(-180,180).all():
        raise ValueError('Invalid station coordinates')
    return df.to_dict('records')


def initializations(c):
    first_day = utc(c.get('sample_anchor', c['start'])).normalize()
    stride = c.get('sample_every_n_days', 1)
    return [t for t in pd.date_range(utc(c['start']),utc(c['end']),freq='h',inclusive='left')
            if t.hour in c['cycles'] and t.minute == t.second == 0
            and (t.normalize() - first_day).days % stride == 0]


def match_values(obs, times, direction, tolerance_minutes):
    """One station at a time. Preserve left order; missing labels stay missing."""
    times = pd.to_datetime(times, utc=True)
    left = pd.DataFrame({'query_utc':times, '_order':range(len(times))})
    right = obs[['observation_utc','temperature_c']].dropna().sort_values('observation_utc')
    if right.empty:
        left['observation_utc'] = pd.NaT
        left['temperature_c'] = np.nan
        return left
    return pd.merge_asof(left.sort_values('query_utc'), right, left_on='query_utc',
                         right_on='observation_utc', direction=direction,
                         tolerance=pd.Timedelta(minutes=tolerance_minutes)).sort_values('_order')


def split_rows(df, spec, label_tolerance_minutes):
    """Purge labels crossing a boundary; holdout stations are excluded from training."""
    result = pd.Series('unsplit', index=df.index)
    if not spec: return result
    t, v = utc(spec['train_end']), utc(spec['val_end'])
    gap = pd.Timedelta(hours=spec.get('embargo_hours',0))
    # Conservative even when a particular matched report was earlier.
    label_end = df.valid_utc + pd.Timedelta(minutes=label_tolerance_minutes)
    result[:] = 'purged'
    result[(df.init_utc < t) & (label_end < t-gap)] = 'train'
    result[(df.init_utc >= t+gap) & (label_end < v-gap)] = 'val'
    result[df.init_utc >= v+gap] = 'test'
    held = df.station_id.isin(spec.get('heldout_stations',[]))
    result[held] = 'heldout_unused'
    result[held & (df.init_utc >= v+gap)] = 'test_spatial'
    return result
