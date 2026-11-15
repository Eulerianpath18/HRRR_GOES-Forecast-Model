from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .core import utc, digest, stations, initializations, match_values, split_rows, write_json, write_parquet, write_npz, validate_config
from .providers import IEM, HRRR
from .satellite import GOES, crop
from .runtime import locked, sha256, StopRequest, status

LOG = logging.getLogger(__name__)
CORE = ['station_lat','station_lon','lead_h','hour_sin','hour_cos','year_sin','year_cos','obs_current_c','obs_age_minutes']


def prepared_crop(root, image, action, geometry, lead, source_arrays):
    """Content-addressed small arrays avoid repeated large-patch IO during training."""
    source = image['path']
    if source not in source_arrays:
        with np.load(root/source, allow_pickle=False) as data:
            source_arrays[source] = (data['image_k'], float(data['footprint_km']), sha256(root/source))
    array, footprint, checksum = source_arrays[source]
    widths = geometry['lead_footprints_km']
    width = widths.get(str(lead), widths.get(lead)) * action.get('scale', 1.)
    key = digest({'source_sha256': checksum, 'width': width, 'pixels': action['pixels'],
                  'east': action.get('east_km', 0), 'north': action.get('north_km', 0)})
    path = root/'crops'/(key+'.npz')
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                if saved['image_k'].shape == (1,action['pixels'],action['pixels']):
                    return path.relative_to(root).as_posix()
        except Exception:
            pass
    values = crop(array, footprint, width, action['pixels'], action.get('east_km',0), action.get('north_km',0))
    write_npz(path, image_k=values)
    return path.relative_to(root).as_posix()


def build(c, output, *, iem=None, hrrr=None, goes=None):
    """Build with atomic initialization checkpoints and an exclusive writer lock."""
    with locked(output), StopRequest() as stop:
        status(output, 'building')
        try:
            result = _build(c, output, iem=iem, hrrr=hrrr, goes=goes, stop=stop)
            status(output, 'complete', **result)
            return result
        except BaseException as exc:
            status(output, 'interrupted' if isinstance(exc, SystemExit) else 'failed', error=str(exc))
            raise


def _build(c, output, *, iem=None, hrrr=None, goes=None, stop=None):
    validate_config(c)
    root = Path(output); root.mkdir(parents=True,exist_ok=True)
    stns = stations(c['stations']); inits = initializations(c)
    if not inits: raise ValueError('No selected initializations')
    fingerprint = digest({'config':c,'stations':stns,'schema':2})
    config_path = root/'build_config.json'
    if config_path.exists() and json.loads(config_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output belongs to another configuration; choose a new output directory')
    write_json(config_path,{'fingerprint':fingerprint,'config':c,'stations':stns})
    failures = []
    def failed(stage, context, exc):
        failures.append({'stage':stage,'context':context,'error':f'{type(exc).__name__}: {exc}'})
        write_json(root/'failures.json',failures)
        if c['strict']: raise RuntimeError(f'{stage} failed: {context}') from exc
        LOG.warning('%s %s: %s',stage,context,exc)
    iem = iem or IEM(root/'raw'/'iem')
    hrrr = hrrr or HRRR(root/'raw'/'hrrr',c['max_grid_distance_km'])
    g = c['satellite']
    if g['enabled']: goes = goes or GOES(root,g,c['goes_latency_minutes'])
    observations = {}
    for st in stns:
        sid = st['station_id']
        try:
            obs = iem.observations(st,min(inits)-pd.Timedelta(hours=max(c['history_hours'],default=0)+3),
                                  max(inits)+pd.Timedelta(hours=max(c['leads'])+2))
        except Exception as e:
            failed('IEM',sid,e)
            obs = pd.DataFrame({'observation_utc':pd.Series([],dtype='datetime64[ns, UTC]'),'temperature_c':[]})
        observations[sid] = obs
        write_parquet(root/'observations'/f'{sid}.parquet',obs)
    rows, optional, sample_meta, resource_meta = [],[],{},{}
    for init in inits:
        if stop and stop.requested:
            raise SystemExit(75)
        progress = root / 'progress' / init.strftime('%Y%m%dT%H%MZ')
        marker = progress / 'complete.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            paths = ['samples.parquet', 'optional.parquet', 'metadata.json']
            intact = saved['fingerprint'] == fingerprint and all(
                (progress / name).is_file() and sha256(progress / name) == saved['hashes'][name]
                for name in paths)
            intact = intact and all((root / name).is_file() and (root / name).stat().st_size == size
                                    for name, size in saved['images'].items())
            if intact:
                rows.extend(pd.read_parquet(progress / 'samples.parquet').to_dict('records'))
                optional.extend(pd.read_parquet(progress / 'optional.parquet').to_dict('records'))
                metadata = json.loads((progress / 'metadata.json').read_text())
                sample_meta.update(metadata['samples'])
                resource_meta.update(metadata['resources'])
                LOG.info('Resumed completed initialization %s', init)
                continue
        row_start, optional_start, failure_start = len(rows), len(optional), len(failures)
        sample_keys, resource_keys = set(sample_meta), set(resource_meta)
        LOG.info('Initialization %s',init)
        images = {}
        source_arrays = {}
        if g['enabled']:
            for band in g['bands']:
                try:
                    scene = goes.scene(init,band)
                    resource_meta[scene[1]['resource_id']] = scene[1]
                    for st in stns:
                        try:
                            path = goes.patch(scene,st,band)
                            images[(st['station_id'],band)] = {'path':path,**scene[1]}
                        except Exception as e: failed('GOES patch',f'{init}/{st["station_id"]}/{band}',e)
                except Exception as e: failed('GOES scene',f'{init}/{band}',e)
        for lead in c['leads']:
            field_results = {}
            for field in c['hrrr_fields']:
                try:
                    r = hrrr.field(init,lead,field,stns)
                    field_results[field['name']] = r
                    resource_meta[r['resource_id']] = {k:v for k,v in r.items() if k != 'values'}
                except Exception as e: failed('HRRR',f'{init}/{lead}/{field["name"]}',e)
            for st in stns:
                sid = st['station_id']; valid = init+pd.Timedelta(hours=lead)
                obs = observations[sid]
                target = match_values(obs,[valid],'nearest',c['label_tolerance_minutes']).iloc[0]
                cutoff = init-pd.Timedelta(minutes=c['observation_latency_minutes'])
                current = match_values(obs,[cutoff],'backward',c['input_tolerance_minutes']).iloc[0]
                sample_id = f'{sid}_{init:%Y%m%dT%H%MZ}_h{lead:02}'
                row = {'sample_id':sample_id,'station_id':sid,'station_lat':st['latitude'],'station_lon':st['longitude'],
                       'init_utc':init,'issue_utc':init+pd.Timedelta(minutes=c['issue_delay_minutes']),
                       'valid_utc':valid,'lead_h':lead,'observed_t2m_c':target.temperature_c,
                       'label_observation_utc':target.observation_utc,
                       'label_offset_minutes':(target.observation_utc-valid).total_seconds()/60 if pd.notna(target.observation_utc) else np.nan,
                       'obs_current_c':current.temperature_c,'input_observation_utc':current.observation_utc,
                       'obs_age_minutes':(init-current.observation_utc).total_seconds()/60 if pd.notna(current.observation_utc) else np.nan,
                       'hour_sin':np.sin(2*np.pi*init.hour/24),'hour_cos':np.cos(2*np.pi*init.hour/24),
                    'year_sin':np.sin(2*np.pi*(init.day_of_year-1)/365.2425),
                    'year_cos':np.cos(2*np.pi*(init.day_of_year-1)/365.2425)}
                opt = {'sample_id':sample_id}
                meta = {'satellite':{},'tabular_resources':{},'history_observation_utc':{}}
                # Temperature from a report BEFORE each lag anchor, and available by init.
                for lag in c['history_hours']:
                    query = min(init-pd.Timedelta(hours=lag),cutoff)
                    past = match_values(obs,[query],'backward',c['input_tolerance_minutes']).iloc[0]
                    name = f'obs_lag{lag}_c'
                    opt[name] = past.temperature_c
                    meta['history_observation_utc'][name] = str(past.observation_utc)
                    meta['tabular_resources'][name] = []  # Same already-downloaded IEM table.
                for field in c['hrrr_fields']:
                    name = field['name']; r = field_results.get(name)
                    val = r['values'][sid]['value'] if r else np.nan
                    if name == 'hrrr_t2m_c':
                        row[name] = val
                        if r: row.update({k:v for k,v in r['values'][sid].items() if k != 'value'})
                    else: opt[name] = val
                    meta['tabular_resources'][name] = [r['resource_id']] if r else []
                row['error_c'] = row['observed_t2m_c']-row.get('hrrr_t2m_c',np.nan)
                row['overview_path'] = ''
                for band in g['bands']:
                    im = images.get((sid,band))
                    if im: meta['satellite'][str(band)] = im
                if g.get('prepare_action_crops', False):
                    meta['prepared_actions'] = {}
                    for action in c['actions']:
                        if action['kind'] == 'satellite':
                            im = images.get((sid,action['band']))
                            if im:
                                meta['prepared_actions'][action['id']] = prepared_crop(root,im,action,g,lead,source_arrays)
                overview = images.get((sid,g['overview_band']))
                if overview:
                    dest = root/'overviews'/Path(overview['path']).name
                    if dest.exists():
                        try:
                            with np.load(dest, allow_pickle=False) as data:
                                image = data['image_k']
                                if image.shape != (1,g['overview_pixels'],g['overview_pixels']) or not np.isfinite(image).any():
                                    raise ValueError('Invalid cached overview')
                        except Exception:
                            dest.replace(dest.with_suffix('.corrupt.npz'))
                    if not dest.exists():
                        with np.load(root/overview['path'],allow_pickle=False) as data:
                            arr = crop(data['image_k'],float(data['footprint_km']),g['footprint_km'],g['overview_pixels'])
                        write_npz(dest,image_k=arr)
                    row['overview_path'] = str(dest.relative_to(root)).replace('\\','/')
                rows.append(row); optional.append(opt); sample_meta[sample_id] = meta
        # Commit only successful initializations; failed acquisitions are retried on restart.
        if len(failures) == failure_start and not any(f['stage'] == 'IEM' for f in failures):
            write_parquet(progress / 'samples.parquet', pd.DataFrame(rows[row_start:]))
            write_parquet(progress / 'optional.parquet', pd.DataFrame(optional[optional_start:]))
            new_samples = {key: sample_meta[key] for key in set(sample_meta) - sample_keys}
            new_resources = {key: resource_meta[key] for key in set(resource_meta) - resource_keys}
            write_json(progress / 'metadata.json', {'samples': new_samples, 'resources': new_resources})
            image_files = {image['path'] for meta in new_samples.values() for image in meta['satellite'].values()}
            image_files.update(path for meta in new_samples.values() for path in meta.get('prepared_actions',{}).values())
            image_files.update(row['overview_path'] for row in rows[row_start:] if row['overview_path'])
            write_json(marker, {'fingerprint': fingerprint,
                'hashes': {name: sha256(progress / name) for name in ['samples.parquet', 'optional.parquet', 'metadata.json']},
                'images': {name: (root / name).stat().st_size for name in image_files}})
    if stop and stop.requested:
        raise SystemExit(75)
    df = pd.DataFrame(rows)
    for col in ['init_utc','issue_utc','valid_utc','label_observation_utc','input_observation_utc']:
        df[col] = pd.to_datetime(df[col],utc=True)
    df['split'] = split_rows(df,c.get('splits'),c['label_tolerance_minutes'])
    optional_df = pd.DataFrame(optional)
    write_parquet(root/'samples.parquet',df)
    write_parquet(root/'optional.parquet',optional_df)
    schema = {'version':2,'fingerprint':fingerprint,'core_features':CORE,'mos_extra':['hrrr_t2m_c'],
              'actions':c['actions'],'satellite':g,'samples':sample_meta,'resources':resource_meta,
              'cost_units':'User-configured relative compute proxy; not measured seconds/FLOPs',
              'availability':'Historical publication unknown: HRRR assumed at issue; IEM/GOES latency margins are assumptions',
              'config':c}
    write_json(root/'manifest.json',schema)
    report = audit(root)
    report['failures'] = len(failures)
    write_json(root/'failures.json',failures)
    write_json(root/'report.json',report)
    if report['violations']: raise ValueError(f'Dataset audit failed: {report["violations"]}')
    return report


def audit(root):
    root = Path(root); df = pd.read_parquet(root/'samples.parquet')
    manifest = json.loads((root/'manifest.json').read_text()); c = manifest['config']
    violations = []
    opt = pd.read_parquet(root/'optional.parquet')
    if opt.sample_id.duplicated().any() or set(opt.sample_id) != set(df.sample_id):
        violations.append('optional sample keys mismatch')
    if not np.allclose(df.error_c,df.observed_t2m_c-df.hrrr_t2m_c,equal_nan=True):
        violations.append('incorrect MOS residual')
    if not (df.split == split_rows(df,c.get('splits'),c['label_tolerance_minutes'])).all():
        violations.append('incorrect split assignment')
    if (df.issue_utc >= df.valid_utc).any(): violations.append('nonpositive issue-relative lead')
    if df.sample_id.duplicated().any(): violations.append('duplicate sample IDs')
    if not (df.valid_utc == df.init_utc+pd.to_timedelta(df.lead_h,unit='h')).all(): violations.append('invalid forecast times')
    cutoff = df.init_utc-pd.Timedelta(minutes=c['observation_latency_minutes'])
    if (df.input_observation_utc > cutoff).any(): violations.append('future input observation')
    if (df.label_offset_minutes.abs() > c['label_tolerance_minutes']).any(): violations.append('label outside tolerance')
    for row in df.itertuples():
        meta = manifest['samples'][row.sample_id]
        for col,time in meta['history_observation_utc'].items():
            if time == 'NaT': continue
            lag = int(col.removeprefix('obs_lag').removesuffix('_c'))
            limit = min(row.init_utc-pd.Timedelta(hours=lag),
                        row.init_utc-pd.Timedelta(minutes=c['observation_latency_minutes']))
            if utc(time) > limit: violations.append('future history observation')
        for image in meta['satellite'].values():
            if utc(image['available_utc']) > row.init_utc: violations.append('future satellite availability')
            if not (root/image['path']).is_file(): violations.append('missing satellite artifact')
    return {'rows':len(df),'labelled_rows':int(df.observed_t2m_c.notna().sum()),
            'hrrr_rows':int(df.hrrr_t2m_c.notna().sum()),'overview_rows':int(df.overview_path.ne('').sum()),
            'splits':df.split.value_counts().to_dict(),'violations':sorted(set(violations))}
