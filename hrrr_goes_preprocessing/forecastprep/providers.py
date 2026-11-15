"""Public archive adapters. Cache keys include the complete acquisition request."""
from __future__ import annotations
from io import StringIO
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .core import utc, digest, atomic_bytes, write_json


class IEM:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                           status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))
        self.last_request = 0.

    def observations(self, station, start, end):
        params = dict(station=station['station_id'], network=station['iem_network'], data='tmpf',
                      sts=utc(start).isoformat(), ets=utc(end).isoformat(), tz='Etc/UTC',
                      format='onlycomma', latlon='no', missing='M', trace='T', direct='no', report_type=[3,4])
        path = self.cache / (digest(params)+'.csv')
        if path.exists():
            try:
                cached = self.parse(path.read_text(), station['station_id'])
                if cached.empty:
                    raise ValueError('Empty cached station observations')
            except (ValueError, KeyError, UnicodeError, pd.errors.ParserError):
                path.replace(path.with_suffix('.corrupt.csv'))
        if not path.exists():
            time.sleep(max(0,1.1-(time.monotonic()-self.last_request)))
            r = self.session.get('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
                                 params=params, timeout=(15,120))
            self.last_request = time.monotonic()
            r.raise_for_status()
            # Validate before committing the cache (no error HTML masquerading as CSV).
            if self.parse(r.text, station['station_id']).empty:
                raise ValueError('IEM returned no usable observations for ' + station['station_id'])
            atomic_bytes(path, r.content)
            write_json(path.with_suffix('.json'), {'url':r.url,'retrieved_utc':pd.Timestamp.now(tz='UTC'),
                                                   'bytes':len(r.content)})
        df = self.parse(path.read_text(), station['station_id'])
        return df[(df.observation_utc >= utc(start)) & (df.observation_utc < utc(end))].reset_index(drop=True)

    @staticmethod
    def parse(text, station_id):
        f = pd.read_csv(StringIO(text), comment='#', na_values=['M'], dtype={'station':str})
        if not {'valid','tmpf'} <= set(f): raise ValueError('Invalid IEM CSV schema')
        if 'station' in f and not (f.station == station_id).all(): raise ValueError('IEM station mismatch')
        out = pd.DataFrame({'observation_utc':pd.to_datetime(f.valid,utc=True,errors='coerce'),
                            'temperature_c':(pd.to_numeric(f.tmpf,errors='coerce')-32)*5/9})
        out = out.dropna()
        out = out[out.temperature_c.between(-100,65)]
        # Conflicting duplicate temperatures are excluded rather than arbitrarily chosen.
        good = out.groupby('observation_utc').temperature_c.nunique()
        return out[out.observation_utc.isin(good[good == 1].index)].drop_duplicates('observation_utc').sort_values('observation_utc').reset_index(drop=True)


def nearest_cell(lat, lon, station, max_distance_km):
    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.shape != lon.shape or lat.ndim != 2: raise ValueError('Expected 2D HRRR coordinates')
    p1, p2 = np.deg2rad(station['latitude']), np.deg2rad(lat)
    dlon = np.deg2rad((lon-station['longitude']+180)%360-180)
    a = np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    d = 6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    cell = np.unravel_index(np.nanargmin(d),d.shape)
    if d[cell] > max_distance_km: raise ValueError(f"Station {station['station_id']} is {d[cell]:.1f} km from HRRR grid")
    return cell, float(d[cell])


class HRRR:
    def __init__(self, cache, max_distance_km):
        self.cache, self.max_distance = Path(cache), max_distance_km
        self.grid_cells = {}

    def field(self, init, lead, field, stations):
        from herbie import Herbie
        request = {'init':str(init),'lead':lead,'field':field,'stations':stations,
                   'max_distance':self.max_distance,'schema':2}
        path = self.cache / 'points' / (digest(request)+'.json')
        if path.exists():
            import json
            try:
                result = json.loads(path.read_text())
                if result['request'] != request or not all(np.isfinite(result['values'][st['station_id']]['value']) for st in stations):
                    raise ValueError('Invalid cached HRRR points')
                return result
            except (ValueError, KeyError, TypeError):
                path.replace(path.with_suffix('.corrupt.json'))
        h = Herbie(utc(init).tz_localize(None).to_pydatetime(), model='hrrr', product='sfc', fxx=lead,
                   save_dir=self.cache/'grib', priority=['aws','google'], verbose=False)
        try:
            ds = h.xarray(field['search'], remove_grib=False)
        except Exception:
            # A partial/corrupt cached GRIB should not poison every retry.
            cached = Path(h.get_localFilePath(field['search']))
            if not cached.is_file():
                raise
            cached.replace(cached.with_suffix(cached.suffix + '.corrupt'))
            ds = h.xarray(field['search'], remove_grib=False)
        if isinstance(ds, list):
            for part in ds: part.close()
            raise ValueError('HRRR search must select one unambiguous message')
        if ds is None: raise ValueError('Empty HRRR dataset')
        try:
            da = ds[field['variable']]
            if utc(ds.valid_time.item()) != utc(init)+pd.Timedelta(hours=lead): raise ValueError('HRRR valid time mismatch')
            if utc(ds.time.item()) != utc(init): raise ValueError('HRRR initialization mismatch')
            if da.attrs.get('units') != field['units']: raise ValueError(f"Unexpected HRRR units {da.attrs.get('units')}")
            import hashlib
            latitude, longitude = ds.latitude.values, ds.longitude.values
            geometry = hashlib.sha256(latitude.tobytes() + longitude.tobytes()).hexdigest()
            values = {}
            for st in stations:
                cell_key = (geometry, st['latitude'], st['longitude'], self.max_distance)
                if cell_key not in self.grid_cells:
                    self.grid_cells[cell_key] = nearest_cell(latitude, longitude, st, self.max_distance)
                (iy,ix), distance = self.grid_cells[cell_key]
                val = float(da.isel(y=iy,x=ix).item())+field.get('offset',0)
                if not np.isfinite(val): raise ValueError('Missing HRRR value')
                values[st['station_id']] = {'value':val,'grid_distance_km':distance,
                    'grid_lat':float(ds.latitude.values[iy,ix]),
                    'grid_lon':float((ds.longitude.values[iy,ix]+180)%360-180)}
            grib = Path(h.get_localFilePath(field['search']))
            result = {'values':values,'source':str(h.grib),'resource_id':f'hrrr:{utc(init).isoformat()}:{lead}:{field["name"]}',
                      'bytes':grib.stat().st_size if grib.exists() else None,
                      'retrieved_utc':pd.Timestamp.now(tz='UTC').isoformat(), 'request':request}
            write_json(path,result)
            return result
        finally: ds.close()
