from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import re
import os
import tempfile
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from .core import utc, digest, write_json, write_npz, atomic_bytes
from .runtime import sha256

_PATTERN = re.compile(r'_s(\d{14})_e(\d{14})_c(\d{14})')

def goes_times(key):
    m = _PATTERN.search(key)
    if not m: raise ValueError('Invalid GOES filename')
    def parse(s):
        return pd.Timestamp(datetime.strptime(s[:13],'%Y%j%H%M%S').replace(
            microsecond=int(s[13])*100000,tzinfo=timezone.utc))
    return tuple(map(parse,m.groups()))


def select_scene(keys, cutoff, band, number, lookback_minutes, latency_minutes):
    candidates = []
    cutoff = utc(cutoff)
    for key in keys:
        if not re.search(rf'C{band:02d}_G{number}_',key) or not key.endswith('.nc'): continue
        start,end,created = goes_times(key)
        available = max(end,created)+pd.Timedelta(minutes=latency_minutes)
        if start <= end <= created and available <= cutoff and end >= cutoff-pd.Timedelta(minutes=lookback_minutes):
            candidates.append((end,created,key))
    if not candidates: raise FileNotFoundError(f'No eligible GOES {number} C{band:02d} at {cutoff}')
    return max(candidates)[2]


def reproject(path, station, footprint_km, pixels):
    """CMI kelvin and DQF=0, north-up AEQD grid, endpoints at +/- width/2."""
    import xarray as xr
    from pyproj import CRS, Transformer
    with xr.open_dataset(path,engine='h5netcdf',mask_and_scale=True) as ds:
        if ds.CMI.attrs.get('units') != 'K': raise ValueError('Only infrared brightness temperature in kelvin supported')
        p = ds.goes_imager_projection.attrs
        local = CRS.from_proj4(f"+proj=aeqd +lat_0={station['latitude']} +lon_0={station['longitude']} +datum=WGS84 +units=m")
        tf = Transformer.from_crs(local, CRS.from_cf(p), always_xy=True)
        axis = np.linspace(-footprint_km*500,footprint_km*500,pixels)
        east,north = np.meshgrid(axis,axis[::-1])
        gx,gy = tf.transform(east,north)
        tx,ty = gx/p['perspective_point_height'],gy/p['perspective_point_height']
        finite = np.isfinite(tx)&np.isfinite(ty)
        if not finite.any(): raise ValueError('Patch is outside satellite view')
        x,y = ds.x.values,ds.y.values
        margin_x,margin_y = abs(np.diff(x)).max()*2,abs(np.diff(y)).max()*2
        def bounds(a,lo,hi): return slice(lo,hi) if a[0] < a[-1] else slice(hi,lo)
        sub = ds.sel(x=bounds(x,tx[finite].min()-margin_x,tx[finite].max()+margin_x),
                     y=bounds(y,ty[finite].min()-margin_y,ty[finite].max()+margin_y))
        if min(sub.sizes['x'],sub.sizes['y']) < 2: raise ValueError('No satellite sector coverage')
        # Flag bad native pixels before interpolation; all contributing pixels must be good.
        arr = sub.CMI.where(sub.DQF == 0).load()
        tx,ty = np.where(finite,tx,0),np.where(finite,ty,0)
        result = arr.interp(x=xr.DataArray(tx,dims=('row','col')),
                            y=xr.DataArray(ty,dims=('row','col'))).values.astype('float32')
        result[~finite] = np.nan
        return result


def crop(image, cached_width, width, pixels, east=0., north=0.):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[1]!=image.shape[2] or image.shape[1]<2:
        raise ValueError('Image must have shape [channels,N,N] with N>=2')
    if not 0 < width <= cached_width or pixels < 2: raise ValueError('Invalid crop geometry')
    if max(abs(east),abs(north))+width/2 > cached_width/2+1e-6: raise ValueError('Crop exceeds cached extent')
    n = image.shape[-1]
    x = east + np.linspace(-width/2,width/2,pixels)
    y = north + np.linspace(width/2,-width/2,pixels)
    rr,cc = np.meshgrid((cached_width/2-y)/cached_width*(n-1),
                        (x+cached_width/2)/cached_width*(n-1),indexing='ij')
    return np.stack([map_coordinates(ch,[rr,cc],order=1,mode='constant',cval=np.nan,prefilter=False)
                     for ch in image]).astype('float32')


class GOES:
    def __init__(self, root, spec, latency):
        import s3fs
        self.root,self.spec,self.latency = Path(root),spec,latency
        self.fs = s3fs.S3FileSystem(anon=True, config_kwargs={'retries':{'max_attempts':3},
                                                               'connect_timeout':15,'read_timeout':120})
        self.listings = {}

    def scene(self, cutoff, band):
        cutoff = utc(cutoff)
        start = (cutoff-pd.Timedelta(minutes=self.spec['lookback_minutes'])).floor('h')
        keys = []
        for hour in pd.date_range(start,cutoff.floor('h'),freq='h'):
            prefix = f"noaa-goes{self.spec['number']}/ABI-L2-CMIPC/{hour:%Y/%j/%H}/"
            if prefix not in self.listings:
                try: self.listings[prefix] = self.fs.ls(prefix,detail=False)
                except FileNotFoundError: self.listings[prefix] = []
            keys.extend(self.listings[prefix])
        key = select_scene(keys,cutoff,band,self.spec['number'],self.spec['lookback_minutes'],self.latency)
        path = self.root/'raw'/'goes'/Path(key).name
        downloaded = 0
        checksum = path.with_suffix('.sha256')
        if path.exists():
            try:
                if checksum.exists():
                    if checksum.read_text().strip() != sha256(path):
                        raise ValueError('Raw satellite checksum mismatch')
                else:
                    import xarray as xr
                    with xr.open_dataset(path, engine='h5netcdf') as cached:
                        cached.CMI.load()
                    atomic_bytes(checksum, sha256(path).encode())
            except (ValueError, OSError, KeyError, EOFError):
                path.replace(path.with_suffix('.corrupt.nc'))
                checksum.unlink(missing_ok=True)
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp = tempfile.mkstemp(dir=path.parent,suffix='.partial'); os.close(fd)
            try:
                self.fs.get(key,tmp)
                size = self.fs.info(key)['size']
                if Path(tmp).stat().st_size != size: raise IOError('Truncated GOES download')
                os.replace(tmp,path); downloaded = size
                atomic_bytes(checksum, sha256(path).encode())
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        start,end,created = goes_times(key)
        return path, {'resource_id':key,'bytes':path.stat().st_size,'downloaded_bytes_this_call':downloaded,
                      'scan_start_utc':start.isoformat(),'scan_end_utc':end.isoformat(),
                      'created_utc':created.isoformat(),
                      'available_utc':(max(end,created)+pd.Timedelta(minutes=self.latency)).isoformat()}

    def patch(self, scene, station, band):
        path,meta = scene
        key = digest({'source':meta['resource_id'],'station':station,'spec':self.spec,'schema':2})
        dest = self.root/'satellite'/f'{key}.npz'
        if dest.exists():
            try:
                with np.load(dest, allow_pickle=False) as cached:
                    image = cached['image_k']
                    if image.shape != (1, self.spec['pixels'], self.spec['pixels']) or np.isfinite(image).mean() < self.spec['min_valid_fraction']:
                        raise ValueError('Invalid cached patch')
            except (ValueError, OSError, KeyError, EOFError):
                dest.replace(dest.with_suffix('.corrupt.npz'))
        if not dest.exists():
            arr = reproject(path,station,self.spec['footprint_km'],self.spec['pixels'])
            valid_fraction = float(np.isfinite(arr).mean())
            if valid_fraction < self.spec['min_valid_fraction']:
                raise ValueError(f'Satellite valid fraction {valid_fraction:.3f} below threshold')
            write_npz(dest,image_k=arr[None],valid_mask=np.isfinite(arr)[None],
                      footprint_km=np.float32(self.spec['footprint_km']),band=np.int16(band),
                      latitude=np.float64(station['latitude']),longitude=np.float64(station['longitude']))
            write_json(dest.with_suffix('.json'),{**meta,'station':station,'valid_fraction':valid_fraction,
                        'grid':'AEQD, north up, east right; endpoints +/- footprint/2','schema':2})
        return str(dest.relative_to(self.root)).replace('\\','/')
