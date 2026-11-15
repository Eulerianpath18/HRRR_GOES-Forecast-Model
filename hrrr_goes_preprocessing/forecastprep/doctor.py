"""Fail early on unavailable GPUs, dependencies, storage, or archive access."""
import importlib.metadata
from pathlib import Path
import shutil
import tempfile
import platform
from .core import write_json


def diagnose(directory, gpu=False, network=False, min_free_gb=0.):
    import torch
    import numpy as np
    import xarray as xr
    import requests
    import eccodes
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free / 1e9
    if free < min_free_gb:
        raise RuntimeError(f'Only {free:.1f} GB filesystem space available, requested {min_free_gb:.1f} GB. Also check your user quota.')
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        path = Path(temporary)/'test.nc'
        xr.Dataset({'CMI': (('y','x'), np.ones((2,2),dtype='float32'))}).to_netcdf(path, engine='h5netcdf')
        with xr.open_dataset(path, engine='h5netcdf') as data:
            assert data.CMI.shape == (2,2)
        handle = eccodes.codes_grib_new_from_samples('regular_ll_sfc_grib2')
        eccodes.codes_release(handle)
    result = {'python': platform.python_version(), 'free_filesystem_GB': free, 'directory': str(directory),
              'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ['numpy','pandas','pyarrow','scipy','xarray','h5netcdf','herbie-data','torch','scikit-learn']}}
    if gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('Requested GPU is unavailable. Check Slurm --gres=gpu:1 and CUDA-enabled PyTorch.')
        tensor = torch.randn(32,32,device='cuda',requires_grad=True)
        tensor.square().sum().backward()
        torch.cuda.synchronize()
        result['gpu'] = torch.cuda.get_device_name(0)
    if network:
        urls = {
            'HRRR': 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20210101/conus/hrrr.t00z.wrfsfcf48.grib2.idx',
            'GOES': 'https://noaa-goes16.s3.amazonaws.com/?list-type=2&max-keys=1&prefix=ABI-L2-CMIPC/2021/001/',
            'IEM': 'https://mesonet.agron.iastate.edu/geojson/network/CO_ASOS.geojson'}
        for name,url in urls.items():
            response = requests.get(url,timeout=(15,60))
            response.raise_for_status()
            result[name] = {'status': response.status_code, 'bytes': len(response.content)}
    write_json(directory/'doctor.json', result)
    return result
