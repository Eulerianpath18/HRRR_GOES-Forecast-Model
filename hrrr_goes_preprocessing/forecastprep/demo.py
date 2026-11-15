"""Entirely synthetic integration fixture. Never represents real weather data."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .core import utc, write_npz, atomic_bytes
from .pipeline import build


class FakeIEM:
    def observations(self,station,start,end):
        t = pd.date_range(start,end,freq='h',inclusive='left')
        return pd.DataFrame({'observation_utc':t,'temperature_c':15+5*np.sin(np.arange(len(t))/12)})


class FakeHRRR:
    def field(self,init,lead,field,stations):
        return {'resource_id':f'fake-hrrr:{init}:{lead}:{field["name"]}','bytes':10000,
                'values':{s['station_id']:{'value':12+lead/24,'grid_distance_km':1.,
                          'grid_lat':s['latitude'],'grid_lon':s['longitude']} for s in stations}}


class FakeGOES:
    def __init__(self,root,spec): self.root,self.spec = Path(root),spec
    def scene(self,cutoff,band):
        return None,{'resource_id':f'fake-goes:{cutoff}:{band}','bytes':20000,
                     'scan_start_utc':(cutoff-pd.Timedelta(minutes=20)).isoformat(),
                     'scan_end_utc':(cutoff-pd.Timedelta(minutes=15)).isoformat(),
                     'created_utc':(cutoff-pd.Timedelta(minutes=12)).isoformat(),
                     'available_utc':(cutoff-pd.Timedelta(minutes=7)).isoformat()}
    def patch(self,scene,station,band):
        key = f"synthetic_{station['station_id']}_{utc(scene[1]['scan_end_utc']):%Y%m%dT%H%M}_C{band}.npz"
        relative = 'satellite/'+key
        n = self.spec['pixels']; x = np.linspace(-1,1,n)
        arr = (260+10*x[None,:]+20*x[::-1,None]+band).astype('float32')[None]
        arr[:,0,0] = np.nan
        write_npz(self.root/relative,image_k=arr,footprint_km=np.float32(self.spec['footprint_km']))
        return relative


def make_demo(output):
    root = Path(output); root.mkdir(parents=True,exist_ok=True)
    atomic_bytes(root/'SYNTHETIC_DATA.txt',b'SYNTHETIC fixture. Not downloaded observations or forecasts.\n')
    atomic_bytes(root/'stations.csv',b'station_id,latitude,longitude,iem_network\nDEN,39.8617,-104.6731,CO_ASOS\n')
    c = yaml.safe_load(Path(__file__).with_name('defaults.yaml').read_text())
    c.update(start='2025-08-01',end='2025-08-10',cycles=[0,12],stations=str((root/'stations.csv').resolve()),
             splits={'train_end':'2025-08-05','val_end':'2025-08-08','heldout_stations':[],'embargo_hours':0})
    c['satellite']['pixels'] = 64
    report = build(c,root,iem=FakeIEM(),hrrr=FakeHRRR(),goes=FakeGOES(root,c['satellite']))
    return {**report,'synthetic':True}
