import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader
from forecastprep.core import utc, match_values, split_rows, validate_config, load_config, initializations
from forecastprep.providers import IEM, nearest_cell
from forecastprep.satellite import goes_times, select_scene, crop, reproject
from forecastprep.demo import make_demo
from forecastprep.pipeline import audit
from forecastprep.torch_data import FeatureStore, AcquisitionSession, ForecastDataset, Standardizer


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp('demo')
    make_demo(root)
    return root


def test_goes_creation_cutoff():
    a='OR_ABI-L2-CMIPC-M6C13_G19_s20252122350100_e20252122353000_c20252122354000.nc'
    b='OR_ABI-L2-CMIPC-M6C13_G19_s20252122355000_e20252122358000_c20252130001000.nc'
    assert select_scene([a,b],utc('2025-08-01'),13,19,60,5)==a
    assert goes_times(a)[0].microsecond==0
    with pytest.raises(FileNotFoundError): select_scene([b],utc('2025-08-01'),13,19,60,0)


def test_observation_alignment_and_missing():
    obs=pd.DataFrame({'observation_utc':pd.to_datetime(['2025-08-01T11:53Z','2025-08-01T12:02Z']),
                      'temperature_c':[10.,99.]})
    t=[utc('2025-08-01T12:00Z')]
    assert match_values(obs,t,'backward',20).temperature_c.iloc[0]==10
    assert match_values(obs,t,'nearest',20).temperature_c.iloc[0]==99
    assert pd.isna(match_values(obs,[utc('2025-08-02')],'nearest',20).temperature_c.iloc[0])


def test_iem_conflicting_duplicate_and_fahrenheit():
    text='station,valid,tmpf\nDEN,2025-08-01 00:00,32\nDEN,2025-08-01 01:00,50\nDEN,2025-08-01 01:00,60\nDEN,2025-08-01 02:00,M\n'
    out=IEM.parse(text,'DEN')
    assert len(out)==1 and out.temperature_c.iloc[0]==0


def test_spherical_grid_wrap_and_reject():
    st={'station_id':'x','latitude':40.,'longitude':-105.}
    cell,d=nearest_cell(np.array([[40.,41.]]),np.array([[255.,256.]]),st,10)
    assert cell==(0,0) and d==0
    with pytest.raises(ValueError): nearest_cell(np.array([[0.]]),np.array([[0.]]),st,10)


def test_crop_orientation_and_boundary():
    # Values increase to the north and east; positive north shift must increase them.
    x=np.linspace(-50,50,101); image=(2*x[::-1,None]+x[None,:])[None].astype('float32')
    central=crop(image,100,20,3)
    assert central[0,1,1]==0 and central[0,0,1]==20 and central[0,1,2]==10
    assert crop(image,100,20,3,north=10)[0,1,1]==20
    with pytest.raises(ValueError): crop(image,100,90,32,east=10)


def test_reprojection_dqf_and_north(tmp_path):
    n=101; x=np.linspace(-.01,.01,n); y=x[::-1]
    arr=np.broadcast_to((280+y[:,None]*1000),(n,n)).copy()
    flags=np.zeros((n,n),dtype='uint8'); flags[45:55,45:55]=2
    ds=xr.Dataset({'CMI':(('y','x'),arr,{'units':'K'}),'DQF':(('y','x'),flags),
                   'goes_imager_projection':((),0,{'grid_mapping_name':'geostationary',
                       'perspective_point_height':35786023.,'semi_major_axis':6378137.,
                       'semi_minor_axis':6356752.31414,'longitude_of_projection_origin':-75.,
                       'latitude_of_projection_origin':0.,'sweep_angle_axis':'x'})},coords={'x':x,'y':y})
    path=tmp_path/'goes.nc'; ds.to_netcdf(path,engine='h5netcdf')
    result=reproject(path,{'latitude':0.,'longitude':-75.},200,31)
    assert np.isnan(result[15,15]) and result[0,15]>result[-1,15]


def test_splits_purge_and_geography():
    df=pd.DataFrame({'init_utc':pd.to_datetime(['2025-01-01T18:00Z','2025-01-02T00:00Z','2025-01-03T00:00Z']),
                     'valid_utc':pd.to_datetime(['2025-01-02T00:00Z','2025-01-02T06:00Z','2025-01-03T06:00Z']),
                     'station_id':['DEN','DEN','XYZ']})
    split=split_rows(df,{'train_end':'2025-01-02','val_end':'2025-01-03','heldout_stations':['XYZ']},20)
    assert list(split)==['purged','val','test_spatial']


def test_manifest_audit_and_scene_reuse(data):
    assert audit(data)['violations']==[]
    f=pd.read_parquet(data/'samples.parquet')
    assert f.groupby(['station_id','init_utc']).overview_path.nunique().max()==1
    assert np.allclose(f.error_c,f.observed_t2m_c-f.hrrr_t2m_c)


def test_budget_no_peeking_and_modes(data,monkeypatch):
    st=FeatureStore(data,'direct')
    assert 'hrrr_t2m_c' not in st.columns and 'hrrr_wind' not in st.actions
    calls=[]; original=st.load_action
    def spy(s,a): calls.append(a['id']); return original(s,a)
    monkeypatch.setattr(st,'load_action',spy)
    sess=AcquisitionSession(st,st.rows.index[0],1.)
    state=sess.observation(); sess.available_actions()
    assert calls==[] and 'target' not in state
    assert not state['optional']['water_vapour']['value'].any()
    after=sess.acquire('water_vapour')
    assert calls==['water_vapour'] and after['optional']['water_vapour']['acquired']
    assert sess.spent==1 and sess.source_bytes==20000
    assert not state['optional']['water_vapour']['acquired']
    with pytest.raises(ValueError): sess.acquire('infrared_local')
    with pytest.raises(ValueError): sess.acquire('water_vapour')


def test_shared_scene_cost(data):
    st=FeatureStore(data)
    sess=AcquisitionSession(st,st.rows.index[0],2.)
    sess.acquire('infrared_local'); sess.acquire('infrared_context')
    assert sess.source_bytes==0 and sess.spent==2. # Raw C13 paid by overview.


def test_train_only_standardizer(data):
    scaler=Standardizer.fit(data)
    f=pd.read_parquet(data/'samples.parquet')
    assert scaler.values['obs_current_c'][0]==pytest.approx(f.loc[f.split=='train','obs_current_c'].mean())
    assert 'observed_t2m_c' not in scaler.values and 'error_c' not in scaler.values


def test_dataloader_and_backward(data):
    normalizer=Standardizer.fit(data)
    for mode in ['direct','mos']:
        d=ForecastDataset(data,mode,'train',normalizer,actions=['history','infrared_local'],budget=1.05)
        b=next(iter(DataLoader(d,batch_size=4,num_workers=0)))
        assert b['inputs']['overview'].shape==(4,1,32,32)
        assert b['inputs']['optional']['infrared_local']['value'].shape==(4,1,128,128)
        assert torch.isfinite(b['inputs']['core']).all()
        model=torch.nn.Linear(b['inputs']['core'].shape[-1],1)
        loss=(model(b['inputs']['core']).squeeze(-1)-b['target']).square().mean()
        loss.backward(); assert torch.isfinite(model.weight.grad).all()


def test_missing_image_remains_masked(data,tmp_path):
    st=FeatureStore(data); sid=st.rows.index[0]
    st.manifest['samples'][sid]['satellite'].pop('8')
    state=AcquisitionSession(st,sid,1.).acquire('water_vapour')
    assert state['optional']['water_vapour']['acquired']
    assert not state['optional']['water_vapour']['valid'].any()


def test_normalizer_wrong_dataset(data):
    with pytest.raises(ValueError): FeatureStore(data,normalizer=Standardizer({},'wrong'))


def test_config_guards():
    c=load_config(Path(__file__).parents[1]/'configs/smoke.yaml')
    c['cycles']=[1]; c['leads']=[48]
    with pytest.raises(ValueError): validate_config(c)


def test_date_sampling_and_default():
    c=load_config(Path(__file__).parents[1]/'configs/smoke.yaml')
    c.update(start='2024-02-27',end='2024-03-04',cycles=[0,12])
    assert len(initializations(c))==12
    c['sample_every_n_days']=4
    assert [str(t) for t in initializations(c)]==[
        '2024-02-27 00:00:00+00:00', '2024-02-27 12:00:00+00:00',
        '2024-03-02 00:00:00+00:00', '2024-03-02 12:00:00+00:00']
    c['sample_every_n_days']=0
    with pytest.raises(ValueError): validate_config(c)


def test_random_masks_budget_and_reproducibility(data):
    from forecastprep.torch_data import RandomAcquisitionDataset
    d=RandomAcquisitionDataset(data,'mos',budgets=[1.05],stop_probability=0)
    torch.manual_seed(12); a=d[0]
    torch.manual_seed(12); b=d[0]
    used=sum(d.store.actions[k]['cost_units'] for k,v in a['inputs']['optional'].items() if v['acquired'])
    assert used<=1.05+1e-9
    for k in d.store.actions:
        assert torch.equal(a['inputs']['optional'][k]['value'],b['inputs']['optional'][k]['value'])


def test_label_columns_rejected_as_actions():
    c=load_config(Path(__file__).parents[1]/'configs/smoke.yaml')
    c['actions'][0]['columns']=['observed_t2m_c']
    with pytest.raises(ValueError): validate_config(c)


def test_prerequisite_cycles_rejected():
    c=load_config(Path(__file__).parents[1]/'configs/smoke.yaml')
    c['actions'][0]['requires']=['history']
    with pytest.raises(ValueError): validate_config(c)
