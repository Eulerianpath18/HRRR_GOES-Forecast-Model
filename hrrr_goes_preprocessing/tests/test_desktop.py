import copy
from pathlib import Path
import pandas as pd
import pytest
import torch
from forecastprep.desktop import latest_cycle, export_bundle, load_bundle, forecast, validate_live_config, fetch_inputs
from forecastprep.core import load_config
from forecastprep.selftest import prepare_fixture
from forecastprep.training import run, load_predictor
from forecastprep.experiment_data import ExperimentDataset
from forecastprep.demo import FakeIEM, FakeHRRR, FakeGOES
from torch.utils.data import DataLoader


def test_latest_cycle_respects_delay():
    c={'issue_delay_minutes':90,'cycles':[0,12]}
    assert str(latest_cycle(c,'2026-11-15 13:29Z'))=='2026-11-15 00:00:00+00:00'
    assert str(latest_cycle(c,'2026-11-15 13:30Z'))=='2026-11-15 12:00:00+00:00'
    assert str(latest_cycle(c,'2026-11-15 00:30Z'))=='2026-11-14 12:00:00+00:00'


@pytest.fixture(scope='module')
def portable(tmp_path_factory):
    root=prepare_fixture(tmp_path_factory.mktemp('desktop_data'))
    output=tmp_path_factory.mktemp('desktop_model')
    cfg=dict(mode='mos',kind='hybrid',hidden=[8],embedding=4,dropout=0,seed=8,
      learning_rate=.005,weight_decay=.0001,epochs=1,patience=2,batch_size=8,workers=0,threads=1,
      checkpoint_batches=1,acquisition='fixed',budgets=[0.,3.15],policy_fraction=.3,
      actions=['history','hrrr_wind','hrrr_dewpoint','infrared_local','infrared_context','water_vapour'])
    run(root,output,cfg)
    export_bundle(root,output/'best.pt',output/'gui.pt')
    return root,output,load_bundle(output/'gui.pt')


def test_portable_inference_matches_training(portable):
    root,output,bundle=portable
    store,model,_=load_predictor(root,output/'best.pt')
    sid=store.rows[store.rows.station_id=='DEN'].index[0]
    batch=next(iter(DataLoader(ExperimentDataset(store,[sid],bundle['config']['actions']),batch_size=1)))
    with torch.no_grad():expected=float(model(batch['inputs'])[0])+float(store.rows.loc[sid,'hrrr_t2m_c'])
    result=forecast(bundle,root,[6],station_id='DEN')
    assert all(r['station_id']=='DEN' for r in result)
    assert result[0]['prediction_c']==pytest.approx(expected,abs=1e-6)
    assert result[0]['correction_c']+result[0]['hrrr_t2m_c']==pytest.approx(result[0]['prediction_c'])
    with pytest.raises(ValueError,match='absent'):forecast(bundle,root,[6],station_id='MISSING')
    with pytest.raises(ValueError):forecast(bundle,root,[6],budget=float('nan'),station_id='DEN')
    with pytest.raises(ValueError):forecast(bundle,root,[6],{sid:{'observed_t2m_c':99.}},station_id='DEN')


def test_portable_inference_without_archive_or_labels(portable,tmp_path):
    import shutil
    import json
    root,_,bundle=portable
    current=tmp_path/'current'
    current.mkdir()
    rows=pd.read_parquet(root/'samples.parquet')
    keep=(rows.station_id=='DEN')&(rows.init_utc==rows.init_utc.min())
    rows=rows[keep].copy()
    rows[['observed_t2m_c','error_c']]=float('nan')
    rows['label_observation_utc']=pd.NaT
    rows.to_parquet(current/'samples.parquet',index=False)
    optional=pd.read_parquet(root/'optional.parquet')
    optional[optional.sample_id.isin(rows.sample_id)].to_parquet(current/'optional.parquet',index=False)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['samples']={sid:manifest['samples'][sid] for sid in rows.sample_id}
    paths=set(rows.overview_path)
    for meta in manifest['samples'].values():
        paths.update(s['path'] for s in meta['satellite'].values())
        paths.update(meta.get('prepared_actions',{}).values())
    for path in paths:
        if path:
            (current/path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/path,current/path)
    (current/'manifest.json').write_text(json.dumps(manifest))
    original=forecast(bundle,root,[6],station_id='DEN')[0]
    result=forecast(bundle,current,[6],station_id='DEN')
    assert len(result)==1
    assert result[0]['prediction_c']==original['prediction_c']


def test_bundle_checks_schema(portable):
    _,_,bundle=portable
    cfg=copy.deepcopy(bundle['schema']['config']);cfg['satellite']['number']=19
    validate_live_config(cfg,bundle)
    cfg['satellite']['pixels']*=2
    with pytest.raises(ValueError):validate_live_config(cfg,bundle)


def test_live_inputs_work_in_background_thread(tmp_path):
    import concurrent.futures
    c=load_config(Path(__file__).parents[1]/'configs/live.yaml')
    c['satellite']['pixels']=32
    st={'station_id':'DEN','latitude':39.83,'longitude':-104.66,'iem_network':'CO_ASOS'}
    def providers(root,config):
        class Inputs(FakeIEM):
            def observations(self,station,start,end):
                return super().observations(station,start,min(pd.Timestamp(end),pd.Timestamp(config['start'])))
        return dict(iem=Inputs(),hrrr=FakeHRRR(),goes=FakeGOES(root,config['satellite']))
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        root=pool.submit(fetch_inputs,c,st,tmp_path,'2026-11-15 14:00Z',providers).result()
    rows=pd.read_parquet(root/'samples.parquet')
    assert len(rows)==4 and rows.observed_t2m_c.isna().all()
    assert (rows.input_observation_utc<rows.init_utc).all()
    assert rows.init_utc.iloc[0]==pd.Timestamp('2026-11-15 12:00Z')
