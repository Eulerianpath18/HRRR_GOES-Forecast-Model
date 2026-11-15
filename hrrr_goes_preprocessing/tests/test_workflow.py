import copy
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data._utils.collate import default_collate
from forecastprep.archive import verify_ready, quality
from forecastprep.core import load_config, write_json
from forecastprep.demo import FakeIEM, FakeHRRR, FakeGOES
from forecastprep.pipeline import build
from forecastprep.selftest import prepare_fixture
from forecastprep.torch_data import FeatureStore, AcquisitionSession, Standardizer
from forecastprep.experiment_data import training_partitions
from forecastprep.models import ForecastModel
from forecastprep.training import run
from forecastprep.baselines import matrix
from forecastprep.evaluation import summary
from forecastprep.runtime import locked


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    return prepare_fixture(tmp_path_factory.mktemp('archive'))


def config():
    return dict(mode='mos',kind='hybrid',hidden=[8],embedding=4,dropout=.1,
                seed=7,learning_rate=.005,weight_decay=.0001,epochs=2,patience=5,
                batch_size=4,workers=0,threads=1,checkpoint_batches=1,
                acquisition='random',budgets=[0.,3.15],policy_fraction=.3,
                actions=['history','hrrr_wind','hrrr_dewpoint','infrared_local','infrared_context','water_vapour'])


@pytest.mark.parametrize('workers',[0,pytest.param(2,marks=pytest.mark.skipif(
    os.environ.get('FORECAST_TEST_MULTIPROCESS')!='1',
    reason='Enable FORECAST_TEST_MULTIPROCESS=1 on ARC to test worker IPC'))])
def test_exact_mid_epoch_resume(archive,tmp_path,workers):
    setting = config()
    setting['workers'] = workers
    run(archive,tmp_path/'continuous',setting)
    with pytest.raises(SystemExit) as stopped:
        run(archive,tmp_path/'resumed',setting,stop_after_batches=1)
    assert stopped.value.code==75
    saved=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert saved['offset']==4 and saved['steps']==1
    run(archive,tmp_path/'resumed',setting)
    uninterrupted=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert resumed['steps']==uninterrupted['steps']
    for name,value in uninterrupted['model'].items():
        assert torch.equal(value,resumed['model'][name]),name
    assert resumed['history']==uninterrupted['history']
    altered={**setting,'learning_rate':.1}
    with pytest.raises(ValueError): run(archive,tmp_path/'resumed',altered)


def test_policy_partition_and_normalizer(archive):
    store=FeatureStore(archive,'mos')
    predictor,policy,boundary=training_partitions(store,.3)
    assert not set(predictor)&set(policy)
    assert (store.rows.loc[predictor].valid_utc+pd.Timedelta('20min')<pd.Timestamp(boundary)).all()
    assert set(store.rows.loc[predictor+policy].split)=={'train'}
    scale=Standardizer.fit(archive,sample_ids=predictor)
    assert scale.values['hrrr_t2m_c'][0]==pytest.approx(store.rows.loc[predictor].hrrr_t2m_c.mean())
    with pytest.raises(ValueError): Standardizer.fit(archive,sample_ids=store.rows[store.rows.split=='test'].index.tolist())


def test_unacquired_images_do_not_run_encoders_or_affect_prediction(archive):
    store=FeatureStore(archive,'mos')
    session=AcquisitionSession(store,store.rows.index[0],0)
    state=default_collate([session.observation()])
    model=ForecastModel(len(store.columns),store.actions,embedding=4,hidden=[8],dropout=0).eval()
    calls=[]
    hooks=[model.encoders[name].register_forward_hook(lambda *args:calls.append(True))
           for name,action in store.actions.items() if action['kind']=='satellite']
    with torch.no_grad():
        expected=model(state)
        for item in state['optional'].values():
            item['value'].fill_(10000)
            item['valid'].fill_(True)
        actual=model(state)
    assert torch.equal(expected,actual) and not calls
    for hook in hooks:hook.remove()


def test_direct_baseline_excludes_hrrr(archive):
    store=FeatureStore(archive,'direct',Standardizer.fit(archive))
    ids=store.rows.index[:3].tolist()
    before=matrix(store,ids)
    store.rows.loc[ids,'hrrr_t2m_c']=9999
    store.optional.loc[ids,['hrrr_u10_ms','hrrr_v10_ms','hrrr_d2m_c']]=9999
    assert np.array_equal(before,matrix(store,ids))


def test_changed_dataset_rejected(archive,tmp_path):
    import shutil
    destination=tmp_path/'copy'
    shutil.copytree(archive,destination)
    verify_ready(destination)
    with (destination/'optional.parquet').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):verify_ready(destination)


def test_import_recovers_after_provider_failure(tmp_path):
    c=load_config(Path(__file__).parents[1]/'configs/smoke.yaml')
    c.update(end='2025-08-03',strict=True)
    c['satellite']['pixels']=16
    calls=[]
    class Provider(FakeHRRR):
        broken=True
        def field(self,init,lead,field,stations):
            calls.append(init)
            if self.broken and init.day==2:raise RuntimeError('injected interruption')
            return super().field(init,lead,field,stations)
    provider=Provider()
    with pytest.raises(RuntimeError):
        build(c,tmp_path,iem=FakeIEM(),hrrr=provider,goes=FakeGOES(tmp_path,c['satellite']))
    assert (tmp_path/'progress/20250801T0000Z/complete.json').exists()
    calls.clear();provider.broken=False
    result=build(c,tmp_path,iem=FakeIEM(),hrrr=provider,goes=FakeGOES(tmp_path,c['satellite']))
    assert result['rows']==4 and result['failures']==0
    assert all(time.day==2 for time in calls)


def test_process_lock_rejects_second_writer(tmp_path):
    with locked(tmp_path):
        with pytest.raises(RuntimeError):
            with locked(tmp_path):pass


def test_acceptance_catches_low_coverage(archive,tmp_path):
    import shutil
    dest=tmp_path/'copy';shutil.copytree(archive,dest)
    frame=pd.read_parquet(dest/'samples.parquet')
    frame.loc[frame.split=='test',['hrrr_t2m_c','error_c']]=np.nan
    frame.to_parquet(dest/'samples.parquet',index=False)
    report,_=quality(dest)
    assert any('low hrrr coverage' in message for message in report['violations'])


def test_paired_block_bootstrap():
    n=100
    frame=pd.DataFrame({'init_utc':pd.date_range('2024-01-01',periods=n,freq='D',tz='UTC'),
                        'prediction_c':np.ones(n),'observed_t2m_c':np.zeros(n),
                        'hrrr_t2m_c':np.full(n,2.),'optional_cost':np.zeros(n)})
    result=summary(frame,100)
    assert result['rmse_c']==1 and result['rmse_improvement_ci_low_c']==1
    assert result['rmse_improvement_ci_high_c']==1


def test_baseline_recovery(archive,tmp_path):
    from forecastprep.baselines import fit, predictions
    settings=dict(policy_fraction=.3,seed=4,ridge_alphas=[1.],gp_max_train=8,
                  gp_length_scales=[1.],gp_noise=.1)
    fit(archive,tmp_path/'full',settings)
    with pytest.raises(SystemExit) as interrupted:
        fit(archive,tmp_path/'resume',settings,stop_after_models=1)
    assert interrupted.value.code==75
    fit(archive,tmp_path/'resume',settings)
    _,expected=predictions(archive,tmp_path/'full/baselines.joblib','test')
    _,actual=predictions(archive,tmp_path/'resume/baselines.joblib','test')
    for name in expected:
        assert np.array_equal(expected[name],actual[name]),name


def test_changed_image_rejected(archive,tmp_path):
    import os
    import shutil
    dest=tmp_path/'copy';shutil.copytree(archive,dest)
    marker=json.loads((dest/'READY.json').read_text())
    name,signature=next(iter(marker['images'].items()))
    path=dest/name
    content=bytearray(path.read_bytes());content[-1]^=1;path.write_bytes(content)
    os.utime(path,ns=(signature['mtime_ns']+1000000000,signature['mtime_ns']+1000000000))
    with pytest.raises(ValueError,match='Image content changed'):verify_ready(dest)


def test_prepared_crops_match_dynamic_loading(archive):
    store=FeatureStore(archive,'mos')
    sid=store.rows.index[0]
    metadata=store.manifest['samples'][sid]
    assert metadata['prepared_actions']
    for name,path in list(metadata['prepared_actions'].items()):
        prepared=store.load_action(sid,store.actions[name])
        del metadata['prepared_actions'][name]
        dynamic=store.load_action(sid,store.actions[name])
        assert torch.equal(prepared['value'],dynamic['value'])
        assert torch.equal(prepared['valid'],dynamic['valid'])
        metadata['prepared_actions'][name]=path
