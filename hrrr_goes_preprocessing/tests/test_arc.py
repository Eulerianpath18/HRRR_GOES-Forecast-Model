"""Check submission dependency wiring without contacting a Slurm scheduler."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('stage',['train','finish'])
def test_submission_dependencies(tmp_path,monkeypatch,stage):
    project=Path(__file__).resolve().parents[1]
    env={'FORECAST_PROJECT':str(project),'FORECAST_DATA':str(tmp_path/'data'),
         'FORECAST_RUNS':str(tmp_path/'runs'),'FORECAST_STORAGE':str(tmp_path),
         'FORECAST_CPU_PARTITION':'cpu2021','FORECAST_GPU_PARTITION':'gpu-a100',
         'FORECAST_GPU_CONCURRENCY':'2','FORECAST_IMPORT_CONCURRENCY':'1','FORECAST_ACCOUNT':''}
    for name,value in env.items():monkeypatch.setenv(name,value)
    (tmp_path/'runs').mkdir()
    (tmp_path/'runs/experiments.json').write_text(json.dumps({'runs':[{}]*48,'settings':{
        'modes':['direct','mos'],'profiles':dict.fromkeys(['mlp','overview','full','masked']),
        'seeds':[11,29,47]}}))
    spec=importlib.util.spec_from_file_location('arc_submit_test',project/'arc/submit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=str(100+len(calls))+'\n')
    monkeypatch.setattr(module.subprocess,'run',run)
    monkeypatch.setattr(module.sys,'argv',['submit.py',stage])
    module.main()
    by_stage={args[-1]:(str(101+index),args) for index,args in enumerate(calls)}
    assert set(by_stage)==({'train','select','baselines','evaluate','report'} if stage=='train'
                           else {'select','baselines','evaluate','report'})
    selection_id,selection_args=by_stage['select']
    if stage=='train':
        train_id,args=by_stage['train']
        assert '--array=0-47%2' in args and '--gres=gpu:1' in args
        assert '--dependency=afterok:'+train_id in selection_args
    assert '--dependency=afterok:'+selection_id in by_stage['baselines'][1]
    assert '--dependency=afterok:'+selection_id in by_stage['evaluate'][1]
    assert '--array=0-23%2' in by_stage['evaluate'][1]
    assert '--gres=gpu:1' not in by_stage['baselines'][1]
    assert '--dependency=afterok:'+by_stage['baselines'][0]+':'+by_stage['evaluate'][0] in by_stage['report'][1]
