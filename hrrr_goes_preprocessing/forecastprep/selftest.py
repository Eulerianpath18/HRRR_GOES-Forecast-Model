"""Small synthetic experiment exercising the same entry points as the ARC jobs."""
from pathlib import Path
import yaml
from .core import atomic_bytes, write_json
from .demo import FakeIEM, FakeHRRR, FakeGOES
from .archive import plan, build_part, finalize
from . import suite


def prepare_fixture(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_bytes(output/'SYNTHETIC_DATA.txt', b'SYNTHETIC integration check; scores have no scientific meaning.\n')
    station_file = output/'stations.csv'
    atomic_bytes(station_file, b'station_id,latitude,longitude,iem_network\nDEN,39.83,-104.66,CO_ASOS\nHOLD,40.5,-101.62,NE_ASOS\n')
    config = yaml.safe_load(Path(__file__).with_name('defaults.yaml').read_text())
    config.update(start='2025-08-01',end='2025-11-01',cycles=[0,12],sample_every_n_days=7,
                  stations=str(station_file), splits={'train_end':'2025-09-01','val_end':'2025-10-01','heldout_stations':['HOLD'],'embargo_hours':0})
    config['satellite']['pixels'] = 32
    config['satellite']['prepare_action_crops'] = True
    for action in config['actions']:
        if action['kind'] == 'satellite':
            action['pixels'] = 16
    config_path = output/'data_config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    data_root = output/'data'
    schedule = plan(config_path, data_root)
    def factory(root, current):
        return {'iem': FakeIEM(), 'hrrr': FakeHRRR(), 'goes': FakeGOES(root,current['satellite'])}
    for index in range(len(schedule['shards'])):
        build_part(data_root, index, provider_factory=factory)
    quality_path = output/'quality.yaml'
    quality_path.write_text('required_splits: [train, val, test, test_spatial]\n')
    finalize(data_root, quality_path)
    return data_root


def complete_workflow(output, device='cpu'):
    output = Path(output).resolve()
    root = prepare_fixture(output)
    settings = yaml.safe_load((Path(__file__).parents[1]/'configs/experiments.yaml').read_text())
    settings.update(seeds=[3],learning_rates=[.005],budgets=[0.,3.15],policy_fraction=.3,bootstrap_replicates=10)
    settings['profiles'].pop('overview_hybrid')
    settings['training'].update(hidden=[8],embedding=4,batch_size=8,epochs=2,patience=2,workers=0,threads=1,dropout=.1,checkpoint_batches=1)
    settings['policy'].update(max_samples=100,states_per_sample=2)
    settings['baselines'].update(gp_max_train=16,gp_length_scales=[1.],ridge_alphas=[1.])
    path = output/'experiments.yaml'
    path.write_text(yaml.safe_dump(settings))
    runs = output/'runs'
    schedule = suite.prepare(root, path, runs)
    for index in range(len(schedule['runs'])):
        suite.train_task(root, runs, index, device)
    chosen = suite.select(runs)
    suite.fit_baselines(root, runs)
    for index in range(len(chosen['selected_ids'])):
        suite.evaluate_task(root, runs, index, device)
    report = suite.report(root, runs)
    report.update(synthetic=True, trained_runs=len(schedule['runs']), selected_runs=len(chosen['selected_ids']))
    write_json(output/'SELFTEST_COMPLETE.json',report)
    return report
