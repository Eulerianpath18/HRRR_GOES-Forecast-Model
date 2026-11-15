"""Submit dependency chains; reuse checkpoints when a stage is resubmitted."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

project = Path(os.environ['FORECAST_PROJECT'])
data = Path(os.environ['FORECAST_DATA'])
runs = Path(os.environ['FORECAST_RUNS'])
logs = Path(os.environ['FORECAST_STORAGE']) / 'logs'
logs.mkdir(parents=True, exist_ok=True)


def command(*args):
    subprocess.run([sys.executable, '-m', 'forecastprep.cli', *map(str,args)], cwd=project, check=True)


def submit(stage, gpu=False, count=None, dependencies=()):
    partition = os.environ['FORECAST_GPU_PARTITION' if gpu else 'FORECAST_CPU_PARTITION']
    args = ['sbatch','--parsable','--export=ALL','--job-name=forecast-'+stage,
            '--output='+str(logs/'%x-%A_%a.log'), '--partition='+partition,
            '--cpus-per-task=4','--mem=32G','--time=23:30:00']
    if os.environ.get('FORECAST_ACCOUNT'):
        args.append('--account='+os.environ['FORECAST_ACCOUNT'])
    if gpu:
        args.append('--gres=gpu:1')
    if count is not None:
        concurrency = os.environ['FORECAST_GPU_CONCURRENCY' if gpu else 'FORECAST_IMPORT_CONCURRENCY']
        args.append(f'--array=0-{count-1}%{concurrency}')
    if dependencies:
        args.append('--dependency=afterok:'+':'.join(dependencies))
    args += [str(project/'arc/job.sh'), stage]
    result = subprocess.run(args, cwd=project, check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected sbatch response: '+result.stdout)
    print(stage, job, flush=True)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['selftest','import','train','finish'])
    stage = parser.parse_args().stage
    if stage == 'selftest':
        submit('selftest', gpu=True)
    elif stage == 'import':
        minimum = os.environ['FORECAST_MIN_FREE_GB'] if not (data/'plan.json').exists() else '0'
        command('doctor','--directory',data,'--min-free-gb',minimum)
        command('plan','--config',project/'configs/research.yaml','--output',data)
        schedule = json.loads((data/'plan.json').read_text())
        job = submit('import', count=len(schedule['shards']))
        submit('finalize', dependencies=[job])
    else:
        plan = json.loads((runs/'experiments.json').read_text())
        jobs = []
        if stage == 'train':
            jobs.append(submit('train', gpu=True, count=len(plan['runs'])))
        selection = submit('select', dependencies=jobs)
        baseline = submit('baselines', dependencies=[selection])
        settings = plan['settings']
        count = len(settings['modes'])*len(settings['profiles'])*len(settings['seeds'])
        evaluation = submit('evaluate', gpu=True, count=count, dependencies=[selection])
        submit('report', dependencies=[baseline,evaluation])


if __name__ == '__main__':
    main()
