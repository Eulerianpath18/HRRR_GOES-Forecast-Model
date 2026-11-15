"""Preprocessing, resumable experiments, and deployment diagnostics."""
from pathlib import Path
import argparse
import json
import logging
from .core import load_config
from .pipeline import build, audit


def main():
    parser = argparse.ArgumentParser(description='HRRR/GOES experiments for PyTorch and ARC')
    sub = parser.add_subparsers(dest='command', required=True)
    build_parser = sub.add_parser('build')
    build_parser.add_argument('--config', required=True)
    build_parser.add_argument('--output', required=True)
    sub.add_parser('audit').add_argument('root')
    normalizer = sub.add_parser('fit-normalizer')
    normalizer.add_argument('root')
    normalizer.add_argument('--output')
    sub.add_parser('demo').add_argument('--output', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--config', required=True)
    plan.add_argument('--output', required=True)
    part = sub.add_parser('build-part')
    part.add_argument('root')
    part.add_argument('--index', required=True, type=int)
    part.add_argument('--retry-failed', action='store_true')
    part.add_argument('--rebuild', action='store_true')
    finish = sub.add_parser('finalize')
    finish.add_argument('root')
    finish.add_argument('--policy', default='configs/quality.yaml')
    finish.add_argument('--skip-deep-images', action='store_true')
    experiments = sub.add_parser('prepare-experiments')
    experiments.add_argument('root')
    experiments.add_argument('--config', default='configs/experiments.yaml')
    experiments.add_argument('--output', required=True)
    for name in ['train-task', 'evaluate-task']:
        item = sub.add_parser(name)
        item.add_argument('root')
        item.add_argument('--runs', required=True)
        item.add_argument('--index', required=True, type=int)
        item.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    sub.add_parser('select').add_argument('--runs', required=True)
    for name in ['baselines','report']:
        item = sub.add_parser(name)
        item.add_argument('root')
        item.add_argument('--runs', required=True)
    check = sub.add_parser('doctor')
    check.add_argument('--directory', required=True)
    check.add_argument('--gpu', action='store_true')
    check.add_argument('--network', action='store_true')
    check.add_argument('--min-free-gb', type=float, default=0.)
    export = sub.add_parser('export-gui')
    export.add_argument('root')
    export.add_argument('--runs', required=True)
    export.add_argument('--output', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    result = None
    if args.command == 'build':
        result = build(load_config(args.config), args.output)
    elif args.command == 'audit':
        result = audit(args.root)
        if result['violations']:
            print(json.dumps(result, indent=2))
            raise SystemExit(1)
    elif args.command == 'fit-normalizer':
        from .torch_data import Standardizer
        path = args.output or str(Path(args.root)/'normalizer.json')
        Standardizer.fit(args.root, path)
        result = {'path': path}
    elif args.command == 'demo':
        from .demo import make_demo
        result = make_demo(args.output)
    elif args.command in ['plan','build-part','finalize']:
        from . import archive
        if args.command == 'plan':
            result = archive.plan(args.config, args.output)
        elif args.command == 'build-part':
            result = archive.build_part(args.root, args.index, args.retry_failed, rebuild=args.rebuild)
        else:
            result = archive.finalize(args.root, args.policy, not args.skip_deep_images)
    elif args.command == 'doctor':
        from .doctor import diagnose
        result = diagnose(args.directory, args.gpu, args.network, args.min_free_gb)
    elif args.command == 'export-gui':
        from .desktop import export_selected
        result = export_selected(args.root, args.runs, args.output)
    else:
        from . import suite
        if args.command == 'prepare-experiments':
            result = suite.prepare(args.root, args.config, args.output)
        elif args.command == 'train-task':
            result = suite.train_task(args.root, args.runs, args.index, args.device)
        elif args.command == 'evaluate-task':
            result = suite.evaluate_task(args.root, args.runs, args.index, args.device)
        elif args.command == 'select':
            result = suite.select(args.runs)
        elif args.command == 'baselines':
            result = suite.fit_baselines(args.root, args.runs)
        else:
            result = suite.report(args.root, args.runs)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
