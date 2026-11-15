"""Resumable single-device training, with validation-only early stopping."""
import copy
import json
import math
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from .core import digest, write_json
from .runtime import locked, atomic_torch_save, StopRequest, status, source_id, sha256
from .archive import verify_ready
from .torch_data import FeatureStore, Standardizer
from .experiment_data import ExperimentDataset, cohort, training_partitions
from .models import ForecastModel, to_device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def get_device(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no GPU is usable. Check Slurm allocation and the PyTorch build.')
    return torch.device(requested)


def make_model(store, config):
    return ForecastModel(len(store.columns), store.actions, kind=config['kind'],
                         hidden=config['hidden'], embedding=config.get('embedding', 16),
                         dropout=config.get('dropout', .1))


def run(root, directory, config, device='cpu', stop_after_batches=None):
    root, directory = Path(root), Path(directory)
    with locked(directory), StopRequest() as stop:
        try:
            return _run(root, directory, config, device, stop, stop_after_batches)
        except BaseException as exc:
            status(directory, 'interrupted' if isinstance(exc, SystemExit) and exc.code == 75 else 'failed',
                   error=str(exc), traceback=traceback.format_exc())
            raise


def _run(root, directory, config, requested_device, stop, stop_after_batches):
    ready = verify_ready(root)
    configuration_id = digest(config)
    identity = {'config_id': configuration_id, 'data_id': ready['data_id'], 'source_id': source_id()}
    status_path = directory / 'status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get('state') == 'complete':
            saved_config = json.loads((directory / 'run.json').read_text())
            if saved_config['identity'] != identity:
                raise ValueError('Completed run belongs to another dataset or configuration')
            if saved_config.get('source_id') != source_id():
                raise ValueError('Implementation changed since this run')
            if not (directory / 'best.pt').exists() or sha256(directory / 'best.pt') != previous.get('best_sha256'):
                raise ValueError('Completed run has a missing or changed best.pt')
            return previous
    device = get_device(requested_device)
    torch.set_num_threads(config.get('threads', 4))
    seed_all(config['seed'])
    store = FeatureStore(root, config['mode'])
    train_ids, policy_ids, policy_boundary = training_partitions(store, config.get('policy_fraction', .2))
    scaler = Standardizer.fit(root, sample_ids=train_ids)
    store.normalizer = scaler
    validation_ids = cohort(store, 'val')
    if not train_ids or not validation_ids:
        raise ValueError('Training and validation cohorts must both contain usable samples')
    model = make_model(store, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-4))
    epoch, offset, steps, bad_epochs, best = 0, 0, 0, 0, math.inf
    history = []
    checkpoint = directory / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Checkpoint configuration or dataset changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        epoch, offset, steps = saved['epoch'], saved['offset'], saved['steps']
        best, bad_epochs, history = saved['best'], saved['bad_epochs'], saved['history']
        restore_rng(saved['rng'])
    write_json(directory / 'run.json', {'identity': identity, 'config': config,
        'n_train': len(train_ids), 'n_policy': len(policy_ids), 'policy_boundary': policy_boundary, 'n_val': len(validation_ids), 'torch': torch.__version__,
        'device': str(device), 'source_id': source_id(), 'normalizer': scaler.values})
    status(directory, 'training', epoch=epoch, steps=steps)

    def save():
        atomic_torch_save(checkpoint, {'identity': identity, 'config': config, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'epoch': epoch, 'offset': offset,
            'steps': steps, 'best': best, 'bad_epochs': bad_epochs, 'history': history,
            'normalizer': scaler.values, 'fingerprint': scaler.fingerprint})

    batches_this_call = 0
    while epoch < config['epochs'] and bad_epochs < config['patience']:
        order = np.random.default_rng(config['seed'] + epoch).permutation(len(train_ids))
        remaining = [train_ids[i] for i in order[offset:]]
        train_data = ExperimentDataset(store, remaining, config['actions'], config['acquisition'],
                                       config['budgets'], config['seed'], epoch)
        loader = DataLoader(train_data, batch_size=config['batch_size'], num_workers=config.get('workers', 0),
                            generator=torch.Generator().manual_seed(config['seed'] + epoch),
                            timeout=config.get('worker_timeout_seconds',120) if config.get('workers',0) else 0,
                            pin_memory=device.type == 'cuda')
        model.train()
        for batch in loader:
            if stop.requested:
                save()
                raise SystemExit(75)
            inputs, target = to_device(batch['inputs'], device), batch['target'].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = torch.mean((prediction - target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss. Last committed checkpoint was preserved.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 5.), error_if_nonfinite=True)
            optimizer.step()
            offset += len(target)
            steps += 1
            batches_this_call += 1
            if steps % config.get('checkpoint_batches', 100) == 0:
                save()
                status(directory, 'training', epoch=epoch, steps=steps, last_batch_loss=float(loss))
            if stop.requested or (stop_after_batches and batches_this_call >= stop_after_batches):
                save()
                raise SystemExit(75)
        validation = ExperimentDataset(store, validation_ids, config['actions'], config['acquisition'],
                                       config['budgets'], 1729, -1)
        valid_loader = DataLoader(validation, batch_size=config['batch_size'], num_workers=config.get('workers', 0),
                                  timeout=config.get('worker_timeout_seconds',120) if config.get('workers',0) else 0,
                                  generator=torch.Generator().manual_seed(1729))
        model.eval()
        squared, count = 0., 0
        with torch.no_grad():
            for batch in valid_loader:
                error = model(to_device(batch['inputs'], device)) - batch['target'].to(device)
                if not torch.isfinite(error).all():
                    raise FloatingPointError('Nonfinite validation predictions')
                squared += float(error.square().sum())
                count += len(error)
        mse = squared / count
        history.append({'epoch': epoch + 1, 'validation_mse': mse, 'steps': steps})
        if mse < best - config.get('min_delta', 1e-5):
            best, bad_epochs = mse, 0
            atomic_torch_save(directory / 'best.pt', {'identity': identity, 'config': config,
                'model': model.state_dict(), 'epoch': epoch + 1, 'validation_mse': mse,
                'normalizer': scaler.values, 'fingerprint': scaler.fingerprint})
        else:
            bad_epochs += 1
        epoch, offset = epoch + 1, 0
        save()
        write_json(directory / 'history.json', history)
        print(f'{directory.name}: epoch={epoch} validation_MSE={mse:.6f} best={best:.6f}', flush=True)
        if stop.requested:
            raise SystemExit(75)
    status(directory, 'complete', best_validation_mse=best, epochs=epoch, steps=steps,
           best_sha256=sha256(directory / 'best.pt'), **identity)
    return json.loads(status_path.read_text())


def load_predictor(root, checkpoint_path, device='cpu'):
    ready = verify_ready(root)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if saved['identity']['data_id'] != ready['data_id']:
        raise ValueError('Prediction checkpoint belongs to another dataset')
    if saved['identity'].get('source_id') != source_id():
        raise ValueError('Implementation differs from the saved predictor; use its original project version')
    normalizer = Standardizer(saved['normalizer'], saved['fingerprint'])
    store = FeatureStore(root, saved['config']['mode'], normalizer)
    model = make_model(store, saved['config']).to(get_device(device))
    model.load_state_dict(saved['model'])
    model.eval()
    return store, model, saved
