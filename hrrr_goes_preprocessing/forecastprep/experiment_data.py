"""Shared sample cohorts and reproducible acquisition masks for training."""
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .torch_data import AcquisitionSession


def cohort(store, split):
    frame = store.rows
    valid = np.isfinite(frame[['observed_t2m_c', 'hrrr_t2m_c', 'obs_current_c']]).all(axis=1)
    return frame.index[(frame.split == split) & valid].tolist()


class ExperimentDataset(Dataset):
    def __init__(self, store, ids, actions, acquisition='fixed', budgets=(0., 1., 2., 4.), seed=0, epoch=0):
        self.store, self.ids = store, list(ids)
        self.actions, self.acquisition = list(actions), acquisition
        self.budgets, self.seed, self.epoch = budgets, seed, epoch
        if set(actions) - set(store.actions):
            raise ValueError('Configured actions unavailable in this mode')
        self.maximum = sum(store.actions[name]['cost_units'] for name in actions)
        self.target = 'observed_t2m_c' if store.mode == 'direct' else 'error_c'

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        token = f'{self.seed}:{self.epoch}:{sid}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], 'little'))
        budget = float(rng.choice(self.budgets)) if self.acquisition == 'random' else self.maximum
        session = AcquisitionSession(self.store, sid, budget)
        if self.acquisition == 'random':
            while True:
                options = [name for name in session.available_actions() if name in self.actions]
                if not options or rng.random() < .15:
                    break
                session.acquire(str(rng.choice(options)))
        else:
            for name in self.actions:
                session.acquire(name)
        return {'inputs': session.observation(), 'target': torch.tensor(float(self.store.rows.loc[sid, self.target]), dtype=torch.float32),
                'sample_id': sid}


def training_partitions(store, policy_fraction=.2):
    if not 0 < policy_fraction < .5:
        raise ValueError('policy_fraction must be between 0 and .5')
    ids = cohort(store, 'train')
    frame = store.rows.loc[ids]
    times = sorted(frame.init_utc.unique())
    if len(times) < 4:
        raise ValueError('Need at least four training initializations for policy holdout')
    boundary = times[min(len(times)-1, max(1, int(len(times)*(1-policy_fraction))))]
    tolerance = pd.Timedelta(minutes=store.manifest['config']['label_tolerance_minutes'])
    predictor = frame.index[(frame.init_utc < boundary) & (frame.valid_utc + tolerance < boundary)].tolist()
    policy = frame.index[frame.init_utc >= boundary].tolist()
    if not predictor or not policy:
        raise ValueError('Empty predictor or policy training partition')
    return predictor, policy, str(boundary)
