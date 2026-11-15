"""Illustrate sequential replay. This fixed ordering is NOT a learned policy."""
import argparse
from forecastprep.torch_data import FeatureStore, AcquisitionSession


def main():
    p=argparse.ArgumentParser(); p.add_argument('root'); args=p.parse_args()
    store=FeatureStore(args.root,mode='mos')
    session=AcquisitionSession(store,store.rows.index[0],budget=1.1)
    state=session.observation()
    for candidate in ['history','hrrr_wind','water_vapour']:
        if candidate in session.available_actions(): state=session.acquire(candidate)
    print('Acquired:',sorted(session.acquired))
    print('Optional compute cost:',session.spent)
    print('Additional source bytes:',session.source_bytes)
    print('Remaining:',state['remaining_budget'].item())

if __name__=='__main__': main()
