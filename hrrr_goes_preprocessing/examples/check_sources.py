"""Check one real 2021 cycle, every research station, and all four lead times."""
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from forecastprep.core import load_config, write_json, stations
from forecastprep.pipeline import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    config = load_config(Path(__file__).resolve().parents[1]/'configs/research.yaml')
    config.update(start='2021-01-01', end='2021-01-02', cycles=[0], splits=None, strict=True)
    config['satellite']['pixels'] = 64  # Geometry/source check only; production remains 800.
    report = build(config, args.output)
    expected = len(stations(config['stations'])) * len(config['leads'])
    if report['failures'] or report['violations'] or any(report[k] != expected for k in ['rows','hrrr_rows','overview_rows']):
        raise RuntimeError('Source preflight failed: ' + str(report))
    # Real observing archives have gaps. Retain missing labels; never fabricate them.
    if report['labelled_rows'] / expected < .9:
        raise RuntimeError('Source preflight label coverage is below 90%: ' + str(report))
    samples = pd.read_parquet(Path(args.output)/'samples.parquet')
    missing_labels = samples.loc[samples.observed_t2m_c.isna(), 'sample_id'].tolist()
    optional = pd.read_parquet(Path(args.output)/'optional.parquet')
    if not np.isfinite(optional.drop(columns='sample_id').to_numpy(dtype=float)).all():
        raise RuntimeError('Missing optional numerical features in source preflight')
    write_json(Path(args.output)/'SOURCE_CHECK_COMPLETE.json', {'synthetic': False, 'report': report,
               'image_pixels': 64, 'missing_labels': missing_labels,
               'purpose': 'Source availability and alignment, not forecast-skill estimation'})
    print(report)


if __name__ == '__main__':
    main()
