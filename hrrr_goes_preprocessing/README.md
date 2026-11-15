# HRRR + GOES forecasting experiments — 0.4.0

This project imports station observations, HRRR forecasts and GOES imagery;
trains numerical and satellite models in PyTorch; fits statistical baselines;
learns budgeted acquisition policies; and produces paired test-set reports.

**Start with [START_HERE_ARC.md](START_HERE_ARC.md).** The full dataset downloads
on ARC; it does not need to pass through your PC.

**Desktop GUI:** run `forecast_gui.py` locally. It supports station selection,
YAML/station-list imports, recent input downloads, satellite viewing, numerical
edits and CPU prediction from portable models exported after ARC training.
See [GUI_GUIDE.md](GUI_GUIDE.md). The full training archive stays on ARC.

## Prepared experiment

| Setting | Default |
|---|---|
| Dates | Every day of 2021–2024 |
| Forecast cycles / leads | 00 and 12 UTC / 6, 12, 24, 48 hours |
| Stations | 12 stations in `configs/stations_research.csv` |
| Training / validation / testing | 2021–22 / 2023 / 2024, with boundary purging |
| Geographic holdout | IML and LBF; their earlier rows are excluded from fitting |
| Satellite | GOES-16 C13 and C08; retained 1600 km, 800 × 800 patches |
| Modes | Direct temperature; MOS residual correction of HRRR |
| Neural models | Numerical MLP, overview hybrid, full hybrid, masked hybrid |
| Search | 2 learning rates × 3 seeds × 4 profiles × 2 modes = 48 runs |
| Selection | Lowest mean validation MSE across seeds; retain every selected seed |
| Baselines | Persistence, HRRR, climatology, bias MOS, linear, ridge, subset GP |
| Acquisition | Random, static, learned sequential policies at five budgets |

The requested grid contains **140,256 station/cycle/lead rows before exclusions**.
Images are shared across leads. Small action crops are prepared once to avoid
repeatedly resampling the large satellite patches during training.

After setup and a successful cluster self-test:

```bash
bash arc/submit.sh import
# Wait for import and finalization to succeed.
bash arc/submit.sh train
```

The training command schedules training, validation selection, policy fitting,
baseline fitting, test inference and reporting with Slurm dependencies.

## Outputs

| Artifact | Purpose |
|---|---|
| `data/.../plan.json` | Immutable monthly import plan |
| `shards/YYYY-MM/progress/` | Completed initialization checkpoints |
| `samples.parquet`, `optional.parquet` | Final sample tables |
| `quality.json`, `coverage.csv` | Acceptance checks and missingness |
| `READY.json` | Accepted dataset identity and file integrity signatures |
| `runs/.../experiments.json` | Frozen experiment configurations |
| `training/RUN/last.pt` | Model, optimizer, RNG, epoch and batch position for resumption |
| `training/RUN/best.pt` | Validation-selected predictor |
| `selected.json` | Frozen learning-rate selection |
| `policies/RUN/`, `baselines/` | Intermediate progress and fitted models |
| `results/` | Predictions, metric CSV files, figures and results notes |
| `gui_models/` | Portable models and fitted policies for the desktop GUI |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the protocol,
[docs/RECOVERY.md](docs/RECOVERY.md) for recovery, and
[VALIDATION.md](VALIDATION.md) for actual validation evidence.

## Scope

The comparison workflow is implemented. Production import and research training
still have to run on ARC. Synthetic training verifies software, not forecast skill.
ARC/CUDA execution cannot be certified without running on your allocation.

The GP uses a reproducible subset of at most 2,000 training cases. Conventional
statistical MOS is included; this version does not ingest NBM or another external
operational postprocessor. It does not establish novel-method or causal claims,
or validate generalization to arbitrary locations.

The original `build`, `audit`, `demo`, `fit-normalizer` and PyTorch Dataset
interfaces remain available. The original two-row real Denver sample is included.
The historical preprocessing reference is in `docs/PREPROCESSING_REFERENCE.md`;
this README and the experiment/recovery guides describe the current workflow.

Optional small local software check:

```bash
python -m pytest -q
python examples/cluster_selftest.py --output data/software_check --device cpu
python -m forecastprep.cli audit example_data/denver_smoke
```
