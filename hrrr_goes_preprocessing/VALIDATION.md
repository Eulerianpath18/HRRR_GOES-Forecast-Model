# Validation of version 0.4.0

Updated 2026-11-15 UTC. The package-source fingerprint and a
machine-readable summary are in `VALIDATION.json`; dependencies are recorded in
`validated_versions.json` and `constraints-tested.txt`.

## Automated checks

**38 passed, 1 skipped** on Linux / Python 3.12.14 / PyTorch 2.5.1 CPU.
The skipped test checks exact resume with two DataLoader workers. This execution
environment restricts the IPC required by those workers; that test is enabled on
ARC by `FORECAST_TEST_MULTIPROCESS=1` in the submitted self-test. Worker waits have
a 120-second timeout so an IPC failure does not hang training indefinitely.

Checks include causal observation/image timing, label matching, unit conversion,
missing/conflicting reports, HRRR grid selection, image masking/reprojection,
purged temporal and station splits, train-only normalization, direct-mode HRRR
exclusion, hard acquisition budgets, unacquired-image encoder exclusion, image
integrity, coverage rejection, identical prepared/dynamic crops, and block-bootstrap
arithmetic across pandas datetime resolutions.

Recovery was tested by interrupting a small training run mid-epoch and comparing
its resumed weights/history with uninterrupted CPU training: they were identical.
A simulated provider failure resumed after the previously completed initialization.
Interrupted baseline fitting produced identical predictions after resumption.
Concurrent-writer rejection and changed-dataset rejection were exercised.

Slurm training and finish dependency chains were checked using a mocked scheduler.
Shell syntax, Python compilation, package installation and `pip check` passed.
Mock submission does not test ARC permissions, resource availability or requeue.

## Desktop GUI and portable inference

The GUI was launched and visually inspected on Linux. Loading the historical
example, changing DEN to COS, updating coordinates/input timestamps and satellite
imagery, and opening the enlarged satellite window were exercised. Windows
execution was not available for testing; the launcher and paths are supplied for
the user's existing Windows interpreter.

The new tests check initialization-cycle selection at the publication-delay
boundary; background-thread input acquisition without signal-handler errors;
station selection; missing-station rejection; finite budget and editable-feature
checks; and configuration/image-geometry compatibility. Portable MOS predictions
match the training predictor with the HRRR residual added exactly once. A reduced
single-cycle snapshot with no verification labels, READY marker or training archive
produces the same prediction using the saved training normalizer.

The complete synthetic workflow was rerun with v0.4 code. All six selected models
were exported, reloaded and used for finite CPU forecasts. Both masked models
respected a 0.05-unit acquisition budget. These synthetic weights are not included
as usable weather models.

A real input-only download was also made through `fetch_inputs` in a background
thread, replaying 2025-08-01 00 UTC at DEN with GOES-19 and 6/12-hour leads. It
returned two HRRR/overview rows, zero future labels, no timing violations and no
source failures. This tests the public-source acquisition path against a known
historical cycle, not guaranteed availability of today's latest cycle. Its small
check report is in `docs/gui_input_check.json`; raw downloads are not distributed.

## Complete synthetic workflow

A three-shard synthetic archive was imported and accepted, then six small models
were trained for two epochs (direct/MOS × numerical/full/masked profiles). The
workflow selected predictors, fitted all statistical baselines and both acquisition
policies, evaluated temporal and geographic test sets, and generated metric tables
and figures. It produced 1,040 prediction rows across the comparisons.

This exercised the same Python entry points used by the ARC jobs. It used one
seed, one learning rate, smaller images/networks and fewer epochs than production.
The synthetic losses/metrics are software checks and are not model-skill evidence.

## Real 2021 source check (retained from v0.3 validation)

The public archives were read for 2021-01-01 00 UTC at all 12 configured stations
and all four leads, using GOES-16 C13/C08 and small 64 × 64 retained test patches:

| Check | Result |
|---|---:|
| Planned/sample rows | 48 / 48 |
| HRRR temperature forecasts | 48 |
| Linked satellite overviews | 48 |
| Matched verification temperatures | 46 |
| Current observations | 48 |
| Missing optional numerical values | 0 |
| Recorded source failures / timing violations | 0 / 0 |
| Maximum nearest-HRRR-cell distance | 1.891 km (rounded up) |

The two missing verification labels are Garden City (GCK), leads 6 and 12 hours.
The corresponding reports are absent from the retrieved archive within the
20-minute label window. They remain missing; neither the tolerance nor the labels
was altered to make the check pass. Source preflight permits at least 90% overall
label coverage. Full dataset acceptance applies separate coverage thresholds by
station, lead and split.

Processed artifacts, source metadata and small images are included under
`example_data/source_check_2021`; raw downloads are excluded. This example has
`unsplit` rows and must not be mistaken for a training dataset. The earlier real
Denver August 2025 example is also retained under `example_data/denver_smoke`.

## What still needs ARC

No ARC account or GPU allocation was used here. CUDA execution, worker IPC on ARC,
Slurm scheduling/requeue, account quotas/network access, full four-year coverage,
production throughput and research training remain to be verified on your cluster
allocation. `bash arc/submit.sh selftest` performs the cluster-specific checks before
full import. Passing it is required by the run guide.

The suite cannot guarantee model skill, policy benefit, optimal hyperparameters,
scientific novelty, exact bitwise reproducibility across GPU types, or operational
availability of historical data. It records publication/latency assumptions and
uses configured acquisition-cost proxies; see `docs/EXPERIMENTS.md`.
