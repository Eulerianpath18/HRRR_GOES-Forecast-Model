# Checkpoints, failures and recovery

All commands below run from the project folder on ARC. First load settings:

```bash
source arc/config.sh
```

## What survives an interrupted job

| Stage | Saved work | Resume point |
|---|---|---|
| Import | Raw caches; per-initialization tables, metadata and commit marker | First uncommitted/failed initialization in each month |
| Finalization | Monthly imports remain intact | Rebuild final tables and acceptance checks |
| Neural training | `last.pt`: weights, AdamW, RNG, epoch, within-epoch position, history | Last committed minibatch boundary |
| Best predictor | `best.pt` | Lowest validation loss seen so far |
| Statistical baselines | `progress.joblib` | Next unfinished model family |
| Policy calibration | `progress.pt`, every 50 cases | Last committed calibration case |
| Test inference | 128-sample parquet parts | Next incomplete part |
| Reporting | Predictions and metric files | Regenerate reports from saved predictions |

Files are written to temporary paths before atomic replacement. Completion markers
are written last. OS-backed locks reject a second writer; the operating system
releases the lock when a process dies. Do not delete lock files to force concurrent
writers. Partial temporary files are not accepted as completed checkpoints.

Neural checkpoints are saved every 100 batches and at every completed epoch.
SIGUSR1/SIGTERM/SIGINT requests a safe checkpoint and exit 75. Slurm requests a
signal five minutes before the limit. `arc/job.sh` attempts to requeue a job that
exits 75, up to ten restarts; site policy must permit it. Hard kills, crashes or
out-of-memory termination can only recover from the last already committed file.
A single long network request/initialization might outlast the warning interval.
Automatic requeue is not guaranteed; manual stage resubmission remains supported.

## Inspect the failure first

Find the relevant job ID using `squeue -u "$USER"`, then:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
ls "$FORECAST_STORAGE/logs"
```

Read that job's log, plus the relevant `status.json`, `failures.json` or
`quality.json`. The usual causes are an unavailable allocation/GPU, public-source
outage, quota exhaustion, invalid data coverage or a changed configuration.

Cancel obsolete dependent jobs by their specific IDs before resubmitting a stage:

```bash
scancel OBSOLETE_JOB_ID
```

Do not cancel all of your jobs indiscriminately. A job waiting on a permanently
failed dependency may remain pending; resubmission makes a new dependency chain.

## Retry import

After resolving network/storage issues and ensuring the old import jobs have stopped:

```bash
bash arc/submit.sh import
```

Intact completed monthly tasks are skipped. Failed tasks reuse intact completed
initializations and raw files. Acquisitions recorded as failures are retried.
The report records remaining failures and the acceptance policy defaults to zero
source failures. A build may finish with explicit missing sources (`strict: false`)
but finalization will not approve them automatically.

Corrupt raw GOES, IEM and decoded HRRR caches are detected where validated and
quarantined/retried. Final acceptance reads/decompresses and hashes all referenced
images. If an already committed month's image is corrupt, rebuild that month with
`build-part --rebuild`; otherwise its completion marker would skip the month.
An import task index is the zero-based position in `plan.json` (0 = January 2021).
On an allocated CPU node:

```bash
"$FORECAST_ENV/bin/python" -m forecastprep.cli build-part "$FORECAST_DATA" --index 0 --rebuild --retry-failed
```

Then resubmit `import` to rerun finalization. Do not run this heavy command on the
login node. `--rebuild` resets progress markers for that month; it keeps usable
raw caches. Do not re-import or mutate an accepted archive while training reads it.

If a source is permanently unavailable, inspect the affected station/lead/split in
`coverage.csv`. Do not silently weaken thresholds or invent labels. A deliberate
change of period, station set or data-acceptance policy is a documented experiment
decision. Changed data configurations require a new `FORECAST_DATA` directory.

## Retry training or downstream work

If a training job was interrupted and its settings remain unchanged:

```bash
bash arc/submit.sh train
```

Completed intact runs are skipped; incomplete ones load `last.pt`. The new chain
also runs downstream selection, baselines, policy fitting, evaluation and reporting.

If all training runs are complete and only later work failed:

```bash
bash arc/submit.sh finish
```

This reuses finished predictors and any intact policy/baseline/evaluation progress.
Never edit or delete `best.pt`/`last.pt` to force a mismatched resume. If a committed
checkpoint is corrupt, restore a backup or use a fresh run directory. Failed partial
writes preserve the previous committed checkpoint; storage corruption is different.

## Identity and quality checks

Import configuration and station metadata are fingerprinted. Accepted tables and
manifests are checked by SHA-256 at run startup. Image acceptance records SHA-256,
size and mtime; startup rehashes images whose mtime changed and rejects size changes.
The deep finalization check reads all images. Size/mtime checking is a practical
integrity check, not a defence against deliberate tampering that preserves both.

Training and experiment identities include dataset, configuration and package-source
hashes. Policy/evaluation identities also include the chosen checkpoint. Do not
edit code while a run is active. To change model/batch size after an OOM, stop the
old chain, change `configs/experiments.yaml`, choose a new `FORECAST_RUNS`, and prepare
another experiment plan on an allocated node:

```bash
"$FORECAST_ENV/bin/python" -m forecastprep.cli prepare-experiments "$FORECAST_DATA" --config configs/experiments.yaml --output "$FORECAST_RUNS"
```

Then submit `train`. A larger Slurm memory request with unchanged model settings
can resume the original run. CUDA and filesystem memory exhaustion are distinct.

Defaults reject low label/current/HRRR/overview coverage separately by required
split, station and lead (90%, 90%, 99%, 95%, respectively), invalid temperatures,
unexpected or duplicate keys, timing violations, bad residuals and corrupt images.
All model comparisons enforce identical paired evaluation IDs. Nonfinite training
loss, gradients or predictions raise an error. No silent CPU fallback occurs when
a GPU was requested.
