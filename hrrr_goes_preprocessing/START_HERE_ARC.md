# Start here: run the project on UCalgary ARC

Once your ARC account is active: upload this project, set storage/allocation
settings, run the cluster self-test, import the archive, and launch experiments.
The full dataset downloads directly on ARC.

**Windows** commands run in PowerShell on your PC. **ARC** commands run in your SSH
terminal. Replace `YOUR_UCALGARY_USERNAME` with your actual account username.

## 1. Upload and extract

**Windows:** save the ZIP as `hrrr_goes_arc_ready.zip` in Downloads. Connect to the
UCalgary VPN if off campus, then run:

```powershell
scp "$env:USERPROFILE\Downloads\hrrr_goes_arc_ready.zip" YOUR_UCALGARY_USERNAME@arc.ucalgary.ca:~/
ssh YOUR_UCALGARY_USERNAME@arc.ucalgary.ca
```

**ARC:** extract into a new directory, preserving any existing project:

```bash
mkdir -p ~/projects/forecasting_v04
unzip ~/hrrr_goes_arc_ready.zip -d ~/projects/forecasting_v04
cd ~/projects/forecasting_v04/hrrr_goes_preprocessing
```

If you uploaded the extracted folder through ARC's file interface instead, enter
that folder: the directory containing `pyproject.toml` and `arc/`.

## 2. Set storage and allocation

**ARC:** open the settings file:

```bash
nano arc/config.sh
```

Defaults: storage `$HOME/forecast-research`, CPU partition `cpu2021`, GPU partition
`gpu-a100`, one concurrent import task and two concurrent GPU jobs. Save changes
with Ctrl+O, Enter; exit with Ctrl+X.

If ARC provides a project storage directory, change the storage line to its actual
absolute path. Example only:

```bash
export FORECAST_STORAGE="/work/YOUR_ALLOCATED_DIRECTORY/forecast-research"
```

Set `FORECAST_ACCOUNT` only if your allocation requires an account code. Use
`arc.hardware` or `sinfo` to check accessible partitions. Each neural run requests
one GPU, four CPUs and 32 GB RAM. Runs are independent; no multi-GPU setup is needed.

The full archive needs substantial persistent storage. The initial 300 GB free-space
check is a configurable guard, **not a measured total or quota guarantee**. Raw
files, retained patches and prepared crops all count. Filesystem free space is
not your personal quota. Arrange sufficient storage with ARC support and inspect
growth after the first month. Do not keep the only copy of data/checkpoints in
job scratch space.

## 3. Install the environment once

If Conda is missing, follow the official [Conda on ARC instructions](https://rcs.ucalgary.ca/Conda_on_ARC)
for Miniforge. The setup script uses `conda` or `~/software/init-conda`.
From this project folder on **ARC**:

```bash
bash arc/setup.sh
```

This creates a separate Python 3.12 environment, installs PyTorch 2.5.1 with CUDA
12.4 wheels and the tested dependency constraints, then records installed versions.
Do not copy your Windows `.venv` to ARC.

## 4. Run the cluster self-test

```bash
bash arc/submit.sh selftest
```

This submits a batch job to check GPU execution, dependencies, writes, source
access, the automated tests, a tiny full synthetic experiment, and one real
2021 initialization at all 12 stations and four leads. The source check uses
64 × 64 stored images; the research configuration remains 800 × 800.

Check jobs and logs:

```bash
squeue -u "$USER"
source arc/config.sh
ls "$FORECAST_STORAGE/logs"
```

The submission prints a job ID. Substitute that number below:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Proceed when the self-test job is `COMPLETED` with exit code `0:0` and both files exist:

```bash
cat "$FORECAST_STORAGE/selftest/workflow/SELFTEST_COMPLETE.json"
cat "$FORECAST_STORAGE/selftest/sources/SOURCE_CHECK_COMPLETE.json"
```

If it fails, the log identifies the failed check. See [docs/RECOVERY.md](docs/RECOVERY.md).
The synthetic scores have no research meaning.

## 5. Import all four years

```bash
bash arc/submit.sh import
```

This submits 48 monthly tasks and a dependent finalization job. It includes every
day of 2021–2024, both cycles, 12 stations, four leads and two satellite channels.
One import runs at a time by default to limit concurrent public-archive requests.

Do not launch training until finalization succeeds:

```bash
source arc/config.sh
cat "$FORECAST_DATA/status.json"
ls "$FORECAST_DATA/READY.json" "$FORECAST_RUNS/experiments.json"
```

The data status must be `ready`. Finalization checks sample keys, time alignment,
coverage and image integrity. It rejects incomplete/low-coverage data. Inspect
`quality.json` and `coverage.csv` if it rejects the dataset.

After a failure, use the recovery guide before resubmitting. Intact completed work
is retained. Avoid concurrent copies of a stage; cancel obsolete dependent jobs.

## 6. Train and generate results

```bash
bash arc/submit.sh train
```

This schedules all 48 training configurations. Dependent jobs select settings on
validation data, fit baselines and acquisition policies, evaluate both test sets,
and generate reports. You may close SSH after submission.

Once the report job completes:

```bash
source arc/config.sh
ls "$FORECAST_RUNS/results"
```

Start with `metrics_across_seeds.csv`, `metrics_overall.csv`, the lead plots and
budget curves. `predictions.parquet` contains every test prediction.

The report job also exports portable neural models to `$FORECAST_RUNS/gui_models`.
Download that folder to use the desktop GUI on your PC; you do not need to copy
the full training dataset. See [GUI_GUIDE.md](GUI_GUIDE.md) for the command and
location/configuration controls. Use this v0.4 package both on ARC and locally;
source identity checks intentionally reject checkpoints from older package code.

**Windows:** download the results folder. Change the remote path if you changed storage:

```powershell
scp -r YOUR_UCALGARY_USERNAME@arc.ucalgary.ca:~/forecast-research/runs/experiment_v1/results "$env:USERPROFILE\Downloads\forecast_results"
```

Keep the accepted data, experiment plans, `best.pt`/`last.pt`, policies, baselines,
and environment lock for reproducibility. Do not edit code or configuration midway
through an experiment; resume checks enforce the original identities.

## Official cluster references

Checked 2026-11-14: [ARC guide](https://rcs.ucalgary.ca/ARC_Cluster_Guide),
[GPU requests](https://rcs.ucalgary.ca/How_to_request_GPUs_for_batch_jobs_on_ARC),
[Conda](https://rcs.ucalgary.ca/Conda_on_ARC),
[PyTorch wheels](https://pytorch.org/get-started/previous-versions/).
Your allocation, drivers, network egress and storage are checked on ARC; local CPU
validation cannot verify those account-specific conditions.
