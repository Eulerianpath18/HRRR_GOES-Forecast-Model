#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@300
#SBATCH --requeue
set -uo pipefail
source "${FORECAST_PROJECT:?FORECAST_PROJECT must be exported}/arc/config.sh"
# Slurm spools this script elsewhere, so use the project path passed in the environment.
cd "$FORECAST_PROJECT" || exit 1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON="$FORECAST_ENV/bin/python"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
case "$1" in
  import)
    srun "$PYTHON" -m forecastprep.cli doctor --directory "$FORECAST_DATA" --network || exit $?
    srun "$PYTHON" -m forecastprep.cli build-part "$FORECAST_DATA" --index "$TASK" --retry-failed
    ;;
  finalize)
    srun "$PYTHON" -m forecastprep.cli finalize "$FORECAST_DATA" --policy configs/quality.yaml &&
    srun "$PYTHON" -m forecastprep.cli prepare-experiments "$FORECAST_DATA" --config configs/experiments.yaml --output "$FORECAST_RUNS"
    ;;
  train)
    srun "$PYTHON" -m forecastprep.cli doctor --directory "$FORECAST_RUNS" --gpu || exit $?
    srun "$PYTHON" -m forecastprep.cli train-task "$FORECAST_DATA" --runs "$FORECAST_RUNS" --index "$TASK" --device cuda
    ;;
  select)
    srun "$PYTHON" -m forecastprep.cli select --runs "$FORECAST_RUNS"
    ;;
  baselines)
    srun "$PYTHON" -m forecastprep.cli baselines "$FORECAST_DATA" --runs "$FORECAST_RUNS"
    ;;
  evaluate)
    srun "$PYTHON" -m forecastprep.cli doctor --directory "$FORECAST_RUNS" --gpu || exit $?
    srun "$PYTHON" -m forecastprep.cli evaluate-task "$FORECAST_DATA" --runs "$FORECAST_RUNS" --index "$TASK" --device cuda
    ;;
  report)
    srun "$PYTHON" -m forecastprep.cli report "$FORECAST_DATA" --runs "$FORECAST_RUNS" &&
    srun "$PYTHON" -m forecastprep.cli export-gui "$FORECAST_DATA" --runs "$FORECAST_RUNS" --output "$FORECAST_RUNS/gui_models"
    ;;
  selftest)
    srun "$PYTHON" -m forecastprep.cli doctor --directory "$FORECAST_STORAGE/selftest" --gpu --network &&
    FORECAST_TEST_MULTIPROCESS=1 srun "$PYTHON" -m pytest -q &&
    srun "$PYTHON" examples/cluster_selftest.py --output "$FORECAST_STORAGE/selftest/workflow" --device cuda &&
    srun "$PYTHON" examples/check_sources.py --output "$FORECAST_STORAGE/selftest/sources"
    ;;
  *) echo "Unknown stage: $1" >&2; exit 2 ;;
esac
RESULT=$?
if [[ "$RESULT" == 75 ]]; then
    RESTARTS="${SLURM_RESTART_COUNT:-0}"
    if (( RESTARTS < FORECAST_MAX_RESTARTS )); then
        echo "Checkpoint committed. Requesting requeue, restart $RESTARTS."
        scontrol requeue "$SLURM_JOB_ID" && exit 0
    fi
    echo 'Resume limit reached or requeue denied. Resubmit this stage to continue from checkpoints.' >&2
fi
exit "$RESULT"
