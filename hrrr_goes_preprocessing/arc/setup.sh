#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$FORECAST_STORAGE"
if ! command -v conda >/dev/null 2>&1; then
    if [[ -f "$HOME/software/init-conda" ]]; then
        source "$HOME/software/init-conda"
    else
        echo 'Activate your ARC Conda installation first. See https://rcs.ucalgary.ca/Conda_on_ARC' >&2
        exit 1
    fi
fi
if [[ ! -x "$FORECAST_ENV/bin/python" ]]; then
    conda create -y --prefix "$FORECAST_ENV" python=3.12 pip
fi
"$FORECAST_ENV/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
"$FORECAST_ENV/bin/python" -m pip install -c "$FORECAST_PROJECT/constraints-tested.txt" -e "$FORECAST_PROJECT[test]"
"$FORECAST_ENV/bin/python" -m pip check
"$FORECAST_ENV/bin/python" -m pip freeze > "$FORECAST_STORAGE/environment-lock.txt"
echo 'Environment ready. Submit a cluster self-test with: bash arc/submit.sh selftest'
