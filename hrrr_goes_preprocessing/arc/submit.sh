#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"
exec "$FORECAST_ENV/bin/python" "$FORECAST_PROJECT/arc/submit.py" "$@"
