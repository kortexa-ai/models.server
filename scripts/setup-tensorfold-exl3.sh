#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/tensorfold_exl3_runtime.py" setup "$@"
