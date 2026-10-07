#!/usr/bin/env bash
set -euo pipefail
MODE=run
if [[ "${1:-}" == --stop ]]; then MODE=stop; shift; fi
exec python3 "$(dirname "$0")/tensorfold_exl3_runtime.py" "$MODE" "$@"
