#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTION=run
if [[ "${1:-}" == "--stop" ]]; then ACTION=stop; shift; fi
MODEL="${1:-${ROOT}/glm-5.3-flash-2x-dgx}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--stop" ]]; then ACTION=stop; shift; fi
if [[ "${1:-}" == "--dry-run" ]]; then ACTION=dry-run; shift; fi
exec python3 "${ROOT}/scripts/glm53_tp2_runtime.py" "$ACTION" "$MODEL" "$@"
