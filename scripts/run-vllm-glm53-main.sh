#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?model directory required}"
shift
ACTION=run
if [[ "${1:-}" == "--stop" ]]; then ACTION=stop; shift; fi
if [[ "${1:-}" == "--dry-run" ]]; then ACTION=dry-run; shift; fi
exec python3 "${ROOT}/scripts/glm53_main_runtime.py" "$ACTION" "$MODEL" "$@"
