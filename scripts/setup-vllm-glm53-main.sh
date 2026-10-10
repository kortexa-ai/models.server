#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-}" == "--build" ]]; then
    exec python3 "${ROOT}/glm-5.3-flash-2x-dgx-experimental/build/build.py"
fi
MODEL="${1:-${ROOT}/glm-5.3-flash-2x-dgx-experimental}"
exec python3 "${ROOT}/scripts/glm53_main_runtime.py" setup "$MODEL"
