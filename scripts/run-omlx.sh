#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(CDPATH= cd "$1" && pwd)"; shift
SCRIPTS_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${OMLX_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} is not configured for oMLX." >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-${ROOT}/.venv-omlx}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
OMLX_BIN="${OMLX_BIN:-${VENV_PATH}/bin/omlx}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${OMLX_BIN}" ]]; then
    echo "Error: patched oMLX is not installed in ${VENV_PATH}." >&2
    echo "Run scripts/setup-omlx.sh first." >&2
    exit 1
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"
MODEL_ROOT="${OMLX_MODEL_ROOT:-${ROOT}/.cache/omlx-models}"
MODEL_PATH="${MODEL_ROOT}/${MODEL_ID}"
BASE_PATH="${OMLX_BASE_PATH:-${ROOT}/.cache/omlx-state/${MODEL_ID}}"

mkdir -p "${MODEL_ROOT}" "${BASE_PATH}"

SNAPSHOT_PATH="$(
    OMLX_MODEL_REPO="${OMLX_REPO}" "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

print(snapshot_download(os.environ["OMLX_MODEL_REPO"]))
PY
)"
ln -sfn "${SNAPSHOT_PATH}" "${MODEL_PATH}"

OMLX_CONFIG_PATH="${MODEL_DIR}/model.json" \
OMLX_BASE_PATH="${BASE_PATH}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

config = json.loads(Path(os.environ["OMLX_CONFIG_PATH"]).read_text())
model_id = config["id"]
omlx = config["omlx"]
settings = {
    "version": 1,
    "models": {
        model_id: {
            "max_context_window": omlx["max_context_window"],
            "max_tokens": omlx["max_tokens"],
            "temperature": omlx["temperature"],
            "top_p": omlx["top_p"],
            "chat_template_kwargs": omlx["chat_template_args"],
            "forced_ct_kwargs": list(omlx["chat_template_args"]),
            "turboquant_kv_enabled": True,
            "turboquant_kv_bits": omlx["turboquant_kv_bits"],
            "turboquant_skip_last": True,
        }
    },
}
path = Path(os.environ["OMLX_BASE_PATH"]) / "model_settings.json"
path.write_text(json.dumps(settings, indent=2) + "\n")
PY

CMD=(
    "${OMLX_BIN}" serve
    --model-dir "${MODEL_ROOT}"
    --base-path "${BASE_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --max-concurrent-requests "${OMLX_MAX_CONCURRENT_REQUESTS}"
)

if [[ -n "${OMLX_MEMORY_GUARD:-}" ]]; then
    CMD+=(--memory-guard "${OMLX_MEMORY_GUARD}")
elif [[ -n "${OMLX_MEMORY_GUARD_GB:-}" ]]; then
    CMD+=(--memory-guard-gb "${OMLX_MEMORY_GUARD_GB}")
fi
if [[ "${OMLX_NO_CACHE:-}" == "true" ]]; then
    CMD+=(--no-cache)
fi

echo "Starting ${MODEL_NAME} via patched oMLX on port ${PORT}..."
exec "${CMD[@]}" "$@"
