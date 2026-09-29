#!/bin/bash
# Start a System One (type: systemone) Shingi decision server.
# Setup is separate and one-time: scripts/setup-shingi.sh <model-dir>. This
# script never writes under the repository, so it works under ProtectHome=read-only.
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG="$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"
eval "$CONFIG"

if [[ "${SHINGI_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} has no shingi configuration." >&2
    exit 1
fi

ENGINE_DIR="${ROOT}/.engines/shingi"
PYTHON="${ENGINE_DIR}/venv/bin/python"
READOUT="${ENGINE_DIR}/bin/readout"

# Both stamps must match the pinned revisions; otherwise setup has not run or is stale.
readout_stamp="$(cat "${READOUT}.stamp" 2>/dev/null || true)"
if [[ ! -x "$PYTHON" || ! -x "$READOUT" \
    || "$(cat "${ENGINE_DIR}/package.stamp" 2>/dev/null || true)" != "$SHINGI_PACKAGE_REVISION" \
    || "$readout_stamp" != "${SHINGI_PACKAGE_REVISION} ${SHINGI_RUNTIME_REVISION} "* ]]; then
    echo "Error: ${MODEL_NAME} is not set up for package ${SHINGI_PACKAGE_REVISION} and runtime ${SHINGI_RUNTIME_REVISION}." >&2
    echo "Run scripts/setup-shingi.sh \"${MODEL_DIR}\" once before starting the server or its service." >&2
    exit 1
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-$MODEL_HOST}"

echo "Starting ${MODEL_NAME} System One server on port ${PORT}..."
exec "$PYTHON" "${SCRIPTS_DIR}/shingi-server.py" \
    --preload-mib "$SHINGI_PRELOAD_MIB" \
    --headroom-mib "$SHINGI_HEADROOM_MIB" \
    --weights-repo "$SHINGI_WEIGHTS_REPO" \
    --weights-revision "$SHINGI_WEIGHTS_REVISION" \
    --model-file "$SHINGI_MODEL_FILE" \
    --calibration-file "$SHINGI_CALIBRATION_FILE" \
    --executable "$READOUT" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
