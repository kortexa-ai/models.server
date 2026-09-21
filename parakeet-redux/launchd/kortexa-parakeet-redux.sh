#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_SERVER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec "$MODELS_SERVER_ROOT/run.sh" "$MODELS_SERVER_ROOT/parakeet-redux" "$@"
