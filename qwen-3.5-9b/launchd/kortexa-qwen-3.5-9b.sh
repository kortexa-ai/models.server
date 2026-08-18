#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && /bin/pwd)"
MODELS_SERVER_ROOT="$(cd "$SCRIPT_DIR/../.." >/dev/null && /bin/pwd)"
exec "$MODELS_SERVER_ROOT/run.sh" "$@"
