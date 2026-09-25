#!/usr/bin/env bash
# Start the local wizard: FastAPI on :8765 and Vite on :5173.
set -euo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UMBRELLA="$(cd "$UI_ROOT/.." && pwd)"
VENV="${ARKGURU_VENV:-$UMBRELLA/.venv}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "No virtualenv at $VENV. From the umbrella repo run: bash scripts/setup_dev_env.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install -q -r "$UI_ROOT/requirements.txt"

export PYTHONPATH="${UI_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$UI_ROOT"

echo "API  http://127.0.0.1:8765/api/health"
echo "UI   http://127.0.0.1:5173/"

python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

cd "$UI_ROOT/frontend"
if [ ! -d node_modules ]; then
  npm install
fi
npm run dev
