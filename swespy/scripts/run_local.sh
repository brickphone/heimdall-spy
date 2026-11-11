#!/usr/bin/env bash
# Start the API and the demo web app on this Mac only. Open http://127.0.0.1:3000
# - API (FastAPI/uvicorn): 127.0.0.1:8000, re-ranking on (K=5, 1,024 keypoints),
#   MegaLoc + ALIKED on the Apple GPU. Loads the model and gallery once (~60 s).
# - Web (Next.js, long-running Node server): 127.0.0.1:3000, forwards /api/* to the API.
# Stop both with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export MEGALOC_DEVICE="${MEGALOC_DEVICE:-mps}" RERANK_DEVICE="${RERANK_DEVICE:-mps}"
export RERANK_ENABLED=true TESTING=false
export TORCH_HOME="${TORCH_HOME:-$PWD/data/torch}"  # Pinned ALIKED, LightGlue and person-mask weights.

if [ ! -d web/node_modules ]; then
  (cd web && npm ci)
fi
(cd web && SWESPY_API_URL="http://127.0.0.1:${API_PORT}" npm run build >/dev/null)

echo "Loading the model and gallery (about a minute)..."
.venv/bin/uvicorn swespy.api.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null; wait "$API_PID" 2>/dev/null' EXIT INT TERM

# Start the web server once the API is ready, so the proxy has something to talk to.
until curl -sf "http://127.0.0.1:${API_PORT}/ready" >/dev/null; do
  kill -0 "$API_PID" 2>/dev/null || { echo "The API exited during startup." >&2; exit 1; }
  sleep 2
done

echo "Demo: http://127.0.0.1:${WEB_PORT}"
cd web && HOSTNAME=127.0.0.1 PORT="$WEB_PORT" npm start
