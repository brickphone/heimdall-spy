#!/usr/bin/env bash
# Staged, pipelined, resumable gallery v2 build (Part 4). Safe to rerun after any
# interruption: finished chunks are skipped. Run from swespy/.
#   caffeinate -i scripts/build_gallery_v2.sh recent   2>&1 | tee -a data/gallery_v2/build.log
#   caffeinate -i scripts/build_gallery_v2.sh historic 2>&1 | tee -a data/gallery_v2/build.log
# plan.json is frozen; run `python -m eval.densify plan` only to rebuild it on purpose.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?usage: build_gallery_v2.sh recent|historic}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv-benchmark/bin/python
echo "=== $STAGE $(date -u +%FT%TZ)"
$PY -u -m eval.gallery_build --stage "$STAGE" --device mps
echo "=== done $STAGE $(date -u +%FT%TZ)"
