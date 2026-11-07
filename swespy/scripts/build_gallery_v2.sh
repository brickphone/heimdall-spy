#!/usr/bin/env bash
# Overnight, resumable gallery v2 build (Part 4). Safe to rerun after any interruption:
# every stage skips work that is already checkpointed. Run from swespy/.
#   caffeinate -i scripts/build_gallery_v2.sh 2>&1 | tee -a data/gallery_v2/build.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv-benchmark/bin/python
# plan.json is already frozen (run `python -m eval.densify plan` only to rebuild it).
for stage in download crops filter encode index coverage; do
  echo "=== $stage $(date -u +%FT%TZ)"
  $PY -u -m eval.densify "$stage"
done
echo "=== done $(date -u +%FT%TZ)"
