#!/usr/bin/env bash
# Both gallery v2 stages on a rented GPU; run inside tmux. Resumable: rerun after failure.
# The sealed test-set guard is deferred: the test manifest is never copied to the pod.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/cloud_env.sh
for stage in recent historic; do
  echo "=== $stage $(date -u +%FT%TZ)"
  .venv/bin/python -u -m eval.gallery_build --stage "$stage" --device cuda \
      --dtype float32 --batch-size 64 --defer-sealed-guard
done
echo "=== BUILD_DONE $(date -u +%FT%TZ)"
