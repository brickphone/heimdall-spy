#!/usr/bin/env bash
# Pod setup (see docs/CLOUD_BUILD.md). Network: PyPI packages + Hugging Face weights only.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/cloud_env.sh
python3.12 -m pip install -q "uv==0.9.5"
python3.12 -m uv sync --frozen --python /usr/bin/python3.12
.venv/bin/python - <<'PY'
import os, torch
from pathlib import Path
from swespy.model.megaloc import MegaLocEmbedder
token = os.environ.get("MAPILLARY_ACCESS_TOKEN", "")
print("token_present:", bool(token) and "RUNPOD_SECRET" not in token)  # Value never printed.
print("gpu:", torch.cuda.is_available() and torch.cuda.get_device_name(0), "torch", torch.__version__)
e = MegaLocEmbedder(Path("data/megaloc"), "cuda")
print("megaloc_loaded: device", e.identity["device"], "pos_cache_modules", e.pos_cache_modules)
PY
echo SETUP_OK
