#!/usr/bin/env bash
# Street View internal-backend pilot: pod setup + run (see docs/experiments/11_gsv_internal_pilot.md).
# Usage on the pod: scripts/gsv_internal_pod.sh <fetch_deadline_unix>
# No API keys or secrets are used. streetlevel lives only in .venv-gsv-internal.
set -euo pipefail
cd "$(dirname "$0")/.."
DEADLINE="$1"
export UV_PYTHON_DOWNLOADS=never HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_TELEMETRY=1
# Ubuntu 24.04 forbids pip into the system Python (PEP 668): pin uv in a bootstrap venv.
python3.12 -m venv /root/uvboot
/root/uvboot/bin/pip install -q "uv==0.9.5"
UV=/root/uvboot/bin/uv
$UV export --frozen --no-dev --no-emit-project --no-hashes -q > /tmp/req.txt
$UV venv -q --python /usr/bin/python3.12 .venv-gsv-internal
$UV pip install -q --python .venv-gsv-internal/bin/python -r /tmp/req.txt \
  "streetlevel==0.12.13" pytest
$UV pip install -q --python .venv-gsv-internal/bin/python --no-deps -e .
.venv-gsv-internal/bin/python -c "import streetlevel, importlib.metadata as m; print('streetlevel', m.version('streetlevel'))"
.venv-gsv-internal/bin/python -m pytest -q tests/unit/test_gsv_internal.py tests/unit/test_gsv.py
# Load MegaLoc once (downloads the pinned weights from Hugging Face; no Google traffic).
.venv-gsv-internal/bin/python -c "
from pathlib import Path
from swespy.model.megaloc import MegaLocEmbedder
MegaLocEmbedder(Path('data/megaloc'), 'cpu'); print('MEGALOC_OK')"
echo SETUP_OK
touch /root/work/START  # Disk-scan marker: any image file newer than this is a violation.
.venv-gsv-internal/bin/python -m eval.gsv_internal_fetch \
  --points data/gsv_internal/pilot_points.csv --out data/gsv_internal --deadline "$DEADLINE" \
  2>&1 | tee data/gsv_internal/run.log
touch data/gsv_internal/DONE
