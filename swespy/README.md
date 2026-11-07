# SweSpy

SweSpy is an experimental image-geolocation API for Linköping. It encodes an
uploaded photo with pretrained MegaLoc and searches a polygon-restricted gallery of
geotagged references (exact FAISS search). The API and the evaluation scripts share
one inference path, `swespy/model/locator.py`. The former CLIP/Qdrant backend was
retired on 2026-11-07 and survives only in git history. Accuracy on ordinary photos
is low and confidence is uncalibrated; see `eval/results/baseline_v2.json` and the
[repository overview](../README.md).

## Quickstart

From this `swespy/` directory, with Docker Engine and Docker Compose 2.24+:

```bash
docker compose up --build
```

The container uses Python 3.12.8, uv 0.9.5 and the committed `uv.lock`, and runs
MegaLoc on CPU. It reads the MegaLoc weights and the exported gallery index from the
mounted `data/` directory (`GALLERY_INDEX`, `MEGALOC_ROOT`); neither is in git, and
startup fails clearly if the index is missing. Uploads are processed in memory and
never stored. Ports are bound to localhost.

For local development, install Python 3.12 and uv 0.9.5, then:

```bash
uv sync --frozen --extra dev
PYTORCH_ENABLE_MPS_FALLBACK=1 MEGALOC_DEVICE=mps uv run --frozen uvicorn swespy.api.main:app
```

## Scripts

`python scripts/download_data.py` runs the original Mapillary ingestion. Reference
acquisition for the MegaLoc gallery lives in `eval/` (see `eval/README.md`).

## Evaluation and checks

See [eval/README.md](eval/README.md) for private photos/labels, real pipeline
evaluation, one-time baseline capture and immutable snapshot comparisons.
No real baseline is available until private queries and a real index are supplied.

`uv run --frozen python scripts/check.py` runs ruff, mypy and pytest, returning
failure for any failed gate. Legacy source lint/type failures remain visible;
fixing them belongs to Part 3, after the measured baseline. The exact historical
compatibility pins establish a baseline environment, not a claim of security
hardening or compatibility with every newer dependency release.
