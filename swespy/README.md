# SweSpy

SweSpy is an experimental image-geolocation API. Its current pretrained CLIP
retrieval pipeline is being audited and measured for Linköping. It does not yet
have a measured accuracy baseline or calibrated confidence.

## Quickstart

From this `swespy/` directory, with Docker Engine and Docker Compose 2.24+
installed, one command builds the locked Python environment and starts Qdrant
and the API:

```bash
docker compose up --build
```

The container uses Python 3.12.8, uv 0.9.5, the committed `uv.lock`, and Qdrant
1.12.5. The API waits for Qdrant readiness and connects to its service hostname.
Ports are bound to localhost. A `.env` is optional; see `.env.example` for local
Python configuration. No Mapillary token or ingestion is required to start.
The first API startup may download the configured pretrained CLIP weights into
`data/huggingface/`; CPU inference is the reproducible container default.

**Starting services does not populate reference data.** With a fresh collection,
current inference has the empty-result behavior documented in the audit. Real
geolocation requires the existing reference index. Do not create a synthetic
index and call it the baseline. No dataset download/training runs at startup.

For local development, install Python 3.12 and uv 0.9.5, then:

```bash
uv sync --frozen --extra dev
docker compose up -d qdrant
```

Copy `.env.example` to `.env` and run:

```bash
uv run --frozen uvicorn swespy.api.main:app --reload
```

## Scripts

These existing scripts require reference data or external credentials. Ingestion
defaults still cover Sweden and have known faults; do not run ingestion for this
Linköping-only baseline. Model, S2 level and Qdrant schema are unchanged.

- `python scripts/download_data.py`
- `python scripts/train.py`
- `python scripts/index_vectors.py`

## Evaluation and checks

See [eval/README.md](eval/README.md) for private photos/labels, real pipeline
evaluation, one-time baseline capture and immutable snapshot comparisons.
No real baseline is available until private queries and a real index are supplied.

`uv run --frozen python scripts/check.py` runs ruff, mypy and pytest, returning
failure for any failed gate. Legacy source lint/type failures remain visible;
fixing them belongs to Part 3, after the measured baseline. The exact historical
compatibility pins establish a baseline environment, not a claim of security
hardening or compatibility with every newer dependency release.
