# SweSpy

SweSpy is a production-grade geolocation inference API for Swedish imagery.

## Quickstart

1. Create `.env` with at least:

```bash
MAPILLARY_ACCESS_TOKEN=your_token
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

2. Install:

```bash
pip install -e .
```

3. Run API:

```bash
uvicorn swespy.api.main:app --reload
```

## Scripts

- `python scripts/download_data.py`
- `python scripts/train.py`
- `python scripts/index_vectors.py`
