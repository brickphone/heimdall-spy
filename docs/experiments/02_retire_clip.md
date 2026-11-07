# Part 2 — Retire the CLIP backend; one MegaLoc inference path

Date: 2026-11-07. No model weights, gallery images or evaluation photos changed.

## What changed

- **One inference path.** `swespy/model/locator.py`:
  - `Gallery` loads the exported flat FAISS index, verifies its checksum and keeps
    only references inside `LINKOPING_AREA` (899 of 3,388 excluded).
  - `GeoLocator.locate_vector` returns the top-ranked reference as the pin, the top
    5 candidates with similarities, and the full top-50 ranking for diagnostics.
  - The API (`POST /predict` → `locate_bytes`) and `eval/dev_eval.py`
    (`locate_vector` on cached descriptors of the same pixels) both call it.
- **MegaLoc moved into the app.** `swespy/model/megaloc.py` (pinned source/weights,
  checksum verification, descriptor cache, new in-memory `embed_bytes`).
  `same_descriptor_space()` compares only model/preprocessing identity, so a CPU
  server can query a gallery encoded on Apple MPS.
- **API.** Settings `MEGALOC_ROOT`, `MEGALOC_DEVICE` (default `auto`), `GALLERY_INDEX`,
  `RETRIEVAL_DEPTH`. Response adds `prediction_method`, `confidence_status`, and
  per-match `rank`, `image_id`, `creator`, `source_url`, `license_url` (the S2 cell
  field is gone). Confidence is still the uncalibrated top-1 cosine (Part 5 replaces
  it). Uploads stay in memory; logs no longer contain an image hash or coordinates.
  Unknown legacy keys in `.env` (`QDRANT_*`, `CLIP_MODEL_NAME`, `EMBEDDING_DIM`) are
  ignored rather than failing startup.
- **Removed (kept in git history only):** `swespy/model/embedder.py` (CLIP),
  `swespy/model/trainer.py` (CLIP fine-tuning trainer), `swespy/vector_store/` (Qdrant),
  `scripts/index_vectors.py`, `scripts/train.py`, `eval/run_eval.py`,
  `eval/mapillary_check.py`, `eval/megaloc_compare.py`, the `public_photos evaluate`
  action, the Qdrant Compose service and their tests. Shared helpers (`write_new`,
  `record`, `signature`, `duplicate`, `thumbnail`, `render_map`) moved unchanged to
  `eval/common.py`.
- **Kept:** the geographic-classifier experiment (`swespy/model/geo_head.py`,
  `eval/train_geo_head.py`, its report and checkpoint); `eval/search_gallery.py
  --classifier` still works and now runs through `GeoLocator`.
- **Dependencies:** removed `transformers`, `datasets`, `qdrant-client` (and 30+
  transitive packages); added `faiss-cpu==1.9.0.post1`, `huggingface-hub==0.36.2`,
  `safetensors==0.8.0` explicitly. `uv.lock` regenerated with uv 0.9.5.

## Verification

| Check | Result |
| --- | --- |
| `dev_eval` through `GeoLocator` vs Part 1 `baseline_v2.json` | Metrics, baselines, robustness and all 66 per-photo rows identical |
| API path equals eval path (`tests/integration/test_api_matches_eval.py`, 66 dev photos uploaded through `POST /predict`) | **Passed on MPS and on CPU**: identical top-5 reference IDs for 66/66, top-1 similarity within 1e-3 |
| Unit/integration suite (`.venv/bin/python -m pytest`) | 81 passed, 4 skipped (opt-in real-data tests) |
| Ruff / mypy | No new findings; remaining ones pre-date this change (some now sit in moved code) |

Not verified: a `docker compose up --build` image build (not run in this session).

## Run

```bash
cd swespy
# Same-path API vs eval check (needs local weights, index, dev photos):
SWESPY_RUN_REAL=1 PYTORCH_ENABLE_MPS_FALLBACK=1 MEGALOC_DEVICE=mps \
  .venv/bin/python -m pytest tests/integration/test_api_matches_eval.py
```
