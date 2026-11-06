# Agent handoff — Linköping geolocation

Updated: 2026-11-06. Read this file before resuming project work.

## User objective and preferences

Build a GeoSpy-inspired demo that locates ordinary outdoor photos in Linköping,
then expand across Sweden. Street scenes and landmarks are an acceptable first
scope; do not redefine the goal as locating only Street View screenshots. The
user prefers automatically sourced evaluation photos over walking outside to
take them. Start on the M1 Mac; cloud migration later is acceptable. No paid
compute or API spend has been authorized. Preserve existing work and datasets.

## Architecture that actually exists

- `swespy/swespy/`: FastAPI API, pretrained CLIP ViT-L/14, 768-dimensional
  embeddings, Qdrant cosine search, top five matches, weighted average GPS pin.
  Confidence is similarity-derived and uncalibrated.
- `swespy/eval/megaloc_compare.py`: pinned pretrained MegaLoc loader, 8,448-D
  normalized descriptors and resumable image-hash cache. Separate experiments;
  the API still uses CLIP.
- `swespy/eval/paper_benchmark.py`: pinned official VPR evaluator, cached
  descriptors, MPS compatibility checks, FAISS exact search, 25m Recall@K.
- PIGEON/PIGEOTTO were discussed earlier but are not implemented in current
  source. No Swedish fine-tuning, geographic prediction model, LLM or RAG answer
  generation is deployed. Do not confuse reference indexing with training.

## Recent completed work

1. Reproduced the released MegaLoc model on the exact author-identified Baidu
   split: 5,207 references / 3,208 queries; R@1 87.7494%, R@10 97.9738%, matching
   published rounded values 87.7% / 98.0%. This reproduces pretrained inference,
   not the original training. See `PAPER_REPRODUCTION.md`.
2. Acquired a dense central gallery of 369 references and 18 held-out queries
   across nine capture sequences. R@1 at25m was 11/18; only14/18 have positives.
   Reference thinning showed density matters on the same frozen queries.
3. Expanded references city-wide with `eval/expand_linkoping.py`: **3,388 photos**,
   retaining all369 central references and adding3,019 unique photos. 305 capture
   sequences,31 creators,143 occupied1km cells. Rectangle
   `(15.50,58.34,15.75,58.48)`; bounded first pass, not exhaustive street coverage.
   See `GALLERY_EXPANSION.md` for protocol and artifacts.
4. Protected39 old/reserved query sequences, screened additional images against
   held-out query hashes/perceptual hashes, removed one exact gallery duplicate.
   Verified old manifests and original CLIP index unchanged; API health HTTP200.
5. Added root `README.md` explaining technologies, architecture, setup and limits;
   updated application/evaluation READMEs. City downloader checks:64 tests passed,
   one private snapshot test skipped; changed modules passed Ruff/mypy.

## Current authorized task — in progress

The user asked to do the next step: encode the expanded gallery with MegaLoc and
evaluate against frozen queries. Also explain RAG/modern ML in a Markdown file.
This handoff was written **before** starting that work; update it with results
and exact artifact paths before finishing. Independent public-photo acquisition
and a production API model switch have not yet been performed.

## Local artifacts (ignored by Git; back these up separately)

All following paths are relative to `swespy/`:

- `data/linkoping_dense/city_v1/manifest.json`: frozen3388-reference gallery.
  `coverage.html` / `coverage.geojson` show actual camera positions; images in
  `originals/` and coordinate-bearing hard links in `images/test/database/`.
- `data/linkoping_dense/central_v1/manifest.json`: original369 references /18
  queries; original central evaluation dataset. Do not modify it.
- `data/mapillary_eval/20261106T095121.923253Z/manifest.json`: older30-query frozen
  set used with317 sparse references. Query capture sequences must stay excluded.
- `data/benchmarks/runs/baidu-20261106T105553.510877Z/`: reproduced paper results.
- `data/benchmarks/runs/linkoping_dense-20261106T115241.727823Z/`: original central
  results, report,map,predictions,descriptor arrays and density experiment.
- `data/megaloc/`: pinned model source, weights and image-hash descriptor cache.
- `data/benchmarks/descriptor_cache/`: official-driver normalized-input cache.
- Original Qdrant CLIP collection:`swespy_embeddings`; earlier MegaLoc collection:
  `swespy_megaloc_20261106t102349_004608`. Never overwrite these for an experiment.

## Running and resuming

Work from `swespy/`. `.venv` is the locked main environment; `.venv-benchmark`
adds pinned optional dependencies from `eval/requirements-benchmark.txt` and
uses the main environment's torch. Native Python3.12.8,torch2.5.1. Use:

```bash
.venv-benchmark/bin/python -m pytest -q
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.paper_benchmark --batch-size 32
```

Use MPS for native Mac GPU inference. The existing loader supplies value-preserving
contiguous-convolution adapters and CPU interpolation fallback. Keep FAISS at one
thread: a prior final search deadlocked in mixed OpenMP runtimes after MPS.
`eval/score_saved_benchmark.py` can score completed arrays in a fresh CPU process
without re-encoding. Preserve caches and source/weight checksums.

Mapillary credentials are in ignored `swespy/.env`; never print or commit them.
Downloaded metadata includes signed image URLs; keep it ignored. Network/MPS and
Docker access may require sandbox escalation. Current gallery directory was281MiB;
Mac had about14GiB free after acquisition. Do not remove unrelated files for space.

## Interpretation guardrails

All-query recall includes uncovered queries as misses; report reference coverage
separately. GPS labels are provider-supplied. Same-provider Mapillary holdouts are
development evidence, not proof of performance on arbitrary user photos. Compare
gallery variants on identical queries/model/metric; central18 and older30 sets
are different. Show retrieved evidence; an LLM explanation alone is not validation.
