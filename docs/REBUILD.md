# Rebuilding the lost reference collection

The user reported that the old Mac project/data directory was deleted and could
not be recovered, and requested rebuilding the references. On 2026-10-16 they
confirmed additional disk space (118.5 GB verified) and configured a private
Mapillary token. This authorizes the limited new reference acquisition below,
revising the original no-new-data restriction for this prerequisite only.

The historical index cannot serve as the baseline. Ingestion repairs must
precede the new baseline; they are not claimed to be behavior-preserving model
refactoring. CLIP model, 768-dimensional cosine vectors, S2 level 14 and online
aggregation/scoring remain unchanged. There is no training or fine-tuning.

The default acquisition rectangle is a **central Linköping pilot**:
west 15.58, south 58.39, east 15.66, north 58.43. It is not the full city or
municipal boundary. The image budget is at most 500 metadata candidates shared
across the query grid, with the existing 50-per-cell sampling and 100 KiB file
size filter. Actual usable coverage/count will be smaller and must be measured.
Legacy Sweden-wide bbox settings are rejected before network calls.

Prerequisite repairs cover resumed-download bookkeeping, checking cached/downloaded
images, atomic file writes, non-retryable credential errors, propagating failed
metadata searches/downloads, geometry checks and preserving previous metadata on
empty/failed runs. Raw cached files are retained rather than deleted by cleaning.
Indexing refuses a nonempty collection and resolves image paths from stable IDs
so metadata generated on Windows can be indexed inside the Linux container.
These guards are not a general-purpose incremental-index/recovery implementation.

Offline verification before acquisition: **31 tests passed, 1 live snapshot
skipped**. New ingestion regressions use synthetic photos and mocked HTTP;
they are not evaluation data. Pytest discovery is restricted to tests because
Linux-created model-cache symlinks cannot be traversed by Windows pytest.

Reference files and parquet stay under ignored `swespy/data/`; Qdrant data stays
under ignored `swespy/qdrant_storage/`. The token stays in ignored `.env`, is never
printed, and is sent only to Mapillary's Graph API. Reference image IDs link back
to their Mapillary source; a public/commercial deployment must retain the source
attribution and satisfy the applicable imagery/API terms. No data is redistributed
or committed here. API access documentation:
https://www.mapillary.com/developer/api-documentation/

The new real-photo baseline remains pending independent user-owned photos and
camera-location labels, including all four viewpoint types. Mapillary reference
images must never be used as that evaluation set. Do not present a rebuilt
retrieval index or a synthetic request as measured geolocation accuracy.

## First pilot result, 2026-10-16

- Mapillary authentication succeeded. 231 reference photos downloaded without
  download failures; 189 passed the unchanged 100 KiB threshold and decode checks.
  Smaller originals remain in raw storage. Total raw data size: 47.13 MiB.
- Accepted references span 34 S2 cells, latitude 58.391011–58.426994 and longitude
  15.580333–15.654169. This is sparse pilot coverage, not dense city coverage.
- Embedded the 189 references inside the running Linux container using the
  already-cached `openai/clip-vit-large-patch14`, offline model loading and batch
  size 8. Vector shape/finiteness/norm checks passed before upsert. The updated
  indexing script was streamed into the existing container; its original image
  has not been rebuilt with these source changes yet.
- Qdrant confirmed 189 points in `swespy_embeddings`; schema remains 768-d cosine.
- One reference replay through the actual API returned HTTP 200, five candidates,
  top similarity 0.9999999, and a candidate at the known reference coordinates.
  Processing time was approximately 473 ms. This is a wiring/same-image retrieval
  smoke test, not a viewpoint benchmark or proof of accuracy on unseen images.
- Final regression suite: **34 passed, 1 live snapshot skipped**. Changed
  ingestion/index code and tests pass Ruff; the three ingestion/index source
  files pass mypy with third-party stubs/imported legacy modules excluded.
- Token, raw photos and metadata are ignored by Git. No training, new model,
  confidence calibration or independent eval baseline was produced. Both services
  remain running with the new collection accessible immediately, without restarting
  the API.
