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
