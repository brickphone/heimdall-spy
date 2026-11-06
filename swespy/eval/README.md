# Private retrieval evaluation

Part 2 preparation is implemented; the real baseline and snapshot capture are
pending user-owned photos, camera-location labels and a populated reference index.
Synthetic unit fixtures are never used as baseline evidence.

1. Place your own JPEG/PNG queries in `photos/` (ignored by Git).
2. Copy `labels.example.csv` to `labels.csv` (also ignored) and populate
   `filename,lat,lng,viewpoint_type`. Viewpoints: `street`, `window`, `elevated`,
   `indoor_view`. Label the camera position, not the building in the picture.
   At least ten distinct images are required for snapshot capture; meaningful
   per-viewpoint evaluation generally needs substantially more.
   Baseline capture requires all four viewpoint types to be represented.
3. Validate: `uv run --frozen python -m eval.run_eval --validate-only`.
4. Connect `.env` to the existing reference Qdrant collection. Never insert
   Mapillary/reference images into evaluation. Exact copies in local `data/raw/`
   are rejected; this does not prove ownership or detect all near-duplicates.
5. With the existing model available, capture once:
   `uv run --frozen python -m eval.run_eval --baseline --capture-snapshot`.
6. After later refactor commits:
   `uv run --frozen python -m eval.run_eval --check-snapshot`.

The runner invokes actual API lifespan and POST /predict in-process, using the
real CLIPEmbedder and Qdrant collection. TESTING is forbidden. It checks for a
populated index before startup so a missing collection is not silently created.
It does not invoke training, ingestion or indexing. Model loading retains current
Hugging Face behavior and may download pretrained model weights if not cached;
use `HF_HUB_OFFLINE=1` to require the existing cache. No new dataset is fetched.
It respects the existing API rate limit and suppresses the route's hash/pin logs
only for evaluation. The ordinary API retains the audit's privacy gaps until Part 3.

Outputs include median error in metres, percentages within 100m/500m/2km and
geographic top-k hit percentage overall and by viewpoint. A hit means **any
returned reference coordinate within 100m of the camera label**, configurable
with `--hit-radius-m`. It is not exact GPS equality or automatic identification
of a building. Empty viewpoint groups have count 0 and null metrics, never
perfect zero error. Confidence remains explicitly uncalibrated in Part 2.

Timestamped JSON includes anonymous per-query measurements and environment
provenance. Baseline JSON contains aggregate metrics/provenance only. The index
fingerprint scans all reference vectors/payloads before and after evaluation;
this can be expensive and requires a read-only/frozen index throughout the run.
The loaded model revision, lock, source and labelled-photo-set hashes are saved.
Freeze the Hugging Face revision/cache for comparisons; a different revision
fails snapshot identity checks. Image bytes and filenames are not saved in results.

Private snapshot predictions are ignored by Git. They must remain available
locally or through an approved private fixture store. Capture refuses to replace
existing snapshots/baseline. Comparisons use 1 metre coordinate tolerance and
1e-5 absolute score tolerance, preserve candidate order/cells, and ignore timing.
For behavior-preserving refactors, predeclare a median-error tolerance of 1m and
zero percentage-point regression on every populated viewpoint group; investigate
any change rather than editing baseline/snapshot artifacts. Part 4 intentionally
changes scoring and requires a separately reviewed comparison.
Subsequent runs automatically compare to baseline.json when present, save deltas
and return failure on regression. `--compare-baseline PATH` selects another
frozen report explicitly. Changed datasets/index/model/preprocessing/top-k or
geographic hit radius are rejected as incomparable.

Run the pytest snapshot gate explicitly with `SWESPY_RUN_REAL_SNAPSHOT=1`
(PowerShell: `$env:SWESPY_RUN_REAL_SNAPSHOT='1'`) and
`uv run --frozen pytest tests/integration/test_real_snapshot.py`. Until fixtures
exist, ordinary pytest reports this gate as skipped, not passed.

Do not publicly commit private photos, labels or snapshot predictions. Review
aggregate baseline/provenance for sensitive collection names or other metadata
before committing it. No baseline.json has been fabricated for missing inputs.

## Automatic Mapillary street sanity check

This is a separate same-provider test, not the independent four-viewpoint baseline
above. It uses licensed third-party Mapillary photos, never `eval/photos/` or the
reference ingestion pipeline. No training, reference upsert, confidence calibration
or baseline/snapshot capture is performed.

With the populated reference collection, private token and API running, execute
from `swespy/`:

```bash
docker compose up --build -d
docker compose exec -T api python -u -m eval.mapillary_check
```

The default examines at most 2,000 metadata candidates within the existing pilot
rectangle and downloads at most 90 photos to select 30. It excludes all local raw
image IDs, every indexed reference capture sequence, exact copies and perceptually
similar images. Accepted queries are at least 100m apart, with at most three from
one sequence, selected using a fixed seed before inference. Missing reference
sequence/attribution data aborts acquisition; an insufficient eligible set is
reported without relaxing the rules. Perceptual thresholds are conservative
screening heuristics, not proof that all similar scenes have been detected.

Metadata, attributed query photos, frozen manifest and JSON/HTML reports stay under
ignored `data/mapillary_eval/`. The HTML contains small attributed photo previews,
camera labels, predicted pins and inspectable ranked candidate locations. Leaflet
and OpenStreetMap tiles require internet; the photo cards remain readable offline.
The index is fingerprinted before/after acquisition and evaluation. Query bytes
are checked before inference; reports retain the manifest, evaluator and inference
source hashes and cached model revision. The test calls the running local API;
keep the same API image/model cache throughout the run.

To select without inference, use `--prepare-only`. To repeat the same frozen test
without downloading more photos:

```bash
docker compose exec -T api python -u -m eval.mapillary_check \
  --manifest data/mapillary_eval/<run>/manifest.json
```

Reports are timestamped and never overwrite earlier measurements. Coverage metrics
show what fraction of camera labels have any indexed reference within 100m/500m/2km,
independent of retrieval. GPS labels may be inaccurate, queries share a provider
with references, and the sample covers street views only. These results must not
be presented as general photo-geolocation accuracy or replace user-owned photos.

Source attribution and capture details are retained for each query and indexed
reference. See Mapillary's [imagery license guidance](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data).
