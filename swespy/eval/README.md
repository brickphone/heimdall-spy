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

## Pretrained MegaLoc comparison

`megaloc_compare.py` compares the fixed street queries against the original CLIP
report using the same reference image IDs, GPS payloads, cosine retrieval, top five,
100m hit radius and similarity-weighted coordinate aggregation. It creates a new
timestamped Qdrant collection with 8,448-dimensional vectors. The API continues to
use its original CLIP collection. No reference/query acquisition or training occurs.

The default manifest/report arguments select the 2026-11-06 street check. Both
must already exist locally. The tool rejects changes to the original reference
index, query bytes/labels/order, production inference source and CLIP protocol.
Source, pretrained weights and their licenses are pinned to immutable upstream
revisions. The reviewed source checksum is checked before executing the downloaded
module; safetensors are loaded strictly against its architecture.

For native Apple GPU execution, use the project's locked Python 3.12 environment
outside Docker, with Qdrant running and `.env` pointing to `localhost`:

```bash
# Requires uv 0.9.5; this environment is already prepared on the development Mac.
UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" uv sync --frozen --python 3.12.8 --extra dev
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -u -m eval.megaloc_compare \
  --device mps --batch-size 4
```

The pinned torch 2.5.1 needs CPU fallback for bicubic positional interpolation
and contiguous Conv2d inputs on MPS. The experiment adapter preserves input values
and leaves upstream source/weights unchanged. Each run first checks CPU/device
descriptor agreement on two images (minimum cosine 0.999, maximum absolute
difference 0.001), then embeds the fixed images. `--probe-only` stops after that
check without creating a collection. `--device cpu` and `--device cuda` are also
supported; only MPS was used for the complete measured experiment.

Descriptors are cached per image SHA256, model/source identity, preprocessing,
device, software versions and adapter settings. Interrupted embedding resumes
without re-encoding verified cached images. Invalid descriptors and changed
input bytes fail before indexing. Collections are never overwritten. The newly
created index must pass a reference replay before query metrics are reported,
and both collections are fingerprinted to detect changes during evaluation.

Ignored `data/megaloc/experiments/<run>/` contains reference-byte provenance,
model identity, query predictions, `results.json`, an interactive `map.html` and
an offline-readable `comparison.html` linking both models' maps. Raw query/reference
photos and the original CLIP artifacts are preserved. Adding `HF_HUB_OFFLINE=1`
requires the already cached pinned assets. The measured results and limitations
are in [`docs/MEGALOC_COMPARISON.md`](../../docs/MEGALOC_COMPARISON.md).

## Published benchmark reproduction and dense Linkoping test

These are pretrained retrieval evaluations. They use the authors' coordinate
labels, exact FAISS search and Recall@1/5/10/20 with a 25m XY positive radius;
there is no averaged-coordinate prediction in the primary metric.

The prepared development Mac has a separate `.venv-benchmark/` environment that
reads the locked `.venv/` packages and adds the exact optional packages listed in
`requirements-benchmark.txt`. The application lock and runtime are unchanged.
Create the optional environment on a fresh Mac/Linux machine with uv 0.9.5:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv venv --python .venv/bin/python .venv-benchmark
.venv/bin/python - <<'PY'
from pathlib import Path
site = Path('.venv-benchmark/lib/python3.12/site-packages')
(site / 'locked-project.pth').write_text(
    str(Path('.venv/lib/python3.12/site-packages').resolve()) + '\n'
)
PY
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install --python .venv-benchmark/bin/python \
  -r eval/requirements-benchmark.txt
```

The complete Baidu split is the author-provided version documented in
[MegaLoc issue 13](https://github.com/gmberton/MegaLoc/issues/13): 5,207 database
images and 3,208 queries. Other Baidu versions use different splits/labels and
cannot reproduce the same reported numbers. Download this exact version with:

```bash
mkdir -p data/benchmarks/baidu
rsync -r --partial \
  rsync://vandaldata.polito.it/sf_xl/VPR-datasets-downloader/baidu/ \
  data/benchmarks/baidu/
```

The official evaluator source is cached under
`data/benchmarks/upstream/9c8dfd496b49a49549323b94a6af204cbee91604/`.
Retrieve `main.py`, `parser.py`, `test_dataset.py`, `visualizations.py`, `LICENSE`,
`README.md` and `requirements.txt` from that immutable revision of
`gmberton/VPR-methods-evaluation` when preparing a fresh machine. Upstream originals
are retained unchanged. Each run saves a separate execution copy with only the
query batch size adapted. A model factory supplies the pinned MegaLoc model/MPS
compatibility adapter and resumable descriptor cache, avoiding unrelated model
submodules. Batch-size invariance and CPU/device parity are checked before running.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.paper_benchmark --batch-size 32
```

The model assets must already be cached when using `HF_HUB_OFFLINE=1`.
The published targets are R@1 87.7% and R@10 98.0%; the protocol declares a
0.15 percentage-point reproduction tolerance before evaluation. Independent
recall recomputation must agree with the official displayed output. This is a
reproduction of released pretrained inference, not the unreleased training run.

All descriptors are cached per model/software/adapter identity and normalized
input tensor bytes. Interrupting the job preserves completed descriptors; rerunning
resumes them and creates a new timestamped result folder. Downloaded imagery,
optional environment and benchmark outputs are ignored by Git. Original CLIP and
MegaLoc collections/reports remain unchanged.

If encoding completed but the process stalled in the Mac OpenMP/FAISS search,
score its saved arrays without re-encoding:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-benchmark/bin/python -m eval.score_saved_benchmark \
  --run data/benchmarks/runs/<run> --dataset data/benchmarks/baidu
```

The normal driver also limits FAISS to one thread. Completed results and artifact
links are recorded in [the reproduction report](../../docs/PAPER_REPRODUCTION.md).

The dense central-Linkoping acquisition is a distinct development experiment:

```bash
.venv-benchmark/bin/python -u -m eval.dense_linkoping
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.paper_benchmark \
  --dataset data/linkoping_dense/central_v1 --name linkoping_dense --batch-size 32
```

It examines at most 6,000 metadata records within approximately 0.6 km², excludes
panoramas and protected old query sequences, balances references across 10m position
and 45° heading bins, and reserves new capture sequences for queries. It downloads
at most 1,200 selected references and up to 90 query candidates for a target of 30,
using decode checks without the historical 100 KiB reference-size filter. Fewer
eligible images/queries are reported without relaxing sequence, duplicate or spacing
rules. Coordinate-bearing hard links format the photos for the official evaluator.

Coverage is always reported separately: uncovered queries count as misses in the
all-query metric; conditional recall uses only queries with a positive at 25m.
Fixed random reference subsets isolate density effects using the same descriptors
and queries. This same-provider experiment has a small sample and unverified GPS/
headings; it does not replace independent user-owned evaluation photos.

After a completed run, render its report with:

```bash
.venv-benchmark/bin/python -m eval.benchmark_report --run data/benchmarks/runs/<run>
# For dense Linkoping, also provide its fixed dataset and attribution manifest:
.venv-benchmark/bin/python -m eval.benchmark_report --run data/benchmarks/runs/<dense-run> \
  --dataset data/linkoping_dense/central_v1 \
  --manifest data/linkoping_dense/central_v1/manifest.json
```

For a later cloud run, transfer the repository and required ignored datasets/model
assets/provenance, then recreate Python environments for the cloud OS. Do not copy
Mac virtual environments or `.env` credentials into a public artifact. The driver
also accepts `--device cuda`; descriptor cache identity records the device/software,
so a device change may require re-encoding. Completed descriptor arrays and source
datasets remain portable. No cloud resources or paid compute are provisioned here.

## City-wide Linkoping reference acquisition

From `swespy/`, with the private Mapillary token in `.env`:

```bash
.venv-benchmark/bin/python -u -m eval.expand_linkoping
```

This creates a separate frozen gallery at `data/linkoping_dense/city_v1/` within
`(15.50, 58.34, 15.75, 58.48)`. Eighty metadata tiles are checkpointed independently;
four metadata requests and six photo downloads can run concurrently. A first-pass
budget of 600 metadata records per tile and a total cap of 10,000 reference photos
bound acquisition. The provider may return fewer records without a pagination
cursor; neither the budget nor the cap implies exhaustive coverage.

All 369 central references are retained byte-for-byte using hard links. Additional
references use the newest photo per 20m position / 45-degree heading bin, balanced
round-robin across 1km cells. The old and central test capture sequences, including
all central reserved sequences, are excluded. Additional images matching protected
test hashes or conservative perceptual-hash thresholds are rejected, as are exact
duplicates within the gallery. This screening does not prove independence.

Downloads are validated and resumable. Transient failures prevent publishing a
final manifest until a successful rerun; a 4 GiB free-disk guard stops acquisition.
The manifest retains GPS, heading, capture time, creator, license, source link and
image hash. `coverage.html` displays actual downloaded camera positions with photo
previews; `coverage.geojson` is available for GIS inspection. The HTML map uses
external Leaflet/OpenStreetMap resources and needs an internet connection.

This command acquires reference images only. It does not encode them, train a model,
modify Qdrant, change the application model, or change frozen evaluation datasets.

## Encode, evaluate and search the expanded gallery

The city evaluation uses the unchanged 18 central and 30 earlier held-out photos.
Preparation hard-links source images into a separate coordinate-labelled dataset;
source manifests, image bytes and reference/query sequence separation are checked.

From `swespy/`:

```bash
.venv-benchmark/bin/python -u -m eval.city_benchmark
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.paper_benchmark \
  --dataset data/linkoping_dense/city_eval_v1 --name linkoping_city --batch-size 32
```

The official driver prints its timestamped run folder and resumes cached
descriptors when rerun. Use that completed folder in the next command:

```bash
MPLCONFIGDIR="$PWD/data/benchmarks/matplotlib" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.city_benchmark \
  --run data/benchmarks/runs/<city-run>
```

Finalization verifies the dataset fingerprint, saved arrays and independent
predictions; exports `gallery.faiss` plus `gallery-index.json` in exact index-row
order; and writes `city-results.json` / `city-summary.html`. It reports all48,
central18 and legacy30 query cohorts at25m/100m/500m, including coverage and
Recall@1/5/10/20. The original369-reference central baseline is recomputed using
identical queries/descriptors and checked against the recorded central result.
The photo/candidate map is `map.html`; `report.html` retains official25m metrics.

Search a local image against the exported gallery:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -m eval.search_gallery \
  --index data/benchmarks/runs/<city-run>/gallery-index.json \
  --image /path/to/photo.jpg --top-k 5
```

The query uses image pixels only; it does not read GPS metadata or infer location
from its filename. It returns ranked reference coordinates/attribution and cosine
similarity, without a weighted pin or calibrated confidence claim. Index checksum,
row count/dimension and encoder/software/device identity must agree. A changed
device/environment requires a compatible new export rather than mixing encoders.
This CLI does not update the API, train weights or contact a generative model.

## Recent ordinary-photo test (Commons only)

See [source constraints and current access status](../../docs/PUBLIC_PHOTO_TEST.md).
Only capture dates2023-11-06 through2026-11-06 qualify; upload dates do not count.
The user declined Flickr/paid services. From `swespy/`:

```bash
.venv-benchmark/bin/python -u -m eval.public_photos acquire
# Inspect review.html and record actual image-review choices in review.json first.
.venv-benchmark/bin/python -m eval.public_photos freeze
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.public_photos evaluate
```

Metadata/photos/results are ignored under `data/public_eval/recent_linkoping_v1/`.
Acquisition is currently externally blocked by Wikimedia HTTP403. No public query
set or new accuracy result has been published. Metadata checkpoints are resumable;
403 is not retried. Review happens before inference, and the reference index is
never modified by this workflow.
