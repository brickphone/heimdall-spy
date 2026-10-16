# SweSpy audit

Date: 2026-10-16. Audited revision: `d8ff4ce1` (before this documentation commit).
Status: Part 1 complete; awaiting human review before Part 2. No application,
test, configuration, model, data or index changes made. Paths below are relative
to `swespy/` unless explicitly prefixed with `docs/`.

## Scope and evidence

Read all 26 Python source files, all tests, README, packaging/build/container
files, ignore rules and tracked egg-info metadata. Inventoried tracked bytecode
as generated artifacts, not authoritative source. No AGENTS.md was found in the
repository or its ancestor directories. The initial working tree was clean.
There is no eval directory, dataset, checkpoint, lockfile, CI workflow, or
persisted Qdrant collection in this checkout. Historical training, the deployed
model, actual geographic coverage and the reported window failure cannot be
verified from this checkout alone. The viewpoint gap is user-reported evidence,
not a measured result of this audit.

Read-only verification used the bundled Python 3.12.14 with `-B` (no bytecode
writes). All 26 source files parsed with `ast.parse`. An isolated execution of
the actual `Trainer.split_by_cell` method on ten synthetic cells confirmed that
validation labels are absent from the training map. A synthetic reproduction
of ingestion's groupby/apply on installed pandas 3.0.1 confirmed that the grouping
column is omitted. These are focused reproductions, not a passing project suite.

`python -B -m pytest -p no:cacheprovider` from the application directory failed
immediately: `No module named pytest`. The PATH Python is an unusable Windows
Store alias; no docker, ruff or pytest executable was found on PATH. No packages,
models or datasets were downloaded, services started, training run, or external
image APIs called. Official dependency documentation was consulted for specific
compatibility issues; GeoSpy was neither queried nor scraped.

## Actual architecture and data flow

| Component | Actual responsibility |
| --- | --- |
| `config.py` | Cached pydantic-settings configuration from environment and CWD `.env` |
| `data/mapillary.py` | Async HTTP metadata pagination, retries and reference-image downloads |
| `data/s2_grid.py` | Coordinates to S2 token, token to center, dataframe assignment |
| `data/pipeline.py` | Bounding-box subdivision/fallback, deduplication, per-cell sampling, downloads, size-only cleaning, parquet output |
| `model/trainer.py` | Separate CLIP cell classifier training, split, validation and checkpoint save |
| `model/embedder.py` | Hugging Face pretrained CLIP features, float32 L2 normalization, file/PIL/byte/batch input |
| `model/evaluate.py` | Haversine in km and an unwired classifier benchmark |
| `vector_store/qdrant_client.py` | Collection creation, upsert, nearest-neighbor query and coordinate/confidence aggregation |
| `api/main.py` | App construction, startup, logging, IP rate limiting, production-hosted test doubles |
| `api/routes/predict.py` | Upload MIME/size check, inference orchestration and response/log formatting |
| `api/routes/health.py` | Unconditional `{"status":"ok"}`; no readiness check |
| `api/middleware.py` | UUID, response header, request latency log |
| `api/schemas.py` | Match and prediction response models |
| `scripts/*.py` | CWD-dependent download/train/index entry points |
| `__init__.py` files | Package documentation and version export only |

Offline ingestion: configured bbox -> subdivided bboxes -> Mapillary metadata
-> deduplicate image IDs -> S2 level 14 -> sample up to 50 per cell -> reference
JPEG files under `data/raw/` -> remove files smaller than 100 KiB ->
`data/metadata/images.parquet` containing image ID, coordinates, cell and path.
Sampling precedes cleaning and failed downloads, so cell caps are not coverage
guarantees. Cleaning does not decode images or validate their coordinates.

Offline training: parquet -> disjoint S2-cell train/validation/test partitions
-> training-only label map -> pretrained CLIP plus linear classifier -> cross
entropy/AdamW/cosine schedule -> `data/checkpoints/best_model.pt`. The vision
encoder is frozen, but the visual projection remains trainable; the whole CLIP
model is in the optimizer. Validation labels cannot be resolved because their
cells were explicitly held out. The test partition is discarded. The returned
model is the last model, not a reloaded best checkpoint; zero validation accuracy
never writes a checkpoint. No seed controls torch initialization or shuffling.

Offline indexing: all parquet paths -> newly loaded pretrained CLIP -> all
embeddings accumulated in RAM -> Qdrant cosine collection with configured 768
dimensions -> payloads `{image_id, lat, lng, s2_cell}` and sequential integer
point IDs. IDs restart at 1 on every call; repeated or reordered indexing can
overwrite existing points and leave stale tail points. No checkpoint is loaded.

Online startup synchronously loads pretrained CLIP and processor, creates a
sync Qdrant client and creates the collection if absent. An empty collection is
therefore accepted. `TESTING=true` replaces components with constant Stockholm
predictions. Startup has no resource teardown or schema/model-revision check.

Online request: multipart parsing -> declared MIME check -> read entire upload
-> byte-size check -> PIL RGB decoding/CLIP in `asyncio.to_thread` -> synchronous
Qdrant search in `asyncio.to_thread` -> weighted coordinate mean -> response
with a pin, confidence and ordered image matches. Request UUID/latency and a
separate image-hash/pin log are emitted. GPS/EXIF coordinates are not used, but
EXIF orientation is also not applied.

**There is no first coarse localisation stage.** The cell classifier is neither
loaded nor called by inference, and there is no region selection or geographic
query filter. CLIP + Qdrant is the sole retrieval stage. S2 is reference metadata,
not an online region estimator. The current default bbox covers Sweden
`(10.5,55.0,24.2,69.1)`, not Linköping; the supplied index's actual extent is
unknown. Do not run ingestion with defaults for this Linköping-only task.

## Dependencies and reproducibility

All declarations are lower bounds, not pins. No transitive lock or model revision
exists. Egg-info repeats declarations; it does not prove installed versions.
The observed versions below belong to the bundled audit runtime, not a known
SweSpy deployment. A dash means absent there.

| Dependency | Declared | Observed | Use / concern |
| --- | --- | --- | --- |
| torch | >=2.2 | — | Embedding/training/evaluation; Docker starts with 2.2.0 but resolver can replace it |
| torchvision | >=0.17 | — | No direct source imports; candidate for dependency review |
| transformers | >=4.40 | — | CLIP loading; no upper bound or model revision |
| datasets | >=2.19 | — | No source imports; candidate unused dependency |
| qdrant-client | >=1.9 | — | Used; allowed 1.16.0 removes called `search` method |
| fastapi | >=0.111 | — | API; unbounded Starlette compatibility |
| python-multipart | >=0.0.9 | — | Required for file parsing, implicitly used |
| uvicorn[standard] | >=0.29 | — | README and container entry point |
| pydantic | >=2.7 | 2.13.5 | Schemas/config; permissive settings and payloads |
| pydantic-settings | >=2.2 | — | Settings; no validation constraints on operational ranges |
| httpx | >=0.27 | — | Mapillary and ASGI test |
| s2sphere | >=0.2.5 | — | Cell utilities |
| pillow | >=10.3 | 12.3.0 | Decoding; no EXIF transpose/explicit pixel budget |
| pandas | >=2.2 | 3.0.1 | Parquet/data; allowed 3.x changes grouping-column behavior |
| pyarrow | >=16 | — | Implicit pandas parquet engine |
| loguru | >=0.7 | — | Text logs; no JSON serialization configured |
| tenacity | >=8.3 | — | Mapillary only; broad HTTP-error retry policy |
| pytest | >=8 | — | Test dependency in production dependency list |
| pytest-asyncio | >=0.23 | — | Test plugin in production dependency list |
| numpy | >=1.26 | 2.3.5 | Embedding, aggregation, split and metrics |
| slowapi | >=0.1.9 | — | Global per-IP middleware limit |
| ruff (dev) | >=0.4 | — | E/F/I/B/UP rules; no type/docstring checks |
| setuptools (build) | >=68 | 84.0.0 | Build backend |
| wheel (build) | unbounded | 0.48.0 | Build tooling |

Python is declared `>=3.11`, without an upper bound or a reproducible minor
version. Tracked bytecode is tagged CPython 3.13, not evidence of supported
runtime compatibility. No mypy/pyright dependency/configuration or check command.
`qdrant/qdrant:latest` is mutable. Docker's PyTorch CUDA base is tagged
`2.2.0-cuda12.1-cudnn8-runtime` without a digest; pip and all dependencies resolve
again during build. No GPU reservation is specified in Compose.

Outdated compatibility assumptions are verified, rather than claiming every
minimum is vulnerable or every package needs upgrading:

- [Qdrant client 1.16.0 release notes](https://github.com/qdrant/qdrant-client/releases/tag/v1.16.0)
  explicitly remove `search`; `VectorStore.query` calls it at line 55.
- [pandas 3.0 release notes](https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html)
  document removal of including grouping columns in `DataFrameGroupBy.apply`.
  Ingestion already has repeated repair code for missing `s2_cell`.

A full latest-version or vulnerability inventory was not performed. There is
no resolved environment to audit for deployment vulnerability status.

Compose requires a nonexistent example `.env`, only orders startup without
readiness, and does not set `QDRANT_HOST=qdrant`. README's `localhost` is wrong
inside the API container. README does not start Qdrant, build an index, specify
working directory or document model download requirements. Its
"production-grade" claim is not supported by current verification.

## Configuration and hardcoded-value inventory

`config.py:12-46` exposes the following defaults via environment variables;
these are configurable defaults, not scattered immutable constants:

| Group | All defaults |
| --- | --- |
| Credentials/index/model | token `""`; host `localhost`; port `6333`; collection `swespy_embeddings`; model `openai/clip-vit-large-patch14`; dim `768` |
| Geography/preprocessing | S2 `14`; Sweden bbox `(10.5,55.0,24.2,69.1)`; image_size `224` (unused) |
| Batching/retrieval | batch `64`; max images/cell `50`; top-k `5`; vector upsert batch `256` |
| Mapillary | query limit `2000`; page size `1000`; minimum page `100`; max bbox area `0.01` square degrees; download concurrency `20`; grid lon/lat `6/6` |
| Input/API | clean minimum `100` KiB; upload max `10` MiB; MIME types JPEG/PNG; rate `60/minute` |
| Training/logging | learning rate `1e-4`; decay `1e-2`; log `INFO`; testing `False` |
| Settings mechanism | `.env` relative to CWD; empty env prefix; case insensitive; cache maxsize `1` |

Other operational constants and repeated values:

| Location | Values / behavior |
| --- | --- |
| `data/mapillary.py:37-85,114-115,149-180,202` | Graph URL; timeout 30s; 5 attempts; exponential wait multiplier/min/max 1/1/30s (duplicated retry decorators); default search 2000; field selection `id,geometry,captured_at,thumb_2048_url`; transient status list 429/500/502/503/504; page halving; default concurrency 20; progress twentieths; `.jpg` filenames |
| `data/pipeline.py:23-25,90-108,170-176,191-192,223-246` | CWD; `data/raw`, `data/metadata`, `images.parquet`; recursive 4-way split and max depth 2; sample seed 42; 1024-byte KiB; output column contract repeated |
| `scripts/index_vectors.py:15-23` | CWD parquet path repeated; payload columns fixed |
| `model/embedder.py:21-28,32-66` | CUDA else CPU; RGB; PyTorch processor tensors; L2 epsilon 1e-12; float32; processor padding; actual resize controlled by processor, not `image_size` |
| `model/trainer.py:35-55,69-83,98-125,127-149` | RGB/device repeated; seed 42; split 80/10/10%; workers 0; default epochs 5; CE/AdamW/cosine; scheduler minimum 1; parquet/checkpoint paths; `best_model.pt`; initial best accuracy 0 |
| `scripts/train.py:17` | epochs 5 repeated |
| `model/evaluate.py:14,49-64` | Earth radius 6371 km; thresholds 25/100/200 km; proportions despite `pct_` names; all-zero empty-eval metrics |
| `vector_store/qdrant_client.py:30-32,41,64-91` | cosine metric; point IDs starting at 1; missing payload defaults 0/empty; nonnegative weight clamp; uniform zero-weight fallback; empty prediction `(0,0,0)`; mean similarity confidence |
| `api/main.py:36-54,86,113` | Dummy dim 768 and Stockholm `(59.3293,18.0686)`, score 0.9, token `89c25`, image ID 1; title/version; module-global app construction |
| `api/routes/predict.py:25-29,45,59` | HTTP 400 messages fixed to JPEG/PNG and 10MB even if settings differ; hash prefix 16; confidence clamp [0,1] |
| `api/schemas.py:13,22` | similarity/confidence bounds [0,1]; no coordinate bounds or state enum |
| `api/routes/*.py`, `api/middleware.py` | `/predict`, `/health`; UUID4; `X-Request-ID`; milliseconds via 1000; fixed log fields |
| `Dockerfile`, `docker-compose.yml` | image tags above; `/app`; ports 8000/6333; host 0.0.0.0; bind paths `./data`, `./qdrant_storage`; `.env`; uvicorn entry point |
| `README.md`, `pyproject.toml`, package init | localhost/6333 example, reload command; package/API version 0.1.0 repeated; Python floor 3.11; ruff length 100/target py311; test asyncio auto |
| `tests/*` | Generated PNG 20x20 and JPEG 10x10; mocked 768-d output and 224x224 input; Stockholm; S2 14; 1km tolerance; norm atol 1e-5; aggregation fixtures 59/18,60/19 and weights .8/.2 |

No configurable HTTP retry/timeout policy, upload pixel budget, model revision,
storage retention, deployment coverage boundary, calibration or abstention
thresholds. Invalid zero/negative batch sizes, grid divisions, limits and S2
levels are not rejected at settings load. Secrets use plain strings rather than
redacted settings types. No secrets were found in tracked source declarations.

## Correctness, maintainability and failure handling

- `MapillaryClient.batch_download` worker declares only `completed, failed`
  nonlocal. Assignment to `skipped_existing` makes it local throughout the
  worker: existing files fail on increment, and a fresh successful download
  reaches an unbound read in `finally`. Missing-record progress excludes skips.
- Metadata fallback catches every HTTP error, including credentials/permission
  failures, splits bboxes and eventually returns `[]`. Download HTTP failures
  are counted without per-image cause or failed-ID output. `run` can overwrite
  usable metadata with an empty parquet while the script reports completion.
- Mapillary retries all HTTP errors, including permanent 4xx, and then layers
  page-size reduction and subdivision on top; no Retry-After policy. File writes
  are non-atomic; existing files are blindly trusted. Geometry extraction assumes
  a dict and at least two coordinates. Invalid/null geometry and disk errors
  propagate; coordinate ranges/finiteness are never checked.
- Ingestion's grouping repair is repeated before/after cleaning/save. Trainer,
  indexer and pipeline duplicate metadata paths. PIL RGB/device/model setup is
  duplicated. Main duplicates state/test-double setup at construction/startup.
- `Settings.image_size` has no consumers. `evaluate_model` has no repository
  callers and expects a three-item dataloader incompatible with GeoDataset's
  two-item `(pixels,label)` output. Cell-center conversion is used only in its
  unit test; classification/checkpoint is disconnected from inference. These
  are unused/disconnected within this repository, not permission to delete them.
- Indexing accumulates all vectors and images per batch, lacks embedding shape,
  finite/norm and payload validation, has no collection schema/version check,
  and accepts an existing collection solely by name. Upserts/search have no
  explicit application timeout or retry policy (SDK defaults are not a contract).
- Match payload omissions silently become coordinates `(0,0)` and empty cells.
  Cosine similarity can be negative, but MatchResult forbids it, so legitimate
  low-similarity hits can fail response construction. Candidate image IDs are
  dropped by the response schema. No rank field, geographic clustering,
  duplicate-location handling or candidate confidence exists.
- Empty retrieval returns a coordinate outside Sweden rather than an explicit
  unavailable/out-of-coverage state. Coordinate weighted averages may fall
  between distant clusters where no reference exists. Mean cosine similarity is
  uncalibrated and ignores dispersion; clamping does not make it a probability.
- Upload validation trusts declared MIME, reads all bytes before size rejection,
  and lacks corruption/empty image/decompression pixel-limit error mapping.
  PIL decode exceptions and backend failures escape as 500s. No EXIF orientation
  correction, explicit image handle context management, or embedding-dimension
  check. Error messages need to reflect configuration.
- Public APIs mostly have annotations, but lifespan lacks its async iterator
  return type; `list[dict]` and unparameterized dictionaries in data/store code,
  `Any` in evaluate/test components and dynamic app.state obscure contracts.
  Fake model/processor methods and monkeypatch fixtures lack types. Public
  docstrings mostly lack Google-style args/returns/raises; some claims are stale
  (fixed 100KB and "best" returned model). Ruff/type checks were not runnable;
  no clean-check claim is made. Import ordering visibly needs attention.

## Async paths and observability

The expensive request embedding and sync Qdrant query are already offloaded to
threads (`predict.py:36-41`). Do not count them as blocking async calls.
Synchronous aggregation, whole-image SHA256 and logging remain on the event
loop; aggregation is normally small (k=5), but k is not bounded. Async lifespan
directly loads CLIP and performs synchronous collection network calls.
Async ingestion directly performs mkdir/exists/write_bytes and dataframe work;
download concurrency bounds HTTP requests but creates one task per image.
There is no inference concurrency limit or queue, so simultaneous thread/model
work can exhaust RAM/VRAM. Neither model nor Qdrant client is closed at shutdown.

Loguru emits human-readable logs, not structured serialized records. Request
middleware logs ID/path/status/latency only after successful call_next; exceptions
can bypass its completion log/header. Prediction log lacks request ID and
separate embedding duration. `processing_time_ms` excludes upload parsing/read
and response validation, and combines embedding/query/aggregation. App creation
removes global Loguru sinks, affecting other consumers/tests.

## Privacy and misuse findings

No application-level `save`/`write` of uploaded query images was found; embedder
uses BytesIO. **This does not establish memory-only processing:** FastAPI's
UploadFile uses Starlette multipart temporary-file spooling before route checks.
[Starlette request documentation](https://www.starlette.io/requests/) describes
uploaded files as spooled temporary files. Exact thresholds depend on the
unlocked version. Framework cleanup is not a configurable retention policy.

`predict.py:45-52` logs a stable content hash, precise predicted coordinates and
confidence to stderr. Raw bytes are not logged, but this metadata links repeated
images and exposes sensitive inferred locations. No log expiry or retention
setting exists; external log collection cannot be assessed from source.
Mapillary downloads persist reference/training images, not uploads, under
`data/raw/`. Metadata, trained checkpoints and Qdrant vectors/payloads persist
separately. Compose mounts data and Qdrant storage. `.gitignore` excludes raw,
processed, metadata parquet, storage and `.env`, but does not exclude future
`eval/photos/` or checkpoints; tracked bytecode/egg-info bypass ignore rules.
No private evaluation photos should be added before ignore rules are fixed.

No authentication, tenant/client quotas or security audit trail. SlowAPI is a
per-IP limit, not a per-client control, and no shared limiter backend is configured.
No custom upload UI is present; default interactive FastAPI docs are enabled,
and Compose exposes API and Qdrant ports. Deferred B2B requirements are recorded
in `docs/NEXT.md`; no safeguards or endpoints were built during this audit.

## Tests: covered and missing

Four tests exist: one fake-model embed_image shape/norm check; one positive
weighted-mean/confidence calculation; one Stockholm S2-center round trip under
1 km; one successful generated-PNG request with constant dummy components and
response-key checks. The endpoint test bypasses ASGI lifespan and leaves TESTING
in process environment/cache, risking order-dependent behavior. Dummy inference
does not even decode uploaded images and therefore cannot test corruption.

Missing: actual CLIP/embed_bytes/embed_batch parity, orientation, malformed and
oversized uploads, MIME spoofing, negative/nonfinite scores, empty/zero/negative
aggregation, missing/bad payloads, config/env constraints, S2 invalid inputs and
Linköping coverage, Mapillary pagination/retry/failure/resume, metadata integrity,
index IDs and schema, real startup/Qdrant integration, latency/concurrency,
privacy/log assertions, trainer splits/checkpoints, real-photo snapshots,
viewpoint evaluation, retrieval top-k geographic hit rate and calibration.
`evaluate_model` uses 25/100/200 km classifier metrics, not the required
100m/500m/2km retrieval evaluation; empty input misleadingly reports zero error.
There are no measured baseline numbers or regression tolerances.

## Ranked problems

Effort is engineering time excluding review, photo collection, model downloads,
index construction and waiting for external services. Ties prioritize immediate
execution correctness, then trustworthy measurement. Fixes remain gated on review.

| Rank | Severity | Problem and evidence | Estimated effort / intended phase |
| --- | --- | --- | --- |
| 1 | High | Clean-checkout inference cannot be reproduced: unpinned client allows removed Qdrant search; Compose hostname/readiness incomplete; no index/model identity | 1-2 days, Part 2 baseline environment, compatibility fixes made explicit |
| 2 | High | Reference downloading fails on unbound skipped_existing (`mapillary.py:184,205,230`) | 1-2 hours, Part 3 |
| 3 | High | No real retrieval/viewpoint baseline or snapshots; current evaluator targets a disconnected classifier | 1-3 days after private labels/photos/index are supplied, Part 2 |
| 4 | High | False precision: mean similarity confidence, spread ignored, empty/malformed results yield (0,0), negative score schema error | 1-2 days for validation/separation in Part 3; 1-3 days calibration/abstention in Part 4 with adequate eval data |
| 5 | High | Upload privacy and validation: temporary disk spooling, hash/pin logs, unbounded read before size check, corruption errors, missing orientation | 1-2 days, Part 3 |
| 6 | High | Training validation cannot resolve held-out cells; checkpoint unused by inference (`trainer.py:76-113`) | 0.5-1 day investigation/contract decision; record model/eval design in NEXT; no training/model changes now |
| 7 | High | Ingestion silently tolerates terminal failures, can replace metadata with empty output; pandas 3 grouping contract changes | 1-2 days, Part 3; any pre-baseline compatibility prerequisite must be recorded |
| 8 | High | Index corruption/staleness risk from restarted integer IDs, unchecked existing schema and missing payload validation | 1-2 days, Part 3 without changing Qdrant schema |
| 9 | Medium | Startup sync work, no explicit Qdrant timeout/retry/cleanup, unconstrained inference concurrency, incomplete failure/timing logs | 1-2 days, Part 3 |
| 10 | Medium | Configuration and quality contract gaps: Sweden defaults, missing env example, unused/disconnected code, weak types/docstrings, generated tracked files and no CI checks | 1-2 days, Parts 2-3; scope/model decisions deferred to NEXT |

B2B auth/quotas/audit controls are additionally high priority before any external
deployment but explicitly outside the implementation scope. Do not expose this
as a public upload service based on the README's current claim.

## Review handoff

Part 2 has not started. Before it, review this audit and identify the actual
reference collection/model revision plus user-owned photos and labels. Capture
current behavior without fabricating a baseline. If a supported pinned older
dependency set is needed to run the current source, record it; do not silently
fix the inference code and call the result an unmodified baseline. Full model
recommendations and costs in NEXT remain provisional until measured evaluation.
