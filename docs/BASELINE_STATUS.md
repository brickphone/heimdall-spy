# Part 2: prepared, measured baseline pending

The user approved continuing after reviewing the audit on 2026-10-16.
Part 3 has not started. Application inference, training, S2 level, embedding model
and Qdrant schema remain unchanged from the audited revision.

## Implemented

- Python 3.12 compatibility environment with exact direct dependency pins,
  pinned build dependencies and a generated 103-package transitive `uv.lock`.
  Qdrant client 1.12.1 retains the existing search API; pandas 2.2.3 retains the
  current ingestion grouping behavior. These are compatibility choices for a
  new reproducible baseline, not a reconstruction of an unknown deployed
  environment or a security-hardening claim.
- One-command Docker Compose setup, fixed Qdrant server 1.12.5, correct container
  hostname, bounded readiness wait, optional `.env`, persistent model cache and
  local-only published ports. This starts services; it does not fabricate an index.
- Private photo/label folders and ignore rules; loader rejects missing/empty
  labels, unsafe paths, invalid coordinates/viewpoints, corrupt images, duplicate
  queries and exact copies of local reference images.
- Real API evaluation, preserving startup/inference behavior. TESTING is rejected;
  empty/missing collections fail before startup. Results include median metres,
  within 100m/500m/2km and geographic top-k hit percentages overall/by viewpoint.
- Once-only baseline/private snapshot capture, model/preprocessing/index/dataset
  identity, timestamped results, per-viewpoint baseline comparisons and opt-in
  live snapshot test. Capturing a baseline requires all four viewpoint types;
  capturing snapshots requires at least ten distinct queries. Snapshot identity
  and prediction comparisons are never replaced automatically.
- CI-style check command runs all lint/type/test gates and preserves failures.

## Verified locally

Initial validation: Windows, bundled Python 3.12.14, locked project dependencies
installed in `.venv`. At that stage no model weights, reference datasets or
evaluation photographs were downloaded;
no training or indexing performed; no external image inference APIs called.

- Full pytest: **25 passed, 1 skipped**. Four pre-existing tests passed.
- The skipped test is the actual CLIP/Qdrant/private-photo snapshot gate. Its
  infrastructure is present; fixed predictions have not been captured or verified.
- Ruff passes for new evaluation, startup/check scripts and tests.
- Mypy passes for the six new evaluation/script source files with missing
  third-party stubs ignored and imported legacy modules checked separately.
- Full-project Ruff: **28 existing findings**; full-project mypy: **3 existing
  errors** in config and API main. `scripts/check.py` correctly exits nonzero.
  These are not hidden or fixed ahead of the baseline.
- Missing-label validation reports an actionable error without importing/loading
  CLIP or writing results. Private image/label/snapshot/results paths are ignored.
- Compose YAML parses and its Qdrant version/hostname settings were checked.
  Docker was unavailable at the initial check; see the follow-up below for the
  completed container verification. Docker Python is pinned to 3.12.8; local
  verification used 3.12.14. Runtime
  provenance records the actual interpreter; cross-platform snapshot parity is
  not claimed.

## Docker follow-up verification, 2026-10-16

After the user installed Docker Desktop and requested a runtime check:

- Started Docker Desktop successfully; engine 29.8.2 and Compose v5.6.0.
- Actual `docker compose config --quiet` passed; `docker compose up --build -d`
  completed successfully, including locked dependency installation on Linux.
- Both `swespy-api-1` and `swespy-qdrant-1` are running with localhost-only ports.
  The real CLIP model loaded and its pretrained weights/cache became available;
  application startup completed without inference-source changes.
- API `/health` and `/docs`: HTTP 200. Qdrant `/readyz`: HTTP 200.
- Collection `swespy_embeddings`: **0 reference points**, cosine vectors of size
  768. No existing Docker containers or named volumes were found before startup.
  This does not establish whether old reference data exists elsewhere on disk.
- An in-memory synthetic 64x64 PNG passed through real `/predict`: HTTP 200,
  approximately 2.59s inference processing time, pin `(0,0)`, confidence 0 and
  empty matches. This verifies runtime execution and the audited empty-index
  behavior, not geolocation accuracy. Non-image MIME input returned HTTP 400.
- No reference imagery, evaluation photos or training data was downloaded;
  no training or indexing was performed. The services were left running.
- Disk usage is material: active images total approximately 9.47 GB. Cleared
  all 12 disposable build-cache records from this first build (reported 9.20 GB),
  then verified both services remained running and build cache was zero.
  Windows C: still reported only approximately 1.7 GB free afterward. No images,
  containers, model cache or reference data were deleted; Windows disk-space
  recovery/VM compaction was not performed.

Docker is installed under
`C:\Users\matti\AppData\Local\Programs\DockerDesktop\resources\bin`.
The existing terminal did not inherit the new PATH entry, including the Docker
credential helper; a process-local PATH addition allowed the build. Restart
VS Code to pick up the installation's PATH update. No permanent PATH changes
were made by this check.

## Missing prerequisites

The historical Mac reference collection was reported lost. A subsequent authorized
rebuild populated a new 189-image central-Linköping pilot collection; see
[REBUILD.md](REBUILD.md) for prerequisite changes, scope and live checks.
On 2026-11-06 a fresh M1 Mac rebuild acquired 399 photos and indexed 317 accepted
references across 42 cells. This is a new collection, not restoration of the
189-reference index. The rebuild also required a narrow API correction for
floating-point cosine overshoot; see the Mac rebuild record for validation.
Independent user-owned evaluation photos/labels are still absent. There is no
measured `eval/results/baseline.json` or captured private snapshot. Creating
placeholder results would violate the task, so neither has been fabricated.

A separate automated Mapillary street sanity check was completed on 2026-11-06:
30 queries from 27 held-out capture sequences, median averaged-pin error 1.90km,
16.67% within 500m. See [STREET_CHECK.md](STREET_CHECK.md) for selection safeguards,
coverage, limitations and the local interactive report. These same-provider
street-only measurements do not replace the independent four-viewpoint baseline.

To complete Part 2, supply user-owned queries labelled with camera positions and
freeze the rebuilt reference collection and current model/cache revision.
Use `swespy/eval/README.md` to
validate and capture once. Actual model loading follows the unmodified source;
recorded model and processor identities must be frozen for subsequent comparisons.
Before committing a measured aggregate baseline, review provenance for sensitive
collection identifiers. Photos, labels and prediction snapshots stay private.

The rebuilt reference index is a new data decision and cannot be called a
reproduction of historical results. Ingestion prerequisites were repaired first
under the user's rebuild authorization; the broader inference refactor is still gated.
Do not begin behavior-changing refactoring, confidence calibration or model
experiments until a genuine baseline and snapshot are available and this Part 2
output is complete and committed.
