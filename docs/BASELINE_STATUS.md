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

Windows, bundled Python 3.12.14, locked project dependencies installed in `.venv`.
No model weights, reference datasets or evaluation photographs downloaded;
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
  **Docker is unavailable here: container build and startup are unverified.**
  Docker Python is pinned to 3.12.8; local verification used 3.12.14. Runtime
  provenance records the actual interpreter; cross-platform snapshot parity is
  not claimed.

## Missing prerequisites

The user does not know the locations of the evaluation photos/labels or the
existing reference collection. None are present in this checkout. There is no
measured `eval/results/baseline.json` or captured private snapshot. Creating
placeholder results would violate the task, so neither has been fabricated.

To complete Part 2, supply user-owned queries labelled with camera positions and
the existing populated Linköping reference collection (or a private backup), plus
the existing model/cache revision if available. Use `swespy/eval/README.md` to
validate and capture once. Actual model loading follows the unmodified source;
recorded model and processor identities must be frozen for subsequent comparisons.
Before committing a measured aggregate baseline, review provenance for sensitive
collection identifiers. Photos, labels and prediction snapshots stay private.

If the historical reference index cannot be recovered, creating one is a
separate data decision: it cannot be called a reproduction of historical results.
Do not begin behavior-changing refactoring, confidence calibration or model
experiments until a genuine baseline and snapshot are available and this Part 2
output is complete and committed.
