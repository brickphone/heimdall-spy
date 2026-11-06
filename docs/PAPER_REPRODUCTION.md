# MegaLoc paper reproduction and dense Linkoping protocol

This work reproduces pretrained evaluation, not the unreleased multi-dataset
training run. The earlier 317-photo city pilot was too sparse for a standard
25m VPR test: only one of its 30 queries had any indexed reference within 25m.
Its averaged pin is also application logic, not the paper's primary metric.

## Published benchmark

Use the complete Baidu split explicitly identified by the author in
[MegaLoc issue 13](https://github.com/gmberton/MegaLoc/issues/13):
**5,207 database images and 3,208 queries**. All 3,208 queries have positives at
25m. Other Baidu splits and 10m/3D-distance evaluation are different protocols.
The paper reports **Recall@1 87.7% and Recall@10 98.0%** for this setup.

Downloaded the complete approximately 360 MiB dataset from the author's public
rsync distribution. The source README and download provenance are retained under
ignored `swespy/data/benchmarks/`; images are used locally and not committed.
Dataset citation: Sun et al., *A Dataset for Benchmarking Image-Based Localization*,
CVPR 2017. Evaluation citation: Berton and Masone, *MegaLoc: One Retrieval to Place
Them All*, CVPR Workshops 2025.

The evaluator source is pinned to
`gmberton/VPR-methods-evaluation@9c8dfd496b49a49549323b94a6af204cbee91604`.
Original source and license are retained. The execution copy changes the query
batch size from one to the requested batch size; a factory supplies the previously
pinned pretrained MegaLoc instead of importing unrelated model dependencies.
Label parsing, exact FAISS `IndexFlatL2` search and recall computation are unchanged.

The Mac uses native Python 3.12.8, torch 2.5.1 and MPS, with the previously verified
CPU fallback/contiguous-input compatibility adapters. A separate benchmark
environment adds pinned optional dependencies from
`swespy/eval/requirements-benchmark.txt`; the application's lock is unchanged.

Checks before inference:

- Exact split counts, coordinate labels and full image-byte/dataset fingerprint.
- CPU/MPS descriptor agreement on two images.
- Individual/batched descriptor agreement; observed maximum absolute difference
  on the probe was approximately `2.51e-7`.
- Normalized, finite 8,448-dimensional descriptors. A persistent input-tensor cache
  supports interruption recovery and rejects malformed cached descriptors.

After inference, saved descriptors are searched again and recall is recomputed
independently. It must agree with the official displayed output within rounding.
The protocol declared a 0.15 percentage-point tolerance for the paper's displayed
R@1/R@10 before observing results. Results must be reported even if that tolerance
is missed; it is not permission to modify the data or targets.

## Dense Linkoping development test

The fixed rectangle `(15.618, 58.406, 15.631, 58.413)` covers approximately
0.6 km² around central Linkoping. Acquisition examined 747 available metadata
records from a budget of 6,000, excluding panoramas and retaining source attribution,
capture sequences, GPS and camera headings. Position/heading sampling uses 10m XY
cells and 45° bins, balanced across spatial cells, with a cap of 1,200 references.

The resulting gallery contains **369 references across all eight heading bins**.
Twelve new capture sequences were reserved before image inspection or predictions.
Original query sequences are protected from entering the new gallery. New queries
also exclude the old reference sequences and old image IDs. Exactly **18 queries**
passed sequence, 40m spacing, three-per-sequence and exact/perceptual duplicate
safeguards. The requested 30-query target was not filled by relaxing the rules.

Fourteen of the 18 queries have a gallery reference within 25m; 15 have one within
100m and all 18 within 500m. The official evaluator includes all 18 in its recall
denominator. A separate covered-query result exposes model retrieval when a
positive exists. No images are added to any existing Qdrant collection.

All JPEGs are decoded/validated without the old arbitrary 100 KiB reference filter.
Coordinate-bearing hard links format them for the authors' dataset loader while
avoiding duplicate image storage. They are portable as ordinary image files.
Metadata, original images, attribution, manifest and links are under ignored
`swespy/data/linkoping_dense/central_v1/`.

A density comparison was fixed before predictions: uniform reference subsets at
10%, 25%, 50% and 100%, using seeds `[0, 1, 2, 3, 42]`, the same frozen queries and
the same encoded descriptors. Coverage and all-query recall are reported for each
subset. This tests reference availability within the same area without retraining
or choosing queries based on predictions. It is a small development experiment,
not a claim of general geolocation accuracy.

## Completed Mac results — 6 November 2026

The full Baidu run reproduced both published results with the released pretrained
weights: **R@1 87.7494% (2,815/3,208)** and **R@10 97.9738% (3,143/3,208)**.
These round to the paper's **87.7% / 98.0%**, within the predeclared tolerance.
Encoding took approximately 44 minutes on the M1. The final FAISS search stalled
in OpenMP after MPS inference; all descriptors had been saved. Scoring resumed in
a fresh CPU process using the unchanged official search/recall code and one FAISS
thread, taking about two seconds. Independent scoring agreed. The driver now sets
one FAISS thread to avoid that wait.

The dense Linkoping run encoded and evaluated 369 references and 18 queries from
nine held-out capture sequences in approximately 50 seconds:

| Metric at 25m | All 18 queries | 14 queries with a positive |
| --- | --- | --- |
| Recall@1 | 61.11% (11/18) | 78.57% (11/14) |
| Recall@5 | 66.67% (12/18) | 85.71% (12/14) |
| Recall@10 | 66.67% (12/18) | 85.71% (12/14) |
| Recall@20 | 72.22% (13/18) | 92.86% (13/14) |

The fixed density comparison used these same 18 queries and pretrained descriptors.
Values below are means across the five predeclared reference subsets:

| Reference count | Coverage at 25m | Recall@1 | Recall@10 |
| --- | --- | --- | --- |
| 37 | 44.44% | 22.22% | 33.33% |
| 93 | 66.67% | 34.44% | 50.00% |
| 185 | 74.44% | 45.56% | 58.89% |
| 369 | 77.78% | 61.11% | 66.67% |

This supports increasing gallery coverage before training. It is a small local
development test; the earlier 30-query pilot used different queries and must not
be treated as a controlled before/after comparison with this new 18-query test.

Local artifacts (ignored by Git):

- [Baidu report](../swespy/data/benchmarks/runs/baidu-20261106T105553.510877Z/report.html).
- [Linkoping report and density plot](../swespy/data/benchmarks/runs/linkoping_dense-20261106T115241.727823Z/report.html).
- [Attributed query photos and candidate map](../swespy/data/benchmarks/runs/linkoping_dense-20261106T115241.727823Z/map.html).

Each run retains protocol, dataset fingerprints, descriptors, official source
execution copies, predictions and independently verified metrics. Additional
persistent benchmark data, dense images and optional environment total about
1.25 GiB; transient macOS memory/swap use is separate. Validation: 61 tests passed,
one private snapshot test skipped; new benchmark modules passed Ruff and mypy.

## Interpretation and repeatability

Paper reproduction checks the implementation on the paper's split; it does not
guarantee identical performance on a new city. The local test uses Mapillary
imagery and unverified GPS/headings. Pretraining overlap has not been audited.
Nearby camera positions also do not guarantee shared visible content.

No paid compute, training, confidence calibration, production model switch or
reference overwrite occurs. Existing CLIP/MegaLoc indexes and reports are retained.
Commands for preparing the optional environment, resuming the Mac job, evaluating
the dense split, rendering reports and moving data to a later CUDA run are in
[`swespy/eval/README.md`](../swespy/eval/README.md).
