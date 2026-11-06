# Expanded Linköping MegaLoc evaluation — 6 November 2026

Encoded the frozen **3,388-reference city gallery** and evaluated all **48 existing
queries**, retaining the central18 and legacy30 cohorts. The original datasets
and CLIP API were not changed. This was pretrained inference/indexing, not training.

The official evaluation took **859.5 seconds (14m19s)** on the M1, including
descriptor saving and search:3,049 images newly encoded,387 descriptors reused.
The run passed CPU/MPS parity checks. GPU inference used substantial macOS swap
and slowed toward the end; that is transient memory pressure, not dataset size.

## Measured results

Uncovered queries remain in the denominator. Recall@K means at least one of the
first K references has its camera within the stated radius of the query label.

| All 48 queries | Reference coverage | Recall@1 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- |
| 25m | 15/48 (31.25%) | 11/48 (22.92%) | 13/48 (27.08%) | 13/48 (27.08%) |
| 100m | 18/48 (37.50%) | 12/48 (25.00%) | 15/48 (31.25%) | 15/48 (31.25%) |
| 500m | 41/48 (85.42%) | 17/48 (35.42%) | 21/48 (43.75%) | 23/48 (47.92%) |

For the original18 central queries,25m R@1 remains **11/18 (61.11%)** and R@10
remains **12/18 (66.67%)**. The original369-reference result was independently
recomputed from the identical reference/query subset of the new descriptor arrays
and matched the recorded baseline. Wider acquisition did not improve this25m score.
At500m, central R@1 fell from16/18 to15/18, illustrating that extra references can
also introduce distracting candidates.

The legacy30 cohort is much weaker: only1/30 queries has a reference within25m,
3/30 within100m and23/30 within500m. Its R@1 is0/30 at25m/100m and2/30 at500m.
Report cohorts separately; comparing the new48-query aggregate with the old
18-query score would mix different evaluation sets.

## What this tells us

The expanded gallery is searchable, but broad geographic extent is not dense
photographic coverage. These results show both reference gaps and retrieval
failures. At500m,41 queries have a possible match but only17 retrieve one first;
coverage alone therefore does not explain all failures. At25m, only15 queries
have any possible positive, limiting what retrieval can achieve.

These remain small same-provider street-image development sets with unverified
GPS labels. They do not establish accuracy on ordinary public/user photos. An
independent ordinary-photo set and targeted coverage inspection should guide the
next training or collection experiment. No claim is made that more images or
fine-tuning will automatically fix all failures.

## Artifacts and verification

Run directory:
`swespy/data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/`.

- [Cohort/radius report](../swespy/data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/city-summary.html).
- [Attributed query photos and candidate map](../swespy/data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/map.html).
- [Official25m report](../swespy/data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/report.html).
- `gallery.faiss`: exported exact L2 index of normalized8,448-D vectors.
- `gallery-index.json`: checksum, encoder identity and reference rows in exact
  index order, including coordinates and attribution.
- `city-results.json`: all cohort/radius metrics and369-reference central baseline.
- `protocol.json`, descriptor arrays, predictions and pinned evaluator execution
  copy retain the full provenance. Dataset fingerprint:
  `8e2b9a628a847369eb79424ed00da4a5f86667de3405d1569bac55628a0285b8`.

The exported index reproduces all saved benchmark rankings. A separate local-photo
MPS search reproduced all five expected candidates. An initial native search
process crashed when FAISS was imported before Torch; matching the verified
benchmark's Torch-before-FAISS import order resolved the crash. No descriptors
or results were changed. Keep that import order and single-thread FAISS on Mac.

The benchmark-environment suite passed66 tests with one private snapshot skipped;
changed Python modules passed Ruff/mypy. Search commands and resume instructions
are in [the evaluation README](../swespy/eval/README.md#encode-evaluate-and-search-the-expanded-gallery).
