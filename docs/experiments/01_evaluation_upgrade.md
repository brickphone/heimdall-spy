# Part 1 — Evaluation upgrade and baseline_v2

Date: 2026-11-07. No model, gallery image or evaluation photo was changed.

## What changed

- `swespy/eval/geo_eval.py` (model-independent): Wilson 95% CIs on every rate;
  per-photo `nearest_reference_m`, `true_in_top_k` (top 1/5/20/50 at 100 m and
  500 m), `true_in_top50_at_100m` and `failure_type`; majority-location and
  random-reference baselines; photographer-balanced accuracy; the two-part demo
  target check; dev-set photographer/spatial distribution with clustering flags.
- `swespy/eval/dev_eval.py`: runs the 66-photo development set against a saved
  MegaLoc index restricted to the project polygon and writes the full report.
- `swespy/swespy/config.py`: approved project area `LINKOPING_AREA` (12-point
  polygon, 158 km², map: `area_polygon_proposal.html`) and `in_area()`.
- Tests: `swespy/tests/unit/test_geo_eval.py` (Wilson vs published values,
  synthetic failure types, baselines, balanced metric, target logic, polygon).

### Definitions

| Term | Definition |
| --- | --- |
| Prediction | Location of the top-ranked reference |
| failure_type | `no_coverage`: no reference ≤100 m of truth · `success`: top-1 ≤100 m · `ranking`: covered and a ≤100 m reference in top 50 · `retrieval`: covered, none in top 50 |
| Re-ranking ceiling | Share with a reference ≤100 m in the top 50. Geometric verification only confirms the same scene, so the 500 m top-50 figure is **not** a re-ranking ceiling |
| Majority location | Mean position of the most populated 1 km UTM gallery cell (no eval labels) |
| Random reference | Exact expectation of predicting a uniformly random reference (fractional count) |
| Demo target | On the test set: (a) ≥40% within 500 m **and** (b) system 95% CI lies entirely above the majority baseline's 95% CI |

## Gallery restriction

The saved 3,388-reference index was filtered to the polygon: **899 references
outside it were excluded** and 2,489 used. The flat index is rebuilt in memory
from stored vectors; raw images and the saved index are untouched.

## baseline_v2 (66 dev photos, 2,489 references) — `swespy/eval/results/baseline_v2.json`

| Within | MegaLoc top-1 | Majority location | Random reference |
| --- | --- | --- | --- |
| 25 m | 0/66 · 0.0% [0.0–5.5] | 1/66 · 1.5% [0.3–8.1] | 0.0% [0.0–5.5] |
| 100 m | 1/66 · 1.5% [0.3–8.1] | 1/66 · 1.5% [0.3–8.1] | 0.1% [0.0–5.6] |
| 500 m | 7/66 · 10.6% [5.2–20.3] | **15/66 · 22.7% [14.3–34.2]** | 2.9% [0.8–10.3] |
| 2 km | 26/66 · 39.4% [28.5–51.5] | 36/66 · 54.5% [42.6–66.0] | 16.2% [9.2–26.9] |
| Median error | 3,073 m | 1,043 m | 4,616 m |

Demo target on dev: (a) 10.6% < 40% — **not met**; (b) CI overlaps the
majority baseline and lies below it — **not met**.

**Coverage** (reference within radius of truth): 25 m 3/66, 50 m 5/66,
100 m 13/66 (19.7% [11.9–30.8]), 500 m 56/66. Median `nearest_reference_m` 190 m
(IQR 120–336 m).

**Truth in top-k** (count of 66):

| Radius | top-1 | top-5 | top-20 | top-50 |
| --- | --- | --- | --- | --- |
| 100 m | 1 | 4 | 4 | 6 |
| 500 m | 7 | 18 | 28 | 33 |

**Re-ranking ceiling: 6/66 = 9.1% [4.2–18.4].** Even perfect re-ranking of the
current top 50 could not approach the 40% target; coverage must change first.

**Failure types:** success 1 · ranking 5 · retrieval 7 · **no_coverage 53**.

## Clustering and photographer robustness

- 15 photographers; effective count 6.8. Largest: Stockholm100 15, Lundgren8 14,
  Bene Riobó 12 — the top three took 41/66 (62%). **Flagged.**
- 34/66 photos lie within 1 km of one point (central Linköping). **Flagged.**
- Photographer-balanced within 500 m (mean of per-photographer rates, photographer
  bootstrap CI): MegaLoc 20.0% [5.0–38.3] vs majority 39.9% [18.0–63.2].
- Excluding the three largest photographers (25 photos): MegaLoc 4/25 · 16.0%
  [6.4–34.7] vs majority 11/25 · 44.0% [26.7–62.9] within 500 m. Failure types:
  ranking 4, retrieval 2, no_coverage 19.
- These are diagnostics only; the dev set is not reweighted or filtered.

## Conclusion

Retrieval over the current gallery is worse than always guessing the city
centre, under every view (raw, photographer-balanced, top-3 excluded). 53/66
failures are coverage failures and only 6/66 have a same-scene candidate in the
top 50, confirming that densification (Part 4) must come before re-ranking can
matter.

## Reproduce

```bash
cd swespy
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-benchmark/bin/python -m eval.dev_eval \
  --out eval/results/baseline_v2.json
```

The MPS fallback flag is required because the saved index identity records it.
Query descriptors come from the hash-keyed cache; the 25-photo test set is not read.
