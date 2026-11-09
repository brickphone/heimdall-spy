# Part 5: local-feature re-ranking (ALIKED + LightGlue) on the historic gallery

Date: 2026-11-08. **Dev set only (66 photos); weight and threshold tuned on dev; test
set untouched.** Gallery: v2 `historic` (91,714 references, now the default).
Compared with plain MegaLoc on the same gallery (`eval/results/dev_historic.json`).
Majority baseline: frozen city-centre coordinate (config), 15/66 within 500 m.

## Method

- **Retrieval:** MegaLoc top **K = 50** (configurable: `retrieval_depth`, `--k`).
- **Verification:** ALIKED (aliked-n16, ≤ 1,024 keypoints, 640 px) + LightGlue
  (ALIKED weights) at pinned commit `5a650c4`. Weights are SHA-256-pinned in
  `swespy/model/rerank.py`. A fundamental-matrix USAC-MAGSAC (2 px) counts RANSAC
  inliers. Degenerate match sets, where OpenCV finds no model, count as 0 inliers.
- **Score:** `cosine + w · log(1 + inliers)`. **w = 0.005**, chosen on dev by a rule
  fixed in advance: maximise photos within 100 m, then within 500 m, then prefer the
  smaller w.
- **Predicted location:** score-weighted mean of the top-5 re-ranked candidates within
  **100 m** of the top one (others treated as outliers). The 100 m was fixed a priori.
- **Low-confidence state** if the top match has **< 25 inliers**, or if another
  verified candidate (≥ 25 inliers) in the top 5 is **> 500 m** from it. The 25 is the
  smallest threshold whose confident subset reaches ≥ 80% within 500 m on dev; the
  500 m was fixed a priori.
- **One path:** `swespy.model.rerank.finalize` is used by both `GeoLocator.rerank_result`
  (the API) and `eval/rerank_dev.py`. The API returns the top 5 with similarity,
  inliers and score, plus `confidence_state`/`confidence_reasons`.

## Results (66 dev photos)

| Within | Plain MegaLoc (historic) | + re-ranking |
| --- | --- | --- |
| 25 m | 13/66 · 19.7% [11.9–30.8] | 13/66 · 19.7% [11.9–30.8] |
| 100 m | 26/66 · 39.4% [28.5–51.5] | **27/66 · 40.9% [29.9–53.0]** |
| 500 m | 30/66 · 45.5% [34.0–57.4] | **31/66 · 47.0% [35.4–58.8]** |
| 2 km | 43/66 · 65.2% [53.1–75.5] | 43/66 · 65.2% [53.1–75.5] |
| Median error | 657 m | 586 m |
| Photographer-balanced (500 m) | 68.8% [50.8–85.6] | 70.5% [52.0–87.5] |
| Excluding top-3 photographers (500 m) | 17/25 · 68.0% | 18/25 · 72.0% |
| Majority baseline (frozen, 500 m) | 15/66 · 22.7% [14.3–34.2] | 15/66 · 22.7% [14.3–34.2] |

| Failure type (100 m) | Plain | + re-ranking |
| --- | ---: | ---: |
| success | 26 | 28 |
| ranking | 11 | 9 |
| retrieval | 26 | 26 |
| no_coverage | 3 | 3 |
| true_in_top50_at_100m (ceiling) | 37 | 37 |
| Truth ≤ 100 m at top 1 / 5 / 20 / 50 | 26/31/34/37 | 28/31/35/37 |

- **Ranking failures fixed: 2 of 11.** Previous successes lost: **0**. Within 500 m:
  1 gained, 0 lost.
- **Demo target on dev:** (a) 47.0% ≥ 40% — met; (b) lower CI bound 35.4% > baseline
  upper bound 34.2% — **met, by one photo**. Caveats: the weight was tuned on these
  photos (selected for 100 m, not 500 m), and the flip from "not met" depends on a
  single photo. The test set (Part 6) decides.

### Weight grid (dev; selection rule fixed in advance)

| w | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.15 | 0.3 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| within 100 m | 25 | **27** | 27 | 25 | 24 | 23 | 23 | 21 | 22 |
| within 500 m | 30 | **31** | 31 | 32 | 32 | 33 | 34 | 32 | 31 |

Larger weights trade 100 m hits for 500 m hits. The pre-declared rule picks w = 0.005;
choosing w = 0.15 for its 34 hits at 500 m would be tuning to the target.

## Why so few ranking failures were fixed

For the 9 unfixed ranking failures, the correct reference's inlier count is similar
to, or lower than, the wrong top-1's. Typical values: correct 12–26 inliers vs wrong
10–21, and in most cases the highest count in the top 50 belongs to a wrong reference.
Unrelated images routinely reach 10–25 "inliers" (the noise floor of a 2 px
fundamental-matrix fit), and pedestrian Commons photos share few true matches with
car-mounted Mapillary frames. The 2 fixes were clear geometric wins (53 vs 21 and
135 vs 39 inliers). Re-ranking cannot help the 26 retrieval failures (nothing correct
in the top 50).

## Confidence: inliers are a strong reliability signal

| Top-1 inliers | Photos | Within 500 m | Within 100 m |
| --- | ---: | --- | ---: |
| 5–14 | 21 | 2/21 · 9.5% [2.7–28.9] | 1 |
| 15–29 | 20 | 6/20 · 30.0% [14.5–51.9] | 5 |
| 30–59 | 10 | 8/10 · 80.0% [49.0–94.3] | 7 |
| ≥ 60 | 15 | 15/15 · 100% [79.6–100] | 14 |

| State | Photos | Within 500 m |
| --- | ---: | --- |
| ok | 24 | **22/24 · 91.7% [74.2–97.7]** |
| low | 42 | 9/42 · 21.4% [11.7–35.9] |

## Latency (per query, cold, K = 50, median of 6 dev photos)

| Device | MegaLoc + search | Verification | Total |
| --- | ---: | ---: | ---: |
| M1 MPS (ALIKED on MPS, LightGlue on CPU) | 1.7 s | 51.3 s | **53 s** |
| CPU only | 1.8 s | 120.8 s | **123 s** |

ALIKED's deformable convolution has no MPS kernel (CPU fallback), and LightGlue is
faster on the M1's CPU (0.31 s/pair) than on MPS (1.54 s/pair). Verification dominates.
Options, not yet implemented: smaller K (cost scales roughly linearly), fewer
keypoints, precomputed reference features (~91k × ~250 KB ≈ 23 GB), or a CUDA server.

## Environment changes

- LightGlue (git, `5a650c4`), kornia 0.9.0 and OpenCV 4.10.0.84 were added to
  `pyproject.toml` and `uv.lock`; torch 2.5.1 and numpy 1.26.4 are unchanged.
- OpenCV 5.0 was rejected because it requires numpy ≥ 2. An accidental torch upgrade
  during installation was reverted, and a partial run under OpenCV 5.0 was discarded.

## Walk routes (pedestrian viewpoints)

Selected from the `historic` gallery and OSM street types only (`python -m
eval.walk_routes`). Eval photo locations were not used. The pedestrian-only network
(footway, pedestrian, path, steps, living_street) totals **1,073 km**, of which
**639 km (60%) has no gallery reference within 15 m**. The gallery is almost entirely
car-mounted imagery, which matches the retrieval failures above (pedestrian photos vs
road references). The proposal picks the five 500 m blocks with the most unserved
pedestrian paths, at least 1.5 km apart:

| # | Centre | Unserved pedestrian paths | Named streets / area |
| --- | --- | ---: | --- |
| 1 | 58.4241, 15.5850 | 7.8 km of 10.5 | Skäggetorp: Skäggetorpsgatan, Storgården, Rusthållaregården, Klockaregården |
| 2 | 58.3792, 15.5765 | 6.9 km of 7.8 | Lambohov: Lantmannagatan, Spannmålsgatan, Utsädesgatan |
| 3 | 58.3927, 15.5850 | 6.7 km of 9.0 | Valla/Vallastaden: Ada Arwedsons gata, Hannes Alfvéns gata, Greifswaldsbron |
| 4 | 58.4061, 15.6108 | 6.1 km of 9.9 | Central-west park and footpaths (unnamed in OSM) |
| 5 | 58.3927, 15.6365 | 5.8 km of 7.5 | South-central park and footpaths (unnamed in OSM) |

Map: `data/gallery_v2/walk-routes.html`; GeoJSON: `walk-routes.geojson`. Record each
walk with a forward-facing phone video plus a GPX track (`eval/walks.py`).
