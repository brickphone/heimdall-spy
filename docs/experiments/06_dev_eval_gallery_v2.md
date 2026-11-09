# Dev evaluation on gallery v2: plain MegaLoc retrieval, no re-ranking

Date: 2026-11-08. **Dev set only (66 Commons photos); the test set was not used.**
Prediction is the top-ranked reference. Rates carry Wilson 95% CIs. Files:
`swespy/eval/results/baseline_v2.json`, `dev_recent.json`, `dev_historic.json`
(`python -m eval.dev_eval --index …`).

| Within | Old gallery (2,489) | `recent` (32,600) | `historic` (91,714) |
| --- | --- | --- | --- |
| 25 m | 0/66 · 0.0% [0.0–5.5] | 6/66 · 9.1% [4.2–18.4] | **13/66 · 19.7% [11.9–30.8]** |
| 100 m | 1/66 · 1.5% [0.3–8.1] | 15/66 · 22.7% [14.3–34.2] | **26/66 · 39.4% [28.5–51.5]** |
| 500 m | 7/66 · 10.6% [5.2–20.3] | 23/66 · 34.8% [24.5–46.9] | **30/66 · 45.5% [34.0–57.4]** |
| 2 km | 26/66 · 39.4% [28.5–51.5] | 36/66 · 54.5% [42.6–66.0] | **43/66 · 65.2% [53.1–75.5]** |
| Median error | 3,073 m | 1,341 m | **657 m** |

### Failure types and re-ranking ceiling

| | Old | `recent` | `historic` |
| --- | ---: | ---: | ---: |
| success (top-1 ≤ 100 m) | 1 | 15 | 26 |
| ranking (≤ 100 m reference in top 50, not top 1) | 5 | 7 | 11 |
| retrieval (covered, none in top 50) | 7 | 28 | 26 |
| no_coverage (no reference ≤ 100 m) | 53 | 16 | 3 |
| true_in_top50_at_100m (re-ranking ceiling) | 6/66 · 9.1% | 22/66 · 33.3% [23.2–45.3] | **37/66 · 56.1% [44.1–67.4]** |
| Truth ≤ 100 m in top 1 / 5 / 20 / 50 | 1/4/4/6 | 15/17/18/22 | 26/31/34/37 |
| Truth ≤ 500 m in top 1 / 5 / 20 / 50 | 7/18/28/33 | 23/31/40/48 | 30/39/46/55 |

### Robustness (within 500 m)

| | Old | `recent` | `historic` |
| --- | --- | --- | --- |
| Photographer-balanced (bootstrap CI) | 20.0% [5.0–38.3] | 62.2% [39.8–83.3] | **68.8% [50.8–85.6]** |
| Excluding the 3 largest photographers (25 photos) | 4/25 · 16.0% | 15/25 · 60.0% | **17/25 · 68.0%** |

## Majority-location baseline: a caveat

The baseline as defined in Part 1 is "mean position of the most populated 1 km gallery
cell". It **depends on the gallery's composition**:

| Gallery | Majority cell | Within 500 m |
| --- | --- | --- |
| Old | city centre [536, 6474] | 15/66 · 22.7% [14.3–34.2] |
| `recent` | [534, 6477] (dense 2018+ drives) | 1/66 · 1.5% [0.3–8.1] |
| `historic` | city centre [536, 6474] | 15/66 · 22.7% [14.3–34.2] |

So the `recent` report's "CI above majority = true" is an artefact of a weak baseline
cell. Against the stable city-centre baseline (15/66, upper CI bound 34.2%):

- `recent`: 23/66, lower bound 24.5% → **overlaps**, part (b) not met.
- `historic`: 30/66, lower bound 34.0% → **overlaps by 0.2 pp**, part (b) not met.
- `historic` meets part (a) (≥ 40% within 500 m): 45.5%.

On the dev set the demo target is therefore **not met** by plain retrieval, narrowly.
Recommendation (user decision): freeze the baseline as the city-centre location
(15/66 on dev) so it no longer moves with gallery composition.

## Does the 2015-era imagery help or hurt?

It helps, clearly and on every measure:

- Within 100 m: 15 → 26 photos; within 500 m: 23 → 30; median error 1,341 → 657 m.
- no_coverage drops 16 → 3; the re-ranking ceiling rises 22 → 37.
- Per photo, compared with `recent`: within 100 m, **12 gained and 1 lost**; within
  500 m, **9 gained and 2 lost**. The error was lower for 22 photos, higher for 13,
  unchanged for 31.
- With `historic`, 46/66 top-1 matches are pre-2018 images, and 21 of the 30 correct
  (≤ 500 m) top-1 matches come from pre-2018 imagery.
- Cost: the two lost photos both switched to a 2015 top-1 match, which is the
  expected distractor effect of older imagery. On this dev set the net effect is
  strongly positive.

## What the numbers say about next steps

- Coverage is no longer the main failure: no_coverage is down to 3/66 (from 53).
- The remaining failures split into **ranking** (11: right place in the top 50, not
  top 1) and **retrieval** (26: covered, but nothing correct in the top 50).
- Re-ranking (Part 5) can at most reach the 100 m ceiling of 37/66. It can only fix
  the 11 ranking failures; the 26 retrieval failures need a better global descriptor
  or more viewpoint-matched references.
- Caveats: 66 photos with clustered photographers give wide CIs; Commons coordinates
  are lower precision; no settings were tuned on these numbers.
