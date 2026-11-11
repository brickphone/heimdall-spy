# Automatic region search: people masking and automatic crops

Date: 2026-11-11. **Dev (66), street-image (48) and field sets only; nothing tuned on
field or test photos; test set untouched.** Gallery: v2 `historic` (91,714 references).
Re-ranking as in experiment 07 (K = 5, w = 0.005, 25 inliers, 500 m disagreement).

## Pre-registration (written before any evaluation run)

All numbers below were fixed before the first run and are not tuned.

**Person mask.** torchvision DeepLabV3-MobileNetV3-Large, weights
`COCO_WITH_VOC_LABELS_V1` (SHA-256 `fc3c493d…`), VOC class 15 ("person"), input
resized to a 520 px shorter side, argmax label per pixel, mask scaled back with nearest
neighbour. Person share = masked pixels / all pixels.

**Variants** (`swespy/model/variants.py`):

- (a) the whole photo, or the user's manual region when one is given;
- (b) the whole photo with people filled with the image's mean colour, before both
  MegaLoc and ALIKED, only if people cover ≥ 2% (otherwise it equals (a));
- (c) up to 6 crops of the whole photo, in this order:
  - the largest rectangle on a 16 × 16 grid with < 10% person area (ties: closest to
    the centre; skipped when it is the whole photo);
  - left, centre and right vertical strips, each 40% wide;
  - the central horizontal band, 50% high.

  A crop is skipped if it covers < 25% of the photo or the existing quality rule calls
  it sky or ground only (edge share < 0.02).

**Search.** All variants are encoded in one batched MegaLoc pass and searched to depth
50. The lists are merged, keeping each reference's best variant (equal similarity: the
manual region or whole photo wins). The merged top 5 are verified, each against the
variant that retrieved it, and the existing re-ranking decides; the winning variant is
that of the top re-ranked candidate. The confidence rule is unchanged.

**Configurations compared:** `whole` (a only), `masked` (a + b), `full` (a + b + c).

**Metrics:** within 100 m / 500 m / 1 km with Wilson 95% intervals, median error,
confident ("ok") and low-confidence counts, and **confident but wrong** ("ok" and more
than 500 m off). Paired changes at 500 m (fixed / broken) against `whole`.

**Tightening rule, if needed.** If `full` has more confident-but-wrong photos than
`whole` on **dev**, a winner from an automatic variant (masked or crop) is "ok" only
with ≥ 50 inliers (twice the base 25), and nothing else changes. It is decided on dev
only and then reported on the other sets as is.

**Field photos with a manual region:** whether the automatic winner's region overlaps
the manual region with IoU ≥ 0.3, and whether both give the same result (within 100 m).

**Named case:** the restaurant photo (whole), with its EXIF GPS read only as ground
truth for scoring, never as an input.

## Results (dev and street-image)

Run 2026-11-11 on a RunPod RTX A5000 (EU-SE-1), `eval/auto_regions.py`, rows in
`data/auto_regions/eval/rows.json`. Before the run, the pod reproduced the Mac on the first
5 dev photos (`eval/pod_check.py`; rule fixed beforehand: same top-1 and state, ≥ 4 shared
top-5 IDs, similarity within 1e-3): 5/5 shared IDs on every photo. The `whole` dev numbers
equal experiment 07 exactly. Wilson 95% intervals; "CBW" = confident but wrong (> 500 m).

**Neither set contains the motivating case.** The largest person share is 7% (dev) and
15% (street-image); only 1 dev and 2 street photos reach the 2% masking threshold, and none
reach 30%. These results therefore measure the automatic crops, not people masking.

### Dev (66)

| Setup | ≤ 100 m | ≤ 500 m | ≤ 1 km | Median | Confident | Confident correct | CBW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| whole | 27 · 40.9% [29.9–53.0] | 31 · 47.0% [35.4–58.8] | 38 · 57.6% [45.6–68.8] | 586 m | 25 | 22/25 · 88.0% [70.0–95.8] | 3 |
| + masking | 27 · 40.9% [29.9–53.0] | 31 · 47.0% [35.4–58.8] | 38 · 57.6% [45.6–68.8] | 586 m | 25 | 22/25 · 88.0% [70.0–95.8] | 3 |
| + masking + crops | 26 · 39.4% [28.5–51.5] | 33 · 50.0% [38.3–61.7] | 36 · 54.5% [42.6–66.0] | 590 m | 22 | 20/22 · 90.9% [72.2–97.5] | **2** |

Paired at 500 m vs whole: crops fixed 2, broke 0. Winners with crops: whole 25, left 14,
band 14, right 10, centre 3. Confidence state whole → crops: ok→low 6, low→ok 3.

### Street-image (48)

| Setup | ≤ 100 m | ≤ 500 m | ≤ 1 km | Median | Confident | Confident correct | CBW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| whole | 29 · 60.4% [46.3–73.0] | 32 · 66.7% [52.5–78.3] | 33 · 68.8% [54.7–80.1] | 15 m | 27 | 25/27 · 92.6% [76.6–97.9] | 2 |
| + masking | 29 · 60.4% [46.3–73.0] | 32 · 66.7% [52.5–78.3] | 33 · 68.8% [54.7–80.1] | 15 m | 25 | 24/25 · 96.0% [80.5–99.3] | 1 |
| + masking + crops | 31 · 64.6% [50.4–76.6] | 33 · 68.8% [54.7–80.1] | 37 · 77.1% [63.5–86.7] | 14 m | 31 | 28/31 · 90.3% [75.1–96.7] | **3** |

Paired at 500 m vs whole: crops fixed 3, broke 2. Confidence state whole → crops:
low→ok 6, ok→low 2.

### Confident but wrong, and the tightening rule

Dev: 3 → 2 with crops, so the pre-registered tightening rule (≥ 50 inliers for automatic
winners) **was not triggered** and nothing was changed. Street-image: 2 → 3 (+1). The rule
is decided on dev only, so it stays off; the increase is reported, not acted on.

### Failure types (wrong = > 500 m)

| Set | Setup | Wrong | Flagged low | Confident | 0.5–2 km | > 2 km |
| --- | --- | --- | --- | --- | --- | --- |
| Dev | whole | 35 | 32 | 3 | 13 | 22 |
| Dev | + crops | 33 | 31 | 2 | 8 | 25 |
| Street | whole | 16 | 14 | 2 | 5 | 11 |
| Street | + crops | 15 | 12 | 3 | 8 | 7 |

Changed photos (whole → crops): dev 2882 m → 134 m (centre), 546 → 493 m (band); street
3814 m → 3 m (right, low → ok), 2561 → 36 m (left), 589 → 283 m (right), and two broken:
135 → 980 m (right) and 187 → 961 m (centre), both still flagged low.

### Latency (median per photo)

| Stage | whole | + masking | + crops |
| --- | --- | --- | --- |
| Person mask | — | 60 ms | 51 ms |
| Variant crops | 19 ms | 1 ms | 50 ms |
| MegaLoc (one batch) | 46 ms | 39 ms | 104 ms |
| Exact search (91,714 refs) | 291 ms | 293 ms | 1,360 ms |
| Verification (top 5) | 258 ms | 260 ms | 290 ms |
| **Total, A5000** | **0.62 s** | **0.67 s** | **1.88 s** |

Mac (M1, MPS), measured on 37 dev photos with `whole` only, under memory pressure: 9.7 s
(search 5.1 s, verification 3.9 s). Search time grows with the number of variants (×4.7 on
the pod), so with crops the Mac is estimated at ≈ 25–30 s under the same pressure: **the
8 s target is not met on the Mac.** Batching the variants and verifying only the merged
top 5 are already in place; the remaining cost is the single-threaded exact search.

**Correction (experiment 15):** the Mac search times above were measured with the
descriptor matrix not resident in memory (swap pressure, cold page cache). Warm, the exact
search on the Mac takes ≈ 70 ms (whole) and ≈ 370 ms (with crops). The estimate of
25–30 s does not hold; with a warm cache a query with crops is estimated at ≈ 5–6 s on the
Mac, dominated by verification. An end-to-end Mac measurement is still to be done.

### Not yet run

The restaurant photo (named case, Mac only, outside these metrics) and the field set with
manual regions (after the field photos are saved).
