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

## Results

Pending the first run.
