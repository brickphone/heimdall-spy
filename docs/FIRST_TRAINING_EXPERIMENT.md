# Failure audit and first trained geographic head — 6 November 2026

Completed the requested15-case failure audit, repaired the classifier training/
checkpoint contract, trained a small geographic model locally, and compared it
with the frozen MegaLoc baseline on all66 recent ordinary photos.

## Findings from the failure audit

The sample was fixed before visual inspection:5 first-choice failures with a
reference within100m,5 with nearest references100–500m away, and5 without a
reference within500m. Cases are selected by a deterministic ID hash; no failures
were removed from the66-photo evaluation set.

Source EXIF GPS was available for14/15 cases and agrees with the stored labels
within0.054m. One source lacks EXIF GPS. This checks consistency with source
metadata, not physical ground-truth accuracy. No location labels were changed.

Visual comparison of query, nearest2 references, top1 match and best top5 showed:

- Nearby cameras often show different subjects:railway paths instead of a station
  forecourt, residential lawns instead of a waterfront pavilion, or street views
  without the sculpture/plaque photographed by the user.
- Detail crops, movable foreground vehicles, seasons and viewing position cause
  substantial image differences. Geographic proximity does not establish visible
  overlap. The nearest2 views are not an exhaustive search of every gallery image.
- A park case has a plausible235m candidate in the top five, but ranks a different
  park1.9km away first. Ranking matters as well as missing visual coverage.
- Five sampled queries have no camera reference within500m; classifier training
  cannot create a missing exact reference image.

[Inspect the15-case evidence](../swespy/data/failure_audit/recent66_v1/evidence.html).
The fixed sample, source-EXIF check, coordinate deltas, contact sheets and per-case
observations are under ignored `swespy/data/failure_audit/recent66_v1/`.

## What was repaired

`Trainer.split_by_cell` now holds out whole capture sequences while retaining
training examples for every output class. It rejects missing sequence metadata;
it does not silently fall back to a random adjacent-frame split. Whole groups
are moved into training when necessary to cover rare classes, and the actual
fold membership is recorded in the experiment protocol.

The legacy CLIP classifier now freezes its complete encoder/projection, keeps
that encoder in evaluation mode, trains its head only, saves even a zero-accuracy
first best checkpoint, refuses an existing checkpoint, restores the selected
best model, and offers explicit `GeoClassifier.load` inference with its recorded
class order. The legacy metadata must include sequences before that script can
run; the old317-image parquet is not a valid substitute for the new protocol.

The actual experiment uses the already cached **MegaLoc descriptors**, avoiding
another expensive image-encoding pass. `GeoHead.load` binds the new checkpoint
to the exact encoder identity and class mapping. `eval.search_gallery` now accepts
`--classifier` and actually loads it for geographic predictions/reranking. The
existing default API still uses CLIP; no automatic production switch occurred.

## Training protocol

- Frozen MegaLoc backbone; a learned linear head for143 fixed1km UTM33V cells.
  This is real head training, not full backbone fine-tuning or a PIGEOTTO reproduction.
- 2,730 training,295 validation and363 internal test Mapillary images; complete
  capture sequences remain disjoint. All classes have training examples.
- All66 public photos stay outside training and hyperparameter selection. They
  were previously inspected during development, so this is not a new blind test.
- Seed42; AdamW learning rate0.03,weight decay0.01; inverse-square-root class
  weights;batch256;maximum60 epochs;patience8. Best epoch selected by validation
  cell accuracy, with validation loss as tie-breaker.
- The first run exposed NumPy scalar metadata incompatible with safe checkpoint
  loading. Numeric metadata was converted to plain floats and the same deterministic
  protocol rerun. The failed artifact was retained; no public result had been
  inspected at that point. Safe weights-only loading and exact round-trip outputs
  now pass.
- Best epoch3:validation cell accuracy57.97%; internal test30.03%. This drop
  cautions against assuming cross-capture or ordinary-photo generalization.
- Combined retrieval reranks50 candidates. Weights0,0.02,0.05,0.1,0.2 were fixed
  before evaluation and selected on Mapillary validation against a train-only
  retrieval gallery. Validation500m hits were203,201,195,190,186 respectively:
  the selected weight is **0**, meaning no classifier contribution.

## Actual ordinary-photo results

First-choice predictions on the same66 photos:

| Method | Within100m | Within500m | Within1km | Median error |
| --- | --- | --- | --- | --- |
| MegaLoc retrieval | 1/66 | 7/66 | 18/66 | 3,643m |
| Trained geographic head | 0/66 | 14/66 | 29/66 | 2,089m |
| Combined, validation-selected weight0 | 1/66 | 7/66 | 18/66 | 3,643m |
| Constant most-common training-cell centre | 0/66 | 15/66 | 33/66 | 1,018m |
| Constant median training coordinate | 0/66 | 9/66 | 27/66 | 1,182m |

The two constant-location checks are post-hoc diagnostics, not model-selection
inputs. They use training coordinates only and ignore query images. The geographic
head improves some thresholds over retrieval but **does not beat the constant
most-common-area guess**. The public sample is spatially concentrated, and a
headline claiming useful learned-geolocation improvement would be misleading.
Classifier predictions are1km grid centres, not exact camera pins.

We therefore retained the baseline rather than switching the API or tuning on
these66 results. Neither training a head nor simply adding more arbitrary road
photos has established the desired ordinary-photo capability.

## Artifacts, commands and checks

Experiment: `swespy/data/classifier/linkoping_linear_v1/` (ignored).

- [Comparison report](../swespy/data/classifier/linkoping_linear_v1/report.html).
- [Trained-head candidate map](../swespy/data/classifier/linkoping_linear_v1/classifier-map.html).
- `head.pt`, `protocol.json`, `results.json` and `geographic-prior-check.json`.

From `swespy/`:

```bash
MPLCONFIGDIR="$PWD/data/benchmarks/matplotlib" .venv-benchmark/bin/python -m eval.failure_audit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-benchmark/bin/python -u -m eval.train_geo_head
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -m eval.search_gallery \
  --index data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/gallery-index.json \
  --classifier data/classifier/linkoping_linear_v1/head.pt --image /path/to/photo.jpg
```

The training command returns the existing completed result rather than replacing
it. Use a new `--out` directory for a separately justified experiment. Preserve the
test images, their labels and original reference/index artifacts.

Validation:78 tests passed,one private snapshot skipped; changed modules passed
Ruff/mypy. Tests cover capture isolation/known labels, frozen encoder behavior,
safe checkpoint round-tripping and encoder mismatch rejection. Real native MPS
image inference loaded the trained checkpoint and reproduced its classifier and
combined candidates. No paid compute or external inference API was used.

## Next decision

Acquire more useful reference/training views of street-facing landmarks and
pedestrian areas, including views like ordinary photos. Keep the66-photo set as
development evidence; add a fresh, more spatially balanced and creator-separated
holdout before further accuracy claims. Evaluate cross-view retrieval fine-tuning
only with appropriate positive pairs and that independent validation protocol.
