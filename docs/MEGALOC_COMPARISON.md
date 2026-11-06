# MegaLoc comparison, 2026-11-06

Completed follow-up: [paper reproduction and dense Linkoping results](PAPER_REPRODUCTION.md). The measurements below retain their original experiment context.

Tested the pretrained [MegaLoc](https://github.com/gmberton/MegaLoc) model against
the frozen [CLIP street sanity check](STREET_CHECK.md). There was no fine-tuning,
new reference acquisition, query selection after prediction, or production-model
replacement. Results are mixed and do not support a straightforward API switch.

## Identical evaluation inputs

- Same 317 indexed reference image IDs, bytes on disk and coordinate payloads.
- Same 30 frozen queries from 27 capture sequences and 11 creators. Their IDs,
  byte hashes, GPS labels and order were checked against the CLIP report.
- Same cosine search, five candidates, geographic thresholds and similarity-weighted
  coordinate aggregation. Reference/query content was not cropped or augmented.
- MegaLoc used the authors' preprocessing: RGB, tensor conversion, ImageNet
  normalization, then antialiased bilinear resize to 322x322. Its pretrained
  output dimension is 8,448. No PCA, whitening, threshold fitting or retraining.
- Original CLIP index fingerprints matched before and after the experiment.
  MegaLoc was indexed in a separate timestamped collection, verified with a
  reference replay and fingerprinted before/after evaluating the queries.

## Results on the same 30 queries

| Measure | CLIP | MegaLoc |
| --- | ---: | ---: |
| Median averaged-pin error | 1,901.77m | 1,604.48m |
| Averaged pin within 100m | 0/30 (0%) | 0/30 (0%) |
| Averaged pin within 500m | 5/30 (16.67%) | 2/30 (6.67%) |
| Averaged pin within 2km | 18/30 (60%) | 23/30 (76.67%) |
| Any top-five candidate within 100m | 1/30 (3.33%) | 3/30 (10%) |
| Any top-five candidate within 500m | 7/30 (23.33%) | 13/30 (43.33%) |
| Top-ranked candidate within 500m | 2/30 (6.67%) | 3/30 (10%) |
| Median top-ranked candidate error | 2,202.63m | 1,839.55m |
| Median best-of-five candidate error | 800.84m | 596.61m |

MegaLoc reduced averaged-pin error for 18 queries and increased it for 12. Candidate
retrieval improved, but ranking and averaged-pin accuracy remain weak. The median
improvement coexists with worse 500m pin accuracy; it must not be described as an
across-the-board accuracy improvement.

Coverage is unchanged: only five queries have any indexed reference within 100m,
and 29 have one within 500m. A nearby reference is not proof that it shows the same
scene or viewing direction. These same-provider street-only results, based on a
small development set and unverified GPS labels, do not establish accuracy for
arbitrary photos or window/elevated viewpoints. Pretraining overlap with these
Mapillary photos was not audited. Similarity values are uncalibrated and not
comparable probabilities across models.

## Runtime and immutable model identity

- Official source revision: `5fe0dd697c4a70ba3e23607f6716ab3c606b16db`.
- Source SHA256: `e3a66ae823abbbbc5e7975c649419a33036400eaa9fa5a955dd97eaa37f46c75`.
- Hugging Face weights revision: `a0f34722c4297ff787e022433799250180860af7`.
- Weights SHA256: `d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8`.
- Official source/license/readme and pretrained license/readme/weights are cached
  under ignored project data. Upstream source and weights were not modified.
- Native macOS Python 3.12.8, locked torch 2.5.1 and torchvision 0.20.1; M1 MPS
  execution with CPU fallback for positional bicubic interpolation. An experiment
  pre-hook supplies value-identical contiguous Conv2d inputs to avoid the pinned
  MPS backend's stride bug. These adapter settings are part of descriptor identity.
- CPU/MPS parity on two real images: minimum descriptor cosine 1.0 and maximum
  absolute difference approximately `7.09e-7`. The adapter also has a CPU
  value/convolution-preservation regression test.
- Encoded 347 images in batches of four in **207.43 seconds**, including cache
  checks and descriptor writes. No paid compute or training. CLIP's previous
  report used a Linux CPU API path, so these runs are not a controlled comparison
  of model latency. MegaLoc query timing fields exclude precomputed embedding.

## Local artifacts and verification

Ignored `swespy/data/megaloc/experiments/20261106T102349_004608/` contains:

- `experiment.json`: reference byte hashes, original index and model provenance.
- `results.json`: both models' metrics, paired results, MegaLoc predictions,
  model/evaluator/lock checksums and collection fingerprints.
- `comparison.html`: comparison table and all 30 paired pin errors.
- `map.html`: MegaLoc camera/prediction/candidate map and attributed query previews.

The comparison links to the preserved original CLIP map. Headless Chromium loaded
both pages, verified all 30 map photo cards and map initialization, and screenshots
were visually inspected. Native regression suite: **54 passed, 1 private snapshot
skipped**. Changed Python files pass Ruff 0.9.1; the experiment module passes mypy
with imported modules skipped and missing third-party stubs ignored.

The original API still serves CLIP; this task added an isolated experiment only.
The repeatable native command is in [`swespy/eval/README.md`](../swespy/eval/README.md).
Further development should investigate denser references, candidate ranking and
location presentation, then validate decisions on fresh held-out photographs.
