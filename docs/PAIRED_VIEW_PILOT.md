# Recent paired-view data pilot — 6 November 2026

Built a bounded free-source pilot for learning to match different views of the
same places. This task prepared data; **no new backbone fine-tuning or accuracy
improvement is claimed**. The earlier66 public photos remain unchanged and unused
for training.

## What was acquired and reviewed

Recent Mapillary queries across the centre, station, campus and Skäggetorp yielded
754 unique metadata records across12 capture sequences. After preserving old
held-out sequences,687 nonpanoramic records remained. The server capture-date
filter was checked again locally against the fixed2023-11-06–2026-11-06 window.

Commons provided recent camera-GPS/license/date-filtered ordinary-photo candidates.
Year-category and search queries added173 previously unseen metadata records,
13 of which passed the date/GPS/license/scene requirements. Existing cached
metadata was reused. No Flickr, paid source or external image-inference service
was used.

Geographic proximity and headings only proposed pair candidates. Visual inspection
rejected bridge-versus-embankment views and other cases without confidently shared
content. The strict cross-day-only search was sparse; a second proposal separately
labels same-capture viewpoint changes, which can be training examples but are
not independent test observations. This follows the broad motivation of
[viewpoint-robust place-recognition training](https://arxiv.org/abs/2308.10832),
not a reproduction of that paper.

## Frozen dataset

Ignored root: `swespy/data/paired_views/recent_pilot_v2/`.

- **13 visually confirmed positive pairs**,26 images,8 spatial groups.
- **8 training pairs**,5 spatial-validation pairs. Whole250m groups stay together;
  training pairs within100m of any validation image were buffered out. The actual
  nearest train/validation camera separation is296.6m.
- **2 cross-day pairs**,11 same-capture viewpoint pairs. All accepted pair images
  are from **one photographer** (`la2`). The spatial validation shares that
  photographer/capture style; it is not creator-independent validation.
- **25 fresh Commons outdoor photos**,9 photographer keys,16 occupied1km cells.
  These are new image IDs, not the original66 or previously downloaded proposals.
  They include streets, buildings, public art, parks, water and transport context.
- **8 separate Mapillary capture-validation photos** from other photographer keys.
  They show vehicle-mounted motorway views and must not be presented as ordinary
  phone-photo evaluation. They are absent from the old reference index.
- All59 included images are dated2024-03-11 through2026-09-23. The Commons subset
  spans2024-03-11 through2026-09-01. JPEGs have no EXIF/XMP location hints; dates,
  GPS claims, source attribution and license links stay in the manifest.

Training and fresh-holdout photographer keys are disjoint after case normalization,
including `LA2` versus `la2`. Identities behind different usernames were not
independently verified. Fresh holdout photographers may have appeared in the old
66-photo development set; they are separated from this pilot's training creators.
Sampling balances1km cells,allows at most one photo per100m cell and caps each
photographer at six. These quotas do not produce a random or representative city
sample, and repeated places/styles remain possible.

Exact/perceptual duplicate checks protect the original66 and previous48 queries,
screen within-pair trivial copies, separate new holdouts from proposed training
images, and check new Commons photos against the old reference gallery. Capture
date and hash checks run again before freeze. No model predictions were used to
select or review these examples. The new25-photo set has not been evaluated yet.

## Evidence and commands

[Open the reviewed pairs and fresh holdout](../swespy/data/paired_views/recent_pilot_v2/report.html).
The dataset retains protocol, proposal, explicit review decisions/reasons, contact
sheets and frozen manifest. The initial strict cross-day proposal is preserved at
`data/paired_views/recent_pilot_v1/`; it was never used to train a model. Raw source
feasibility metadata, including signed download URLs, remains ignored under
`data/paired_views/feasibility_v1/`.

From `swespy/`, with the private Mapillary token already configured:

```bash
.venv-benchmark/bin/python -m eval.paired_views discover
.venv-benchmark/bin/python -m eval.paired_views acquire
# Only after actually reviewing images and recording review.json:
.venv-benchmark/bin/python -m eval.paired_views freeze
.venv-benchmark/bin/python -m eval.paired_views report
```

Existing proposals validate cached bytes rather than silently selecting new
examples. Freeze refuses overwriting a manifest. Re-running discovery reads its
metadata checkpoints. A separately scoped future acquisition should use a new
versioned dataset, preserving this pilot and all old evaluation artifacts.

## Is this enough to expect training improvement?

**No guarantee, and this pilot is too small for a credible broad fine-tuning
claim.** Eight training pairs from one photographer establish the acquisition,
review and separation workflow; they do not replace a diverse multi-location
training corpus. Same-session pairs can teach viewpoint consistency, but also
invite memorization of one camera, walk, lighting condition or scene.

Before spending on GPUs, expand the reviewed pair set across more independent
photographers, capture sessions and locations, with visible shared content. Keep
the new Commons holdout unused during model/hyperparameter selection. Use
validation-only decisions, unchanged baselines and constant-location checks when
the next controlled training experiment is justified. Do not claim training
improved the project merely because the data pipeline now works.

Validation:81 tests passed,one private snapshot skipped; paired-view code passed
Ruff/mypy. Source/query hashes, date interval, photographer separation, spatial
buffer and metadata stripping were checked. No old model, reference index or
evaluation membership was changed.
