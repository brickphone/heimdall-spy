# Agent handoff — Linköping geolocation

Updated: 2026-11-06. Read this file before resuming project work.

## Latest completed task — free paired-view data pilot

The user asked whether training will improve results and authorized the next
concrete job:check free paired-view sources and build a small data pilot. Do not
promise improvement. No new backbone fine-tuning or paid compute was run here.

Read `docs/PAIRED_VIEW_PILOT.md`. `eval/paired_views.py` now supports discover,
acquire,freeze,report. Uses the configured private Mapillary token and truthful
Commons bot identification;Flickr remains excluded. The capture-date window
2023-11-06–2026-11-06 was enforced. Raw discovery metadata is preserved under
`swespy/data/paired_views/feasibility_v1/` (ignored,includes signed image URLs).

Recent Mapillary discovery found754 unique records/12 sequences,687 after old
held-out-sequence/nonpano protections. Commons search/category discovery inspected
173 new file records,13 recent geotagged scenes qualified. The first strict
cross-day-only proposal yielded too few genuine shared scenes; preserved as
`data/paired_views/recent_pilot_v1/` and never trained. A second proposal labels
cross-day and same-capture viewpoint tiers separately. All candidates reviewed
BEFORE model predictions;GPS proximity alone did not qualify positive pairs.

Frozen pilot at `swespy/data/paired_views/recent_pilot_v2/`:

- 13 reviewed pairs/26 photos/8 spatial groups:8 train pairs,5 spatial validation.
- 2 cross-day,11 same-capture pairs. ALL pair photos come from one creator la2.
  Do not call the spatial pair validation creator-independent.
- Train/validation groups separated with100m buffer;actual minimum296.6m.
- 25 new Commons holdout photos:9 photographer keys,16 occupied1km cells,disjoint
  creators from pair training. No previous proposal/test image IDs reused. Authors
  may overlap old66 test authors;they are separated from the NEW training creators.
- 8 separate Mapillary validation photos from other creators show motorway driving;
  they are not ordinary-photo validation and are absent from the old reference index.
- 59 used images,all dated2024-03-11 through2026-09-23. New Commons dates through
  2026-09-01. Pixels have no EXIF/XMP labels;source GPS/heading remain unverified.
- protocol.json,proposal.json,review.json,manifest.json,contact sheets,report.html
  preserve selection and evidence. New25 holdout has NOT had predictions run.
- Original66 and older48 preserved;new images duplicate-screened;no old model,
  reference index or evaluation membership was changed. Code checks:81 passed,
  one private snapshot skipped;Ruff/mypy passed.

Important:8 training pairs from ONE photographer is a workflow pilot,not enough
for a credible fine-tuning gain. Do not start a large GPU run or tune on the new25
photos. Next expand real shared-content pairs across photographers/places/sessions
and define a controlled validation-only training protocol. No new accuracy gain
has been measured in this latest step.

## Latest completed task — failure audit and first actual model training

The user explicitly authorized auditing failures, repairing training/inference and
running the first controlled geographic-classifier experiment. This work is now
complete; no paid compute or production API model switch was authorized/performed.

Read `docs/FIRST_TRAINING_EXPERIMENT.md` for full results and exact commands.

- Deterministic15-case failure audit:five failures each with nearest references
  <=100m,100–500m,>500m. Inspected query,nearest2,top1,best top5 images before
  training. Many nearby views show different subjects; proximity overstates
  usable visual overlap. Source EXIF GPS agrees within0.054m for14/15 cases;
  one lacks EXIF GPS. This is metadata consistency,not independent ground truth.
- Audit artifacts:`swespy/data/failure_audit/recent66_v1/`,including evidence.html,
  fixed sample,contact sheets,source EXIF,label deltas,per-case assessment.
- Repaired legacy `Trainer.split_by_cell` to split whole captures while keeping
  classes known; missing sequence metadata fails explicitly. CLIP encoder and
  projection now fully frozen;head-only optimization;best checkpoint restored;
  explicit `GeoClassifier.load` inference. No default API switch.
- Actual experiment trained a143-class1km geographic linear head on cached
  MegaLoc descriptors,not a PIGEOTTO reproduction or backbone fine-tuning.
  Splits:2730 training,295 validation,363 internal test images,with disjoint
  sequences. All66 public queries stayed out of training and tuning.
- Best epoch3:validation cell accuracy57.97%,internal test30.03%. Selected
  fusion weight0 using Mapillary validation,so combined retrieval is unchanged.
- Public66 first-choice500m:retrieval7/66,trained head14/66,combined7/66.
  But always guessing the most common TRAINING cell gets15/66. Median error:
  retrieval3643m,head2089m,constant majority-cell1018m. These post-hoc prior checks
  were NOT used for tuning. No useful learned-location improvement established;
  do not advertise the14/66 versus7/66 comparison without this baseline.
- Model/results:`swespy/data/classifier/linkoping_linear_v1/` has head.pt,
  protocol.json,results.json,report.html,classifier-map.html and prior diagnostic.
  Safe checkpoint roundtrip is exact. Initial NumPy-scalar metadata save/load
  failed; fixed to plain floats and reran the same seed/protocol before evaluating
  public queries. Failed artifact preserved; do not load it.
- `eval/search_gallery.py --classifier .../head.pt` loads the trained model and
  returns geographic candidates and combined matches. Actual native MPS image
  inference reproduced the experiment's results. The API still uses CLIP.
- Checks:78 tests passed,one private snapshot skipped;changed modules Ruff/mypy
  clean. No held-out photo/label/reference edits;no paid services.

Next:improve visual reference/training coverage for pedestrian/landmark views and
add a fresh spatially balanced,creator-separated holdout before further claims.
The existing66 are development evidence and must remain out of training. Additional
model tuning on them would compromise evaluation; do not keep fishing for a win.

## Latest task — recent ordinary-photo evaluation completed

User constraints:only photos captured in the last2–3 years; free Commons only;
Flickr/paid services explicitly declined. The HTTP403 was fixed: our original
user-agent lacked contact information. Use the truthful HeimdallSpyResearchBot
identifier with https://github.com/brickphone/heimdall-spy
and HTTP library version, following Wikimedia's policy. Actual API and media
requests now work. Do not tell the user source access is still blocked.

`eval/public_photos.py` acquired date/GPS/license-filtered metadata and JPEGs.
A first broad75-candidate proposal at `data/public_eval/recent_linkoping_v1/`
contained events/portraits/interiors and was never frozen or evaluated. A tighter
scene filter (recorded in protocol) produced75 candidates under
`data/public_eval/recent_linkoping_streets_v1/`. Codex inspected all five contact
sheets, excluded9 inappropriate images BEFORE inference and froze66 outdoor photos
from15 creators. Review decisions/timestamp and hashes are retained. Actual dates:
2023-11-21 through2026-09-01; years2023:1,2024:11,2025:21,2026:33. Query JPEGs have
no EXIF/XMP labels. Do not train on these held-out photos.

Completed MPS evaluation against the unchanged3,388-reference MegaLoc index:

- 25m:coverage3/66;first-choice0/66;top-five0/66.
- 100m:coverage13/66;first-choice1/66 (1.52%);top-five2/66 (3.03%).
- 500m:coverage56/66;first-choice7/66 (10.61%);top-five14/66 (21.21%).
- Median first-choice error3,643m. These results are poor; central Mapillary61.1%
  accuracy did not transfer. Both coverage and incorrect retrieval matter.

Artifacts relative to `swespy/`:
`data/public_eval/recent_linkoping_streets_v1/runs/20261106T135549400456Z/`
contains raw results and the attributed66-query candidate map. Source manifests,
metadata checkpoints, images and pre-inference review sheets live in the dataset
root (all ignored). `docs/PUBLIC_PHOTO_TEST.md` records full results/limitations
and resume commands. Local CLI can evaluate again using cached descriptors.

Checks:75 tests passed,one private snapshot skipped; changed code passed Ruff/mypy.
All66 query hashes/dates and lack of EXIF verified; source-gallery/exported-index
fingerprints preserved. No training, paid services, index writes or API switch.

Next:inspect failures/coverage against this actual-use-case baseline, then repair
the training/evaluation/inference contract for a controlled classifier or retrieval
fine-tuning experiment. Do not remove failures after seeing model predictions.

## User objective and preferences

Build a GeoSpy-inspired demo that locates ordinary outdoor photos in Linköping,
then expand across Sweden. Street scenes and landmarks are an acceptable first
scope; do not redefine the goal as locating only Street View screenshots. The
user prefers automatically sourced evaluation photos over walking outside to
take them. Start on the M1 Mac; cloud migration later is acceptable. No paid
compute or API spend has been authorized. Preserve existing work and datasets.

## Architecture that actually exists

- `swespy/swespy/`: FastAPI API, pretrained CLIP ViT-L/14, 768-dimensional
  embeddings, Qdrant cosine search, top five matches, weighted average GPS pin.
  Confidence is similarity-derived and uncalibrated.
- `swespy/eval/megaloc_compare.py`: pinned pretrained MegaLoc loader, 8,448-D
  normalized descriptors and resumable image-hash cache. Separate experiments;
  the API still uses CLIP.
- `swespy/eval/paper_benchmark.py`: pinned official VPR evaluator, cached
  descriptors, MPS compatibility checks, FAISS exact search, 25m Recall@K.
- PIGEON/PIGEOTTO were discussed earlier but are not implemented in current
  source. A geographic head has now been trained and connected to explicit CLI inference,
  but has not beaten a constant geographic prior or replaced the API. No backbone
  fine-tuning, LLM or RAG answer generation is deployed.

## Recent completed work

1. Reproduced the released MegaLoc model on the exact author-identified Baidu
   split: 5,207 references / 3,208 queries; R@1 87.7494%, R@10 97.9738%, matching
   published rounded values 87.7% / 98.0%. This reproduces pretrained inference,
   not the original training. See `PAPER_REPRODUCTION.md`.
2. Acquired a dense central gallery of 369 references and 18 held-out queries
   across nine capture sequences. R@1 at25m was 11/18; only14/18 have positives.
   Reference thinning showed density matters on the same frozen queries.
3. Expanded references city-wide with `eval/expand_linkoping.py`: **3,388 photos**,
   retaining all369 central references and adding3,019 unique photos. 305 capture
   sequences,31 creators,143 occupied1km cells. Rectangle
   `(15.50,58.34,15.75,58.48)`; bounded first pass, not exhaustive street coverage.
   See `GALLERY_EXPANSION.md` for protocol and artifacts.
4. Protected39 old/reserved query sequences, screened additional images against
   held-out query hashes/perceptual hashes, removed one exact gallery duplicate.
   Verified old manifests and original CLIP index unchanged; API health HTTP200.
5. Added root `README.md` explaining technologies, architecture, setup and limits;
   updated application/evaluation READMEs. City downloader checks:64 tests passed,
   one private snapshot test skipped; changed modules passed Ruff/mypy.

## Previous step — expanded-gallery encoding completed

Encoded all 3,388 city references and evaluated the frozen central18 + legacy30
queries (48 total), using native MPS, batch32 and the existing official driver.
Run: `data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/` relative to
`swespy/`; immutable evaluation dataset: `data/linkoping_dense/city_eval_v1/`.
Time:859.5 seconds;3,049 descriptors newly computed,387 cache hits. No training,
paid compute, production API model switch or source-dataset changes occurred.

Results:

- All48:25m coverage15/48; R@1 11/48 (22.92%), R@5/R@10 13/48 (27.08%).
- All48:100m R@1 12/48 (25%);500m R@1 17/48 (35.42%), R@10 23/48 (47.92%).
- Central18:25m R@1 remains11/18 (61.11%), R@10 remains12/18 (66.67%).
  The369-reference baseline was recomputed from identical saved descriptor subsets
  and exactly matched the recorded central baseline. Expansion did not improve it.
- Legacy30:only1 query has a25m positive; R@1 is0/30 at25m and2/30 at500m.
- These are same-provider development sets, not ordinary-photo accuracy. Missing
  coverage AND wrong retrievals matter; do not claim gallery growth fixed the demo.

`eval/city_benchmark.py` prepares and finalizes the evaluation. The run now contains
`gallery.faiss`, `gallery-index.json` (exact index-row mapping and checksums),
`city-results.json`, `city-summary.html`, `map.html`, official `report.html`, saved
arrays/predictions/protocol and density comparison. `eval/search_gallery.py` accepts
an arbitrary local image and returns ranked coordinates/attribution using pixels.
A real MPS query matched all five saved benchmark candidates. Keep Torch imported
before FAISS and use one FAISS thread; the opposite import order initially caused
an isolated native process crash, fixed before completion.

Checks:66 tests passed,one private snapshot skipped; changed modules passed
Ruff/mypy. Reports were rendered and inspected. Commands are in `swespy/eval/README.md`.
Detailed results: `docs/CITY_EVALUATION.md`. RAG/ML/training explanation:
`docs/ML_TECHNOLOGY.md`.

## Next work and the user's latest question

The user remembers training models months ago and asked why we are not training,
what we are doing now, and when to train. The legacy classifier trainer DOES exist
in `swespy/swespy/model/trainer.py`, invoked by `scripts/train.py`: frozen CLIP
vision backbone, trainable projection/head, S2-cell classification. Its OLD split
held out entire classes while mapping only training labels. That split and the
explicit checkpoint inference contract have now been repaired; see the latest task.
The default API still does not load a classifier checkpoint. See `AUDIT.md` and
`ML_TECHNOLOGY.md`; do not deny historical training work or claim old runs finished.

Next: independent ordinary-photo evaluation, targeted coverage inspection, and
repair the training/split/inference contract before a controlled classifier-head
or retrieval fine-tuning experiment. Keep test photos untouched and compare with
these frozen pretrained baselines. The user subsequently authorized the controlled first training experiment, now
completed above. Paid cloud spend and an API switch have not been authorized.

## Local artifacts (ignored by Git; back these up separately)

All following paths are relative to `swespy/`:

- `data/linkoping_dense/city_v1/manifest.json`: frozen3388-reference gallery.
  `coverage.html` / `coverage.geojson` show actual camera positions; images in
  `originals/` and coordinate-bearing hard links in `images/test/database/`.
- `data/linkoping_dense/central_v1/manifest.json`: original369 references /18
  queries; original central evaluation dataset. Do not modify it.
- `data/mapillary_eval/20261106T095121.923253Z/manifest.json`: older30-query frozen
  set used with317 sparse references. Query capture sequences must stay excluded.
- `data/benchmarks/runs/baidu-20261106T105553.510877Z/`: reproduced paper results.
- `data/benchmarks/runs/linkoping_dense-20261106T115241.727823Z/`: original central
  results, report,map,predictions,descriptor arrays and density experiment.
- `data/megaloc/`: pinned model source, weights and image-hash descriptor cache.
- `data/benchmarks/descriptor_cache/`: official-driver normalized-input cache.
- Original Qdrant CLIP collection:`swespy_embeddings`; earlier MegaLoc collection:
  `swespy_megaloc_20261106t102349_004608`. Never overwrite these for an experiment.

## Running and resuming

Work from `swespy/`. `.venv` is the locked main environment; `.venv-benchmark`
adds pinned optional dependencies from `eval/requirements-benchmark.txt` and
uses the main environment's torch. Native Python3.12.8,torch2.5.1. Use:

```bash
.venv-benchmark/bin/python -m pytest -q
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.paper_benchmark --batch-size 32
```

Use MPS for native Mac GPU inference. The existing loader supplies value-preserving
contiguous-convolution adapters and CPU interpolation fallback. Keep FAISS at one
thread: a prior final search deadlocked in mixed OpenMP runtimes after MPS.
`eval/score_saved_benchmark.py` can score completed arrays in a fresh CPU process
without re-encoding. Preserve caches and source/weight checksums.

Mapillary credentials are in ignored `swespy/.env`; never print or commit them.
Downloaded metadata includes signed image URLs; keep it ignored. Network/MPS and
Docker access may require sandbox escalation. Current gallery directory was281MiB;
Mac had about14GiB free after acquisition. Do not remove unrelated files for space.

## Interpretation guardrails

All-query recall includes uncovered queries as misses; report reference coverage
separately. GPS labels are provider-supplied. Same-provider Mapillary holdouts are
development evidence, not proof of performance on arbitrary user photos. Compare
gallery variants on identical queries/model/metric; central18 and older30 sets
are different. Show retrieved evidence; an LLM explanation alone is not validation.
