# Recent ordinary-photo evaluation

The user authorized collecting and testing recent ordinary outdoor Linköping
photos, using free sources only. Flickr was explicitly declined. Use Wikimedia
Commons only; do not switch to paid APIs or substitute Mapillary test queries.

## Access issue resolved

The initial HTTP403 was caused by our incomplete client identification. The
user-agent now truthfully identifies `HeimdallSpyResearchBot`, supplies the real
project URL `https://github.com/brickphone/heimdall-spy` as a contact, and includes
its HTTP library version. Commons metadata and image downloads succeeded after
this fix. No browser impersonation, proxy rotation, paid service or API key was
used. See [Wikimedia's User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).

The earlier broad candidate set contained many event portraits, aircraft-event
photos and indoor exhibits. It remains an unfrozen acquisition proposal under
`data/public_eval/recent_linkoping_v1/`; it was never evaluated. The final street/
landmark selection uses `data/public_eval/recent_linkoping_streets_v1/`, reusing
metadata and image bytes where applicable. Scene filtering is fixed before any
predictions and recorded in the protocol. Photographs still require visual review
before the evaluation set is frozen.

## Fixed photo requirements

- **Capture date:2023-11-06 through2026-11-06 inclusive.** This implements the
  requested last2–3 years at the time of the task. Use actual `DateTimeOriginal`,
  requiring a complete calendar day. Upload, edit and transfer dates do not count.
- GPSLatitude/GPSLongitude camera labels inside the same city rectangle as the
  reference gallery:west15.50,south58.34,east15.75,north58.48. Do not geocode a
  landmark's name or substitute object coordinates for the camera position.
- JPEG photos with attribution and a supported Commons license link:CC BY,
  CC BY-SA or Creative Commons public-domain declarations.
- Ordinary outdoor street/landmark scenes. Metadata filtering removes obvious
  interiors, portraits, rehearsal/press-conference images and aerial/maps; an
  agent or human must visually review candidates before freezing the test set.
- Target75 photos, with at most200 selected downloads and2,500 discovered metadata
  IDs; at most15 accepted per photographer and at least25m between query labels.
  Fewer eligible photos must be reported without relaxing the date rule.
- Reject exact/perceptual duplicates against all3,388 reference images and other
  accepted queries. This is conservative screening, not proof of independence.

Reference photos stay separate from test queries. Query pixels are written to
new JPEGs without GPS/EXIF/XMP metadata; the answer key remains only in the
manifest. Camera labels and dates remain provider claims and are not guaranteed
correct. Commons landmarks and a few photographers may be overrepresented.

## Run or resume

From the repository root:

```bash
cd swespy
.venv-benchmark/bin/python -u -m eval.public_photos acquire
```

This requires no API key. It uses the existing benchmark environment. Completed
metadata batches and images are cached under ignored
`data/public_eval/recent_linkoping_streets_v1/`. Requests identify the client
accurately, use one connection at a time, and pause after expensive API responses.
If a future403 occurs, inspect its recorded diagnostic; do not evade the block or
interpret it as an empty eligible dataset.

On successful acquisition, inspect `review.html` before running any predictions.
The agent can perform this visual review; the user need not collect photos.
Write `review.json` with real reviewer/time and the accepted IDs, for example:

```json
{
  "reviewer": "name of the person or agent who inspected the images",
  "reviewed_at": "actual ISO timestamp",
  "include_ids": ["commons-actual-page-id"]
}
```

Then freeze the reviewed set and evaluate:

```bash
.venv-benchmark/bin/python -m eval.public_photos freeze
PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-benchmark/bin/python -u -m eval.public_photos evaluate
```

Freeze validates capture dates and hashes and refuses empty/unknown/repeated IDs.
The evaluation checks encoder/index compatibility, computes cached descriptors in
batches of4 on MPS, and searches the existing exported MegaLoc gallery. It writes
a separate timestamped report with all-query top1/top5 hits and reference coverage
at25m/100m/500m, plus attributed photo cards and a candidate map. Scores remain
uncalibrated.25m results are provisional because camera labels are unverified.

Unit checks currently pass75 tests with one private snapshot skipped,
including date boundaries, misleading upload dates, missing camera GPS, license
filtering, and fail-closed HTTP403 handling. Actual network acquisition, visual review, freezing and MPS inference have now completed; results follow.

Source API reference: [MediaWiki Imageinfo](https://www.mediawiki.org/wiki/API:Imageinfo).

## Completed ordinary-photo results

Reviewed all75 street/landmark candidates before inference and excluded9 indoor,
portrait, object-only or context-free vehicle close-ups. The frozen set contains
**66 outdoor photos from15 creators**. Actual capture dates range from
**2023-11-21 through2026-09-01**:one from2023,eleven from2024,twenty-one from2025,
andthirty-three from2026. Weak visual clues were not grounds for exclusion.

The unchanged pretrained MegaLoc model searched the existing3,388-reference city
index. All66 queries stayed in the denominator:

| Radius | Any reference nearby | First-choice hits | Top-five hits |
| --- | --- | --- | --- |
| 25m | 3/66 (4.55%) | 0/66 (0.00%) | 0/66 (0.00%) |
| 100m | 13/66 (19.70%) | 1/66 (1.52%) | 2/66 (3.03%) |
| 500m | 56/66 (84.85%) | 7/66 (10.61%) | 14/66 (21.21%) |

Median first-choice error: **3,643m**. These results are poor. The earlier61.1%
central Mapillary result does not transfer to this independent-provider ordinary-
photo sample. Limited25m/100m reference coverage explains some failures, but
56/66 queries have a reference within500m while only7 retrieve one first. Model/
image-domain mismatch and candidate ranking therefore also need investigation;
more photos alone are not established as the complete solution.

This is a small, curated Commons sample with repeated photographers, locations
and public art/landmarks. GPS and capture-date claims are unverified; provider
independence and duplicate screening do not prove complete pretraining independence.
Do not present these numbers as accuracy on every random Linköping photo.

- [Attributed query photos, predictions and candidate map](../swespy/data/public_eval/recent_linkoping_streets_v1/runs/20261106T135549400456Z/map.html).
- [Raw results](../swespy/data/public_eval/recent_linkoping_streets_v1/runs/20261106T135549400456Z/results.json).
- Frozen manifest, pre-inference review decisions and review sheets are retained
  under `swespy/data/public_eval/recent_linkoping_streets_v1/` (ignored by Git).

All66 query hashes and capture dates were checked, and all query JPEGs were
verified to contain no EXIF. The reference-gallery and exported-index fingerprints
were preserved. No training, paid service or production model switch occurred.
The next modelling decision should use these frozen queries unchanged; do not
train on them or remove failures after observing predictions.
