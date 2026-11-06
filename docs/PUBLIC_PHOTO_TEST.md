# Recent ordinary-photo evaluation — acquisition blocked

The user authorized collecting and testing recent ordinary outdoor Linköping
photos, using free sources only. Flickr was explicitly declined. Use Wikimedia
Commons only; do not switch to paid APIs or substitute Mapillary test queries.

## Current result

**No new photos were downloaded or evaluated.** Wikimedia returned HTTP403 with
"Please respect our robot policy" for its Commons Action API, public REST API,
file pages and the public media host from this environment. Cached search results
show recent Linköping photographs exist, but those snippets are not downloaded
image pixels and cannot be used to produce a valid image benchmark.

The reproducible collector is `swespy/eval/public_photos.py`. Its live run exited2
and wrote an explicit blocked status, with no frozen test manifest or fabricated
accuracy result. It does not retry403, change identities or route through mirrors
to evade the restriction. There was no paid spend, training, gallery modification
or production API model change.

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

## Resume when Commons access is available

From the repository root:

```bash
cd swespy
.venv-benchmark/bin/python -u -m eval.public_photos acquire
```

This requires no API key. It uses the existing benchmark environment. Completed
metadata batches and images are cached under ignored
`data/public_eval/recent_linkoping_v1/`. The current `acquisition-status.json`
records HTTP403. A blocked request must not be treated as zero eligible images.
Run from an ordinary network where Commons permits these requests; repeated403
means access still needs resolving, not that date criteria should be broadened.

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

The acquisition needs real source access before end-to-end performance can be
validated. Unit checks currently pass74 tests with one private snapshot skipped,
including date boundaries, misleading upload dates, missing camera GPS, license
filtering, and fail-closed HTTP403 handling. No ordinary-photo accuracy is known yet.

Source API reference: [MediaWiki Imageinfo](https://www.mediawiki.org/wiki/API:Imageinfo).
