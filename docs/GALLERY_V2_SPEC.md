# Reference gallery v2 — requirements (Part 4)

Agreed with the user on 2026-11-07. The Part 4 code (`swespy/eval/densify.py`,
`swespy/eval/leakage.py`, `swespy/eval/walks.py`, `swespy/eval/provenance.py`)
implements these; deviations must be documented in
`docs/experiments/04_densify_gallery.md`.

## Area and fairness

- Only references inside `LINKOPING_AREA` (`swespy/swespy/config.py`).
- Densify **uniformly** across the polygon. Never prioritise or add extra fetching
  near dev (or test) photo locations.

## Every reference carries

| Field | Rule |
| --- | --- |
| `source`, `source_id`, `license_url` | See `docs/LICENSES.md` (commercial allowlist) |
| `lat`, `lng` | Required |
| `location_source` | Required; one of the values below |
| `heading` | Key required; `null` when unknown |
| `captured_at` | Capture time (ms since epoch, UTC) |
| `capture_year` | Required (used for the Part 6 pre-2018 ablation) |

### Coordinates by source

- **Mapillary:** use `computed_geometry` when present, otherwise raw `geometry`;
  `location_source` = `mapillary_computed` or `mapillary_raw`.
- **Self-collected frames:** interpolate the position from the GPX track by
  timestamp (`location_source` = `gpx_interpolated`). Reject a frame if it is more
  than 5 s from the nearest GPX point, or if GPS accuracy is worse than 15 m.
- **Wikimedia Commons:** keep the file's coordinates but tag them as lower precision
  (`location_source` = `commons_exif_low_precision`).
- Part 4 reports the count per `location_source`.

## Mapillary acquisition

- Fetch **all** metadata inside the polygon through the official API/vector tiles.
- Before downloading, thin each sequence: drop a frame within **10 m and ±20°**
  heading of an already-kept frame of the same sequence (capture order). Report
  counts before and after, per capture year. *Decision 2026-11-07: the original ~3 m
  rule kept 152,966 frames (~13 GB + ~12 h encoding), which did not fit the 12 GB
  free disk; 10 m was chosen over 5 m and 15 m.*
- Store downloaded images at **640 px** on the longest side (JPEG quality 88, EXIF
  orientation applied, metadata dropped). Panorama sources stay at 2048 px until
  cropped; crops are 640 px. MegaLoc itself resizes to 322 px.
- All parameters are written as `build_parameters` into `plan-report.json`,
  `filter-report.json` and `gallery-index.json` (`eval/densify.py:build_parameters`).
- Keep all panoramas; split each into perspective crops (separate references with
  derived headings).
- Download and encoding are resumable and checkpointed (run overnight); respect API
  rate limits with backoff.

## Quality filters (`swespy/eval/quality.py`)

- **blurry** — motion/defocus blur only: Crete et al. (2007) blur metric, worse of
  horizontal/vertical, threshold 0.47 (originals <= 0.43 at the 99th percentile;
  Gaussian or motion-blurred copies >= 0.50 at the 1st). Replaced a Laplacian-variance
  rule that rejected sharp low-detail rural road views.
- **sky_or_ground_only** — share of pixels with gradient > 20 below 0.02.
- **near_duplicate** — the same image stored twice: identical bytes within 30 m, or a
  perceptual-hash match within 5 m, or within 30 m across sources. Not used for
  spatial redundancy (thinning does that).
- **indoor** — Commons keyword rule only (no image classifier).
- Review sheet: `docs/experiments/quality_filter_review.html`.

## Leakage guard

- Exclude any reference within 50 m **and** 30 minutes of an eval photo's capture
  time, and any image by the same photographer as an eval photo in the same session.
- **Test set:** a sealed filter reads the test metadata internally and outputs only
  counts (how many references excluded, by which rule). It never prints, logs or
  saves which references or locations matched.
- **Dev set:** the same guard, with normal visible reporting.

## Self-collected walks

After the Part 4 coverage map exists, propose 3–5 walk routes that fill the largest
coverage gaps across the polygon, chosen by reference density only (not by where
eval photos are). Then stop so the user can record them.
