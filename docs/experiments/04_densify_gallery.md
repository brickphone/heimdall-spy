# Part 4 — Densify the reference gallery (interim: planned, not yet downloaded)

Date: 2026-11-07. Requirements: [`GALLERY_V2_SPEC.md`](../GALLERY_V2_SPEC.md).
Status: metadata fetched and plan frozen. Download, quality filtering, encoding,
coverage map and the dev `nearest_reference_m` distribution follow after the
overnight build (`swespy/scripts/build_gallery_v2.sh`).

## Sources and metadata (all inside the approved polygon)

| Source | Metadata | Accepted into plan |
| --- | --- | --- |
| Mapillary (vector tiles z14 + Graph API) | 184,560 images inside the polygon | 93,457 after thinning (257 panoramas) |
| Wikimedia Commons (700 m geosearch grid, saturated cells split) | 2,610 file pages | 2,123 pass metadata rules → 1,736 after the leakage guards |
| KartaView (500 m nearby search, 700 m grid) | 343 photos | 334 |
| Self-collected walks | — | 0 (routes are proposed after the coverage map) |

Commons metadata decisions: 170 indoor (keyword), 135 non-commercial or no-derivatives
licence, 162 without a capture time, 16 not JPEG, 4 outside the polygon.

### Mapillary location source

`computed_geometry` for 184,559 images and raw `geometry` for 1.

### Mapillary thinning (10 m and ±20° within a sequence; panoramas always kept)

The 16,530 images in the 39 earlier street-eval query sequences were excluded first.

| Year | Before | After |
| --- | ---: | ---: |
| 2014 | 3,380 | 2,918 |
| 2015 | 101,701 | 46,014 |
| 2016 | 10,896 | 9,740 |
| 2017 | 4,022 | 2,302 |
| 2018 | 8,276 | 3,967 |
| 2019 | 18,276 | 11,602 |
| 2020 | 8,969 | 6,510 |
| 2021 | 7,487 | 6,555 |
| 2022 | 445 | 420 |
| 2023 | 1,028 | 971 |
| 2024 | 224 | 211 |
| 2025 | 1,041 | 988 |
| 2026 | 2,285 | 1,259 |
| **Total** | **168,030** | **93,457** |

60,974 of the 93,457 (65%) are pre-2018, so the Part 6 pre-2018 ablation matters.
The originally agreed 3 m rule kept 152,966 frames, which didn't fit on disk (about
13 GB plus about 12 h of encoding). The user chose 10 m and storage at 640 px.

## Leakage guard (before download)

| Guard | Excluded | Eval photo itself | ≤50 m and ≤30 min | Same photographer + day |
| --- | ---: | ---: | ---: | ---: |
| Dev (66 photos, visible; list in `data/gallery_v2/leakage-dev.json`) | 379 | 30 | 20 | 329 |
| Test (sealed, counts only) | 8 | 5 | 0 | 3 |

All exclusions were Commons files. A check after the plan confirmed that 0 dev photos
and 0 test photos (count only) are in the plan. The filter stage re-runs both guards
on pixel hashes after download.

**Planned:** 95,527 references: 93,457 Mapillary, 1,736 Commons, 334 KartaView.
By location source: 93,456 `mapillary_computed`, 1 `mapillary_raw`, 1,736
`commons_exif_low_precision`, 334 `kartaview_gps`. With 8 crops per panorama, about
97,300 references reach the filter stage.

## Quality filters (re-run on the existing 3,388 references at 640 px)

| Rule | Rejected | Notes |
| --- | ---: | --- |
| blurry (Crete et al. 2007 blur metric ≥ 0.47) | 30 | Low-light and motion-smeared frames; sharp low-detail rural views are now kept |
| sky_or_ground_only (edge share < 0.02) | 24 | Unchanged rule |
| near_duplicate (same image: identical bytes, or hash match ≤5 m, or cross-source ≤30 m) | 99 | Consecutive frames about 5 m and 0.3 s apart |
| kept | 3,235 | |

The old Laplacian rule rejected 35 images, mostly sharp low-detail rural roads. A
hash-only near-duplicate rule flagged 401, of which 389 were consecutive frames of one
drive a median 20 m apart, so it was narrowed. Contact sheet with 30 random rejections
per rule: [`quality_filter_review.html`](quality_filter_review.html).

## Pending (after the overnight build)

Download failures, panorama crop count, per-rule quality counts on the new images,
final gallery size, coverage heatmap (`data/gallery_v2/coverage.png`), empty
250 m cells, and the dev `nearest_reference_m` distribution (gallery v1 vs v2). After
that come the 3–5 walk-route proposals (`python -m eval.walk_routes`).
