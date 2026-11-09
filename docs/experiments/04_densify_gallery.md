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

## Build results (RunPod A40, 2026-11-08)

Built on one RunPod A40 pod (EU-SE-1, $0.59/h; the RTX 3090 had no EU stock) with
`scripts/cloud_build.sh`. It ran 75.6 min from pod creation to termination and cost
≈ $0.75. Setup took 4 min 19 s; both stages ran 09:42–10:50 UTC. 96/96 chunks
finished with 0 errors and 0 download failures. Encoding ran at 66–95 img/s on the
A40, while download plus filtering ran at ~25–33 img/s and was the bottleneck. Two
mid-run fixes, each followed by a resumable restart, moved image resizing off the
asyncio event loop and raised download concurrency (24 parallel images, 8 Graph API
batches in flight).

The test manifest was never copied to the pod. The sealed pixel-hash guard ran on
the Mac (`--finalize-sealed`) and excluded **0** references in both indexes.
**`--verify 200` passed:** CUDA vs Mac MPS descriptors had minimum cosine 0.9999996.

| | `recent` | `historic` (recent + pre-2018) |
| --- | ---: | ---: |
| Input to quality filter | 36,588 | 97,326 |
| Dropped: near-duplicate | 3,575 | 4,046 |
| Dropped: blurry | 337 | 903 |
| Dropped: sky/ground only | 76 | 663 |
| **References in index** | **32,600** | **91,714** |
| Mapillary / Commons / KartaView | 30,558 / 1,710 / 332 | 89,672 / 1,710 / 332 |
| location_source: mapillary_computed / raw | 30,557 / 1 | 89,671 / 1 |
| location_source: commons_exif_low_precision / kartaview_gps | 1,710 / 332 | 1,710 / 332 |
| Pre-2018 references | 2,747 (panorama crops, Commons, KartaView) | 61,861 |
| Empty 250 m cells inside polygon | 1,807 / 2,519 | 1,326 / 2,519 |

Near-duplicates were checked on a sample of 200 dropped `recent` frames. 169 matched
a kept frame within 5 m, all by the **same photographer, from a different sequence,
captured < 1 minute apart**: the same capture split across Mapillary sequences.
Panorama crops survived (2,035 of 2,056 in `recent`).

### Dev set: distance from each dev photo to its nearest reference (66 photos)

| | Gallery v1 (polygon, 2,489) | v2 `recent` (32,600) | v2 `historic` (91,714) |
| --- | ---: | ---: | ---: |
| Median | 190 m | 47 m | **11 m** |
| 25th / 75th percentile | 120 / 336 m | 15 / 85 m | 4 / 36 m |
| ≤ 25 m | 3 | 22 | **43** |
| ≤ 50 m | 5 | 33 | **54** |
| ≤ 100 m | 13 | 50 | **63** |
| ≤ 500 m | 56 | 63 | **65** |

This measures coverage only, not accuracy. Nearby references may face another way
or show a different scene. Retrieval accuracy for each stage is evaluated in Part 6.
Coverage maps: `data/gallery_v2/coverage-recent.png` and `coverage-historic.png`
(not in git).

## Next

Walk-route proposals from the `historic` density grid (`python -m eval.walk_routes`),
then stop for the user to record them.
