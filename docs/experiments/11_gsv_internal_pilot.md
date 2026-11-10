# Street View via the unofficial backend (`streetlevel`): feasibility pilot

Decision (recorded): using the unofficial Google Street View backend via the open-source `streetlevel` library is approved for a pilot. Risk and responsibility accepted by Mattis Hüttenrauch on 2026-11-10.

All rules of the official Street View index (`docs/experiments/10_gsv_index.md`) still apply:
- descriptors only, with panoramas and views in memory only;
- always "Google Street View, © Google", never a licence;
- every pull logged;
- retrieval index only: never training, fine-tuning or pair mining;
- never shown in the demo.

This is a feasibility pilot, not a build. Nothing from it is merged into the demo or the evaluation.

Library: `streetlevel` 0.12.13 (MIT), pinned in a separate environment for this pilot only.

## Part 1: pilot set (Mac, 2026-11-10)

Source: the Part 1 gap plan (`data/gsv/sample_points.json`), using the deduplicated
metadata queue (one point per 10 m cell).

**Selection:** seeded k-means (seed 20261110) splits all priority-1 and priority-2 gap
points into 10 disjoint areas. In each area, 25 priority-1 and 25 priority-2 points
are drawn at random. Points were not chosen by proximity to dev, field or test photos;
those files were not read.

| | Priority 1 (empty, non-road) | Priority 2 (empty, road) |
| --- | ---: | ---: |
| Points | 250 | 250 |
| Areas | 10 × 25 | 10 × 25 |

| Way type | Points |
| --- | ---: |
| service | 141 |
| path | 131 |
| footway | 66 |
| residential | 51 |
| track | 46 |
| unclassified | 37 |
| tertiary | 10 |
| living_street | 9 |
| bridleway, pedestrian, steps, secondary | 3, 2, 2, 2 |

**Smallest spacing:** 3.5 m. Points that close will usually reach the same panorama,
which is deduplicated by pano ID.

**Files:**
- `data/gsv_internal/pilot_points.csv`;
- the map, `data/gsv_internal/pilot_map.html`.

Both are git-ignored.

## Part 2: fetch run (pod, 2026-11-10)

**Setup:**
- One RunPod CPU pod: cpu3g, 4 vCPU, 16 GB, in EU-RO-1 (Romania; the Sweden regions had
  no CPU stock). $0.164/h.
- No API key, and no secret passed to the pod.
- `streetlevel` 0.12.13 only in `.venv-gsv-internal`.
- All 30 mocked tests passed on the pod before fetching started.

**Tiles:**
- **Fetched:** only the tile rows a level 90° view needs, through the library's own tile
  URL builder, one at a time, with the library's default tile headers.
- **Zoom 3:** 16 tiles for 4096×2048 panoramas (rows 1–2 of 4), and 21 for older
  3328×1664 panoramas (rows 0–2).
- **Kept:** official captures only: not third-party, copyright "© Google", source
  `launch` or `scout`.

**Rate limits:**
- One request in flight; the measured maximum was 1.
- At least 1 s from the end of one request to the next; the measured minimum was 1.000 s.
- The circuit breaker was armed for 403, 429, consent or CAPTCHA pages, and 3
  consecutive errors. **It never fired.**

| | |
| --- | --- |
| Result | **Completed: all 500 points, no throttling, no block** |
| Requests | 4,337 (500 searches + 3,837 tiles); 4,336 successful (99.98%) |
| Errors | 1 × HTTP 503 on a tile (that pano was skipped; no retry) |
| Fetch time | 1.29 h, at 56 requests per minute |
| Panoramas | 228 unique: 177 per hour, 19.0 requests per panorama (search + tiles) |
| Views | 912 (4 per panorama, FOV 90, 640×640, pitch 0). All passed the quality filters. float16 descriptors, 15 MB |
| Encoder | MegaLoc fp32 on CPU, run concurrently. Its queue never held more than 1 panorama, so it never delayed fetching |

### Coverage (official Google panorama within 25 m)

| | Points | With a panorama |
| --- | ---: | ---: |
| Priority 1 (empty, non-road) | 250 | 81 (32%) |
| Priority 2 (empty, road) | 250 | 148 (59%) |

| Way type | Points | With a panorama |
| --- | ---: | ---: |
| **path** | 131 | **42 (32%)** |
| footway | 66 | 32 (48%) |
| track | 46 | 5 (11%) |
| service | 141 | 70 (50%) |
| residential | 51 | 47 (92%) |
| unclassified | 37 | 13 (35%) |
| tertiary | 10 | 9 |
| living_street | 9 | 7 |
| others (bridleway, pedestrian, steps, secondary) | 9 | 4 |

- **Priority-1 paths** have official coverage at about a third of points: mostly
  trekker or car imagery along paths that run next to roads. Forest tracks have almost none.
- **Residential streets** missing from the gallery are nearly all covered (92%).

### Capture dates and resolution

| Capture year | Panoramas |
| --- | ---: |
| 2026 | 124 |
| 2022 | 49 |
| 2024 | 12 |
| 2011 | 19 |
| 2019 | 9 |
| 2009, 2017, 2021, 2023, 2025 | 3, 2, 4, 4, 2 |

- **Recent:** 54% of the panoramas are from 2026, so Google has driven Linköping this
  year. That's newer than most of the gallery, where 67% predates 2018.
- **Resolution at zoom 3:** 192 panoramas at 4096×2048 and 36 older ones at 3328×1664,
  giving 1,024 and 832 px per 90° view.
- **Older captures:** 152 of 228 panoramas have older captures available (403 in total).
  Per the rules these were counted from the metadata only, not fetched.

## Part 3: comparison with the official API

**Pending:** waiting for the API key. 50 views, counted against the official monthly cap.

## Part 4: pull log

`data/gsv_internal/pull_log.jsonl`: 101 rows, one per 10 points and request type, with:
- UTC time, bounding box, request type, request count, success count, cumulative count;
- machine (`runpod:fac721047965`);
- stop reason (`completed` on the last rows).

It contains no URLs, keys or image data (checked).

## Part 5: disk, cost, cleanup

- **Disk scan on the pod**, just before copy-back: 240 files newer than the
  fetch-start marker, **0 image files** (by extension or magic bytes, whole container
  filesystem). The Mac copy also has no image files.
- **Copied back:** only the descriptor index (`gsv_internal_pilot/`: float16
  descriptors, `views.jsonl`, `points.jsonl`, `summary.json`), the pull log, and the run
  and setup logs.
- **Pod:**
  - created 08:14:33 UTC; terminated by the Mac watcher 09:35:55 UTC, 21 seconds after
    the pilot finished;
  - the API then reported **no pods**;
  - **cost $0.22** (1.36 h at $0.164/h).
- **Not merged:** the pilot index is not in the demo, the evaluation or the gallery.

## Why it was slow, and how to speed it up

**Where the time went:**
- **The pause, not the network.** 4,337 requests in 1.29 h is 1.07 s per request. The
  mandatory pause was 1.0 s of that and the request itself about 0.07 s, so **93% of
  the run was the politeness pause.** The pod's speed was irrelevant: encoding a
  panorama took under a second, and its queue never held more than one.
- **Tiles dominate:** each panorama needed 16–21 tile requests, about 20 s, at one per second.
- **Empty searches:** 271 of 500 searches (54%) found no panorama. That's 271 requests,
  about 5 minutes, spent learning that nothing was there.

So speed is set by requests per panorama, not by the computer. Three ways to cut
requests without changing the politeness rules:

| Change | Effect | Note |
| --- | --- | --- |
| **Zoom 2 instead of zoom 3** | 8 tiles per panorama instead of 16–21: about 2× fewer requests | MegaLoc resizes every view to 322×322 anyway, and zoom 2 still gives 512 px per 90° view. This breaks the "~600 px" rule, so it needs your approval, plus an offline check that zoom-2 and zoom-3 descriptors agree |
| **Coverage tiles instead of point searches** | One request lists every panorama in a ~600 m × 600 m map tile; replaces the per-point searches, including the 54% that found nothing | Supported by `streetlevel` (`get_coverage_tile`); also deduplicates up front |
| **Fewer headings where they add little** | E.g. 2 sideways views on roads, as in the official plan's priority 3: fewer tile columns per panorama | A coverage trade-off |

With zoom 2 and coverage tiles, this pilot would have needed about 2,000 requests
instead of 4,337: about 35 minutes instead of 77, at the same politeness.

**Not done, and not recommended without an explicit decision:**
- **Shorter pauses or parallel requests:** that trades directly against block risk.
- **More pods or IP addresses:** your rules forbid it as evasion.

**The legitimate fast route is the official Map Tiles API:** up to 6,000 tile requests a
minute, about 100× this pilot's rate, and 100,000 free tiles a month.
