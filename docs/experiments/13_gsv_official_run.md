# Street View gap fill via the official APIs: first manual run (2026-11-10)

**Setup:**
- Route: Map Tiles API (zoom 3) and Street View Static API, with the owner's key from
  the Keychain.
- Billing: Google Cloud free trial.
- Rules: all earlier Street View rules apply. Descriptors only, no images on disk,
  "Google Street View, © Google", every pull logged, never training or display.
- The Mapillary/Commons/KartaView gallery is unchanged.
- One manual run; nothing is scheduled.

## Caps (enforced in `swespy/gsv.py` `Budget`)

Every billable request is written to `data/gsv/billable_journal.jsonl` *before* it is
sent, and all caps are counted from that journal, so a crash can only over-count.

| Cap | Limit | Used today | Remaining |
| --- | ---: | ---: | ---: |
| Tiles per Pacific day | 14,500 | **14,491** | 9 |
| Tiles in total | 220,000 | 14,491 | **205,509** |
| Static per billing month | 9,500 | **9,188** (50 comparison + 9,050 build + 88 evaluation) | 312 |

**Uncharged requests (logged, not counted):** 6 sessions, 83 panoIds (100 locations
each), 7,329 metadata.

## Part 1: smoke test and official comparison

- **Smoke test passed.** One session, one panoIds call for 5 gap points (5 panoramas
  found) and one metadata call, then 6 tiles for one view. The panorama was 16384×8192
  (4096×2048 at zoom 3); the view was rendered, passed the quality filter and was
  encoded. No image file was written.
- **Copyright format:** the Map Tiles API labels Google's own captures **"From the
  Owner, Photo by: Google"**, while the Static API uses "© 2026 Google". The official
  filter now accepts both; user photospheres ("Photo by: <name>") are still excluded.
- **Comparison, 50 pilot views against the Static API:**
  - same-view descriptor cosine: median 0.876 (5th percentile 0.79, min 0.58);
  - for scale, same panorama other heading 0.11, other panoramas 0.04;
  - every Static view retrieves its own pilot view first: **50/50**.
- **Official tiles:** 3 views rendered from official tiles gave cosine 1.0000 with the
  pilot's internal-tile views. The tiles are the same, and so is the rendering.
- **So the two routes see the same place but aren't interchangeable descriptors.**
  The likely cause is that Static levels the horizon (panorama tilt about 86° in the
  metadata), while our renderer ignores tilt. Every view is labelled with its route.

## Parts 2–3: build

**Discovery** (uncharged): the first 8,000 gap points in the owner's order, which are
all priority-2 residential streets. That gave 7,259 official panoramas and 7 rejected
(not Google-captured). Discovery stopped once the day's caps had enough work; it resumes
from point 8,000.

| Route | Panoramas | Views stored | Dropped by quality filter | Stopped |
| --- | ---: | ---: | ---: | --- |
| Tiles (zoom 3, 2 sideways views on roads) | 1,181 | 2,359 | 3 | daily tile cap |
| Static (640×640, FOV 90, same headings) | 4,525 | 9,040 | 10 | its share done |
| **Total** | **5,706** | **11,399** | 13 | |

- **No overlap:** each panorama is assigned to exactly one route, and the assignment is
  saved in `build_plan.json`, so the routes never fetch the same panorama, today or
  later. 1,552 discovered panoramas are not assigned yet.
- **Index:** `data/gsv/gsv_index/`: float16 descriptors, 188 MB. One record per view
  with source `gsv`, route, priority, way type, way ID, capture date and attribution.
  The checkpoint is `done.jsonl`, by panorama ID and heading.
- **Time and errors:** 52 minutes. Encoding (MegaLoc fp32 on the Mac GPU, about 4 views
  per second) was the bottleneck. 0 errors and no quota errors; 7 tile requests failed
  transiently and were retried within the budget.

## Part 4: evaluation (gallery alone vs gallery + Street View)

**Method:**
- Production rules: exact retrieval, the top 5 by cosine geometrically verified, then
  `rerank.finalize`.
- Configuration B merges the gallery's and the Street View index's candidates by cosine.
- Street View candidates are verified against the same view fetched live from Static
  (88 requests, in memory, no feature caching); 0 were left unverified.
- The gallery-only run on dev reproduces the frozen dev results exactly.
- **Correct** means within 500 m; **confident but wrong** means "ok" but more than 500 m off.

### Dev (66 photos)

| | Within 100 m | Within 500 m | Confident (correct) | Low (correct) | Confident but wrong |
| --- | --- | --- | --- | --- | ---: |
| Gallery | 27 (41%, CI 30–53) | 31 (47%, CI 35–59) | 24 (22) | 42 (9) | 2 |
| + Street View | 28 (42%, CI 31–54) | 32 (48%, CI 37–60) | 24 (23) | 42 (9) | 1 |

- **Within 500 m:** 4 photos fixed and 3 broken.
- **Source of the correct top matches:** Street View supplied 5 (2 from tiles, 3 from
  Static), Mapillary 16, Commons 11.

| Dev subgroup | n | Gallery, within 500 m | + Street View | Fixed / broken |
| --- | ---: | --- | --- | --- |
| Within 100 m of a new Street View reference | 25 | 9 (36%, CI 20–56) | **12** (48%, CI 30–67) | 4 / 1 |
| All others | 41 | 22 (54%, CI 39–68) | 20 (49%, CI 34–64) | 0 / **2** |

### Street-image set (48 Mapillary photos)

| | Within 100 m | Within 500 m | Confident (correct) | Confident but wrong |
| --- | --- | --- | --- | ---: |
| Gallery | 29 (60%) | 32 (67%, CI 53–78) | 24 (23) | 1 |
| + Street View | 30 (63%) | 33 (69%, CI 55–80) | 26 (24) | 2 |

- **Near new Street View** (n = 17): 14 → 15 within 500 m, 1 fixed. **Others** (n = 31):
  unchanged.

### Field photos

None saved yet (n = 0).

### Reading

- **Near new coverage, it helps:** dev +3 (4 fixed, 1 broken), street-image +1.
- **Elsewhere it hurts slightly:** 2 dev photos far from the new coverage were pulled to
  a wrong place by Street View references.
- **Net:** +1 on both sets.
- **Every confidence interval overlaps.** One day's build covers only part of the
  residential gaps, and the sets are small, so this is a direction, not a significant
  result.

## Part 5: field testing and continuation

**Top 10 streets by new panoramas** (`data/gsv/top_streets.json`; names from OSM):

1. Kvinnebyvägen (79)
2. Rydsvägen (61)
3. Vindarnas väg (50)
4. Lövsbergsvägen (40)
5. Atterboms gata (40)
6. Vårbruksgatan (39)
7. Passadvindsvägen (36)
8. Stallmästaregatan (34)
9. Kompanigatan (33)
10. Roshagsvägen (33)

**Continue the tile build** manually, after 09:00 Swedish time, when Google's daily
quota resets at midnight Pacific. It resumes discovery from the checkpoint, then fills
up to the day's tile cap:

```bash
cd ~/Documents/GitHub/heimdall-spy/swespy && caffeinate -i .venv-benchmark/bin/python -m eval.gsv_build discover && PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i .venv-benchmark/bin/python -m eval.gsv_build build --routes tiles
```

Check the caps at any time with
`.venv-benchmark/bin/python -m eval.gsv_build status`.
