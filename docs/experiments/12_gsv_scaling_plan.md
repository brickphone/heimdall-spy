# Street View: zoom check, full gap-build plan, official comparison, value check

Date: 2026-11-10. These are decisions for the owner; nothing has been built.

## 1. Zoom 2 vs zoom 3 (offline, Mac, no Google traffic)

**Method** (`eval/gsv_plan.py zoom-check`):
- **Queries:** the 66 dev photos.
- **References:** each photo's exact top-k from the full historic gallery (stored
  descriptors), re-encoded on the same device in two conditions:
  - **zoom 3:** the stored 640 px image, which is about the detail a zoom-3 view gives;
  - **zoom 2:** the same image downsampled (antialiased) to 512 px on its long side,
    which is what zoom 2 gives per 90° view. A 416 px variant matches zoom 2 on older
    3328 px panoramas.
- **Encoding:** MegaLoc resizes every input to 322×322.
- **Compared:** descriptors (cosine) and top-5 retrieval.

**Results:**
- **First run** (depth 100, 512 px then enlarged back to 640, i.e. a 640 px render of
  512 px detail):
  - mean descriptor cosine 0.980, minimum 0.819;
  - top-1 identical 58/66, **top-5 set identical only 30/66**;
  - so a zoom-2 render enlarged to 640 is not equivalent.
- **Second run** (depth 30, downsampled only, rendered at 512 / 416 px, 1,812 references):

| | 512 px (zoom 2, 4096 px panoramas) | 416 px (zoom 2, older 3328 px panoramas) |
| --- | ---: | ---: |
| Descriptor cosine vs zoom 3: mean / 1st percentile / min | 0.985 / 0.930 / 0.866 | 0.965 / 0.870 / 0.763 |
| Top-1 identical | 58 / 66 | 53 / 66 |
| Top-5 set identical | **34 / 66** | **23 / 66** |
| Top-5 order identical | 19 / 66 | 11 / 66 |

  Re-encoding on the Mac reproduces the stored descriptors (cosine ≥ 0.9999996), so the
  differences come from the resolution, not the device.

**Result: zoom 2 changes top-5 retrieval** for 32 of 66 dev photos (and the top-1 for
8). The check therefore did not pass. As instructed, **no decision is taken on zoom 2**;
zoom 3 remains the current default, and a full-depth pod run is proposed for the owner.

- **Depth doesn't matter here:** these changes are already within each photo's top 30,
  so checking deeper (top 100) could only add changes, never remove them.
- **The confidence labels were not computed:** that would need re-ranking against the
  downsampled references, and the rule already triggered on top-5.
- **The open question is impact:** do these swaps change accuracy (within 100/500 m)
  or the confident/low label? A full-depth GPU run that also re-ranks would answer it
  (proposed in the report).

## 2. Full gap build: estimates

**Inputs:**
- The Part 1 gap plan's metadata queue: 103,427 points (priority 1: 36,369;
  priority 2: 40,790; priority 3: 26,268).
- The pilot's measured coverage per way type. Examples: residential 47/51, service
  70/141, path 42/131, footway 32/66, track 5/46, unclassified 13/37, tertiary 9/10.
- Way types with fewer than 20 pilot points: main roads 0.9, foot-like ways at the
  footway rate.
- Priority 3 (old-only) was not in the pilot; it uses the same per-way-type rates.

**Unique panoramas** (lower bound: points within 15 m share a panorama; upper bound:
every covered point gives its own):

| | Priority 1 | Priority 2 | Priority 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| Panoramas | 9,900–12,300 | 16,500–24,300 | 8,500–13,700 | **34,900–50,300** |

**Tiles per panorama** (exact, by simulating the views over random orientations;
16% of pilot panoramas were the older 3328 px kind):

| | 4 headings (priorities 1–2) | 2 sideways (priority 3) |
| --- | ---: | ---: |
| Zoom 3 | 16.8 | 12.7 |
| Zoom 2 | 8.0 | 7.9 |

**Total tiles:** zoom 3 **551k–789k**; zoom 2 **278k–401k**. Until the zoom decision,
**zoom 3 is the planning default**; the zoom-2 rows show what the open question is worth.

### (a) Internal route

- **Discovery:** 4,420 area requests (zoom-17 map tiles, 160 m × 160 m) list every
  current panorama, so there are no point searches.
- **Per panorama:** 1 metadata request (image size, date, copyright, source) plus its tiles.
- **Pace:** 1 request per second, one in flight, measured at 1.07 s per request.

| | Requests | Hours | Pod cost ($0.164/h) |
| --- | ---: | ---: | ---: |
| Zoom 2 | 317k–456k | 94–136 h (3.9–5.7 days) | $15–22 |
| Zoom 3 | 590k–844k | 175–251 h (7.3–10.5 days) | $29–41 |

**Risk:** the pilot stopped at 4,337 requests without any throttling. These runs are
70–190× longer, and nothing is known about Google's limits at that volume. A block
mid-run stops it; the rules allow no retries or evasion.

### (b) Official Map Tiles API

- **Discovery:** the `panoIds` endpoint takes up to 100 locations per request, so about
  1,035 requests. These aren't billed.
- **Per panorama:** Street View metadata (not billed) for image size, heading, date and
  copyright.
- **Tiles:** only the ones each view needs. These are billed after 100,000 free per month.
- **Rate limit:** 6,000 tile requests per minute.

| | Tiles | Months at 95,000 free tiles/month | Months at 100,000 | Fetch time at 6,000/min |
| --- | ---: | ---: | ---: | ---: |
| Zoom 2 | 278k–401k | **2.9–4.2** | 2.8–4.0 | 46–67 min in total (about 16 min per monthly batch) |
| Zoom 3 | 551k–789k | **5.8–8.3** | 5.5–7.9 | 92–132 min in total |

- **Encoding:** a monthly batch is about 12,000 panoramas and 40,000 views at zoom 2,
  or 2–3 hours on the Mac. No pod is needed.
- **Paid alternative:** everything at once would cost about $360–600 at zoom 2, at $2
  per 1,000 tiles beyond the free 100,000. That needs explicit approval.

### Recommendation

**Use the official Map Tiles API, within the free tiles each month, in priority order.**
- **Speed is no reason for the internal route.** The official route's monthly batch
  fetches in minutes, whereas the internal route would need a pod running for 4–10 days.
- **The official route has no block risk.**
- **It's the one with an API agreement**, with the indexing risk you already accepted.
- **It's free:** within the 100,000 free tiles a month, the whole build costs nothing.
- **It suits a staged build:** month 1 can be the value check (section 4); continue only
  if that shows value.

A combination (official within the free tiles, internal for the rest) is possible. I'd
only use it if a deadline makes 3–8 months too slow, and then for priority 3 last.

**Conditions:**
- the official tile cap is still 0 in code until you approve it (95,000/month proposed);
- the EEA "no use with any map" term still applies, and nothing is displayed.

## 3. Tile-rendered vs official views

**Blocked:** no Google Maps key is in the Keychain. There's no item for service
`google_maps_key`, nor any Google- or Maps-related item (checked by name only), and
`GOOGLE_MAPS_API_KEY` isn't set.

The script is ready (`eval/gsv_internal_compare.py`):
- 50 pilot views (one per panorama, seeded);
- the free metadata check, then 50 Static API views reserved against the 9,500 cap;
- encoding on the CPU in memory;
- outputs: cosine for the same view, other headings of the same panorama for scale, and
  whether each official view retrieves its own pilot view first among all 912.

## 4. Value check before scaling: proposal

**Goal:** find out whether a Street View gap index improves accuracy without choosing
where to build by dev, field or test photo locations.

**Design:**
1. **Areas:** the 10 areas from the pilot (k-means over gap points only, seed 20261110),
   with every gap point assigned to its nearest centre. Area membership is the Voronoi
   cell of the centres, clipped to the polygon. The build size per area is known from the
   gap plan alone (table below).
2. **Pre-registered draw:** a fixed seed, written down before anything is built, puts
   the 10 areas in random order. Build in that order until the next area would exceed
   one month's free tiles (95,000 at zoom 2). Expect 1–2 areas, about 10–25% of all gap
   points.
3. **Build:** all priorities inside the drawn area(s), on the official route, within
   the free tiles.
4. **Evaluate once,** with everything frozen beforehand (metrics, thresholds,
   configurations):
   - all 66 dev photos and any saved field photos (currently 0) in two configurations:
     the gallery only, and the gallery plus the Street View index (merged candidates,
     source kept);
   - **primary:** within 500 m and within 100 m, for photos inside vs outside the built
     area(s), paired per photo (exact McNemar inside);
   - **secondary:** confident-but-wrong counts, the ok/low split, and which source gave
     the top match;
   - **the outside group is the control.** Its accuracy should not change; a drop there
     means Street View references are pulling photos to wrong places.
   - **Go criterion** (to be fixed before evaluating): improvement inside, and no loss
     outside beyond noise.
5. **Limits stated in advance:**
   - The dev photos are street photos of places that may already be covered, so the
     number inside the drawn areas may be small and the confidence intervals wide.
     That is a real answer about value for typical photos, not a flaw to fix by
     choosing areas.
   - After the evaluation, the share of dev photos standing in gap points can be
     reported descriptively.
   - The test set is not touched.

**Cost per area** (official route, zoom 2; upper estimate of panoramas, 8 tiles each):

| Area | Gap points | Panoramas (upper) | Tiles (zoom 2) |
| ---: | ---: | ---: | ---: |
| 0 | 10,423 | 4,797 | 38k |
| 1 | 12,002 | 5,474 | 44k |
| 2 | 8,449 | 3,756 | 30k |
| 3 | 7,474 | 3,796 | 30k |
| 4 | 9,199 | 4,372 | 35k |
| 5 | 8,709 | 4,024 | 32k |
| 6 | 15,697 | 8,320 | 67k |
| 7 | 7,062 | 3,651 | 29k |
| 8 | 8,101 | 3,810 | 30k |
| 9 | 16,311 | 7,949 | 64k |

- **Any one area fits in a single month of free tiles:** 29k–67k tiles, about twice
  that at zoom 3.
- **Discovery and metadata are free.**
- **Encoding** is about 15,000–33,000 views, 1–2 hours on the Mac.
- **The same design works on the internal route:** about 33,000–75,000 requests per
  area at zoom 2, or 10–22 pod hours.
