# Google Street View gap-filling index (`gsv_v1`)

> **Decision (recorded):** Google Street View is approved for indexing, risk accepted
> by Mattis Hüttenrauch on 2026-11-09.

Mandatory rules that follow from it (enforced in `swespy/gsv.py`):
- descriptors only, and no image on disk at any point;
- never labelled with a CC or other licence; always "Google Street View, © Google";
- every pull logged;
- retrieval index only: no training, fine-tuning or pair mining;
- at most 9,500 Static image requests per billing month without explicit approval.

Scope: fill gaps in the existing historic gallery (91,714 references). Street View is a
second index searched alongside it.

## Part 1: gap map and plan (metadata only)

Status 2026-11-09:
- **Done:** steps 1, 2, 4, 5, 7 and 8.
- **Blocked:** step 3 (metadata) is waiting for the API key, so steps 6 and 9 are only
  partly done.
- **No Google request has been made yet.**

### 1. Sample points (OSM)

- **Data:** Geofabrik Sweden extract `sweden-261108.osm.pbf` (MD5 verified), clipped to
  the polygon's bounding box → `data/osm/linkoping-ways.json`: 30,252 ways, ODbL. The
  954 MB extract was deleted after clipping; its SHA-256 is recorded in the output.
- **Ways included:**
  - roads: motorway to residential, living_street, service, links;
  - non-road: pedestrian, footway, cycleway, path, track, steps, bridleway.
- **Ways excluded:** `area=yes` outlines.
- **Sampling:** every 20 m along each way; only points inside the Linköping polygon are kept.

### 2. Classification against the historic gallery

A point is:
- **empty** if no reference is within 25 m;
- **old-only** if every reference within 25 m was captured before 2018;
- **covered** otherwise.

A reference with an unknown year would count as covering the point, so budget is never
spent on places that might already be covered. All 91,714 references have a year.

Checks:
- 300 random points agree with a brute-force search;
- 2,000 random references all classify their own location as non-empty.

| Class | Sample points | Share |
| --- | ---: | ---: |
| empty | 83,922 | 61% |
| old-only | 31,017 | 23% |
| covered | 22,853 | 17% |
| **total** (2,756 km of ways) | **137,792** | |

Why so much is empty or old:
- 67% of the gallery (61,861 references) is from before 2018.
- The empty length is mostly paths (21,840 points), service roads (22,118: parking
  aisles, driveways, yards), footways (11,679), residential streets (10,311) and tracks
  (5,397).
- Main roads are covered: motorway 100%, primary 99.8%.

<details><summary>Per way type</summary>

| Way type | empty | old-only | covered |
| --- | ---: | ---: | ---: |
| path | 21,840 | 11,229 | 5,539 |
| service | 22,118 | 5,184 | 2,397 |
| footway | 11,679 | 3,906 | 1,985 |
| residential | 10,311 | 3,486 | 1,017 |
| unclassified | 7,302 | 1,670 | 667 |
| track | 5,397 | 252 | 270 |
| living_street | 2,454 | 335 | 33 |
| tertiary | 1,755 | 2,601 | 1,626 |
| secondary | 250 | 1,564 | 4,785 |
| bridleway | 323 | 159 | 3 |
| steps | 215 | 186 | 119 |
| cycleway | 132 | 136 | 180 |
| pedestrian | 106 | 201 | 64 |
| motorway_link | 40 | 75 | 525 |
| primary (+link) | 0 | 6 | 1,692 |
| motorway | 0 | 0 | 1,871 |
| secondary/tertiary links | 0 | 27 | 80 |

</details>

### 3. Metadata queue (ready, not run)

- **Points to query:** one metadata call per 10 m cell and priority, which removes
  sidewalk/road twins. **103,427 calls:**
  - priority 1 (empty, non-road): 36,369;
  - priority 2 (empty, road): 40,790;
  - priority 3 (old-only): 26,268.
- **Request settings:** `radius=25`, `source=outdoor`.
- **Kept panos:** Google-captured only (copyright `© Google`, with or without a year);
  dedup by `pano_id`. A pano reached from several points keeps the best priority and
  that way's bearing.
- **Cost and logging:** metadata is free and has no published monthly cap. Each batch of
  500 is logged.
- **Covered points never reach Google:** this is asserted in the code and tested with a
  mocked API.

### 4–5. Priorities and headings

- **Order:** priority 1 = empty non-road, 2 = empty road, 3 = old-only.
- **Headings:**
  - priorities 1–2: way bearing + 0/90/180/270°;
  - priority 3: way bearing + 90/270°;
  - 0/90/180/270° if the bearing is unknown.
- **View settings:** FOV 90, 640×640, pitch 0.

### 6. Pano counts, capture dates, image requests, months

**Pending:** these need the metadata run.

The image requests are bounded by roughly 3–4 × the number of unique panos. Unless most
gap points have no Google pano, the plan will need many months at 9,500/month. More than
50,000 image requests does not make RunPod useful here: the bottleneck is the monthly
cap, not compute.

### 7. Static API vs Map Tiles API (Street View tiles)

Sources: Google Maps Platform pricing, the Map Tiles API Street View documentation,
policies and usage pages, and the EEA adjustment pages (fetched 2026-11-09).

| | Street View Static API | Map Tiles API, Street View tiles |
| --- | --- | --- |
| Free per month | 10,000 requests | **100,000 tiles** |
| Price after that | $7.00 / 1,000 | $2.00 / 1,000 |
| Free requests | metadata | session token, metadata, panoIds |
| Output | rendered perspective view, up to 640×640 | 512×512 equirectangular tiles, zoom 0–5; we render the view ourselves |
| Requests per view (FOV 90, ≥600 px) | 1 | 4–6 at zoom 3 (see below) |
| **Free usable views per month** | **10,000** (9,500 cap) | **≈16,700–25,000** |
| Resolution of a 90° view | 640 px | ≈1,024 px at zoom 3 |
| Rate limit | — | 6,000 tile requests per minute |

- **Tiles per view:** at zoom *z* a tile spans 360°/2^*z* horizontally (documented:
  zoom 0 = 360°, zoom 5 = 11.25°).
  - Zoom 2 gives only 512 px per 90°, too small; zoom 3 gives 1,024 px.
  - A pitch-0, FOV-90 square view spans ±45° elevation, which is 2 tile rows, and 2–3
    tile columns.
  - Four headings at 90° steps cover the whole ring, 16 tiles, so **4 tiles per view**
    for priorities 1–2. Two sideways headings take 8–12 tiles, so 4–6 per view.
  - **Uncertainty:** the documented maximum pano size (13,312 px wide) implies about 6.5
    columns rather than 8 at zoom 3. The pilot should confirm the grid with a handful
    of tiles.
- **Thumbnail endpoint:** at most 600×250, so it can't produce the same 640×640 FOV-90
  view as the gallery crops. Not recommended for the index.
- **Terms:**
  - Both APIs forbid pre-fetching, indexing, storing or caching content; the Tiles
    policy also explicitly says "don't … bulk download map tiles". This index conflicts
    with both, which is the risk you accepted.
  - **Pano IDs:** Tiles explicitly allows storing pano IDs indefinitely. Google also
    calls them transient, so live re-fetch (Part 5) needs a location-and-heading
    fallback when an ID has expired.
  - **Displayed tiles** must show the copyright and a "Report a problem with this image"
    link.
- **Recommendation: Street View tiles**, if you approve a tile cap. I suggest 95,000
  tiles/month, 95% of the free allowance. That gives 1.7–2.6× more free views per month
  and sharper views.
  - The cost is our own equirectangular-to-perspective rendering. The pilot should check
    it on about 20 views, comparing retrieval and inliers against the Static API
    rendering of the same pano and heading.
  - Without that approval, stay on the Static API at 9,500/month.
  - The code currently caps tiles at 0, so nothing can be fetched as tiles until you
    approve.

### 8. Descriptor storage: float16

Test on the 66 dev photos: exact search over the historic gallery with fp32 descriptors
vs fp16-stored descriptors cast back to fp32. Compute stays fp32.

| | Result |
| --- | --- |
| Top-1 identical | 66 / 66 |
| Top-5 identical, same order | 66 / 66 |
| Largest descriptor error | 2.9 × 10⁻⁵ |
| Largest top-5 score change | 8.2 × 10⁻⁶ |

The test search reproduces the eval's own similarities exactly (checked on 10 photos).
**Decision: store `gsv_v1` as float16**, 16.9 KB per view instead of 33.8 KB.

| Size | Disk (fp16) | In memory (fp32 FAISS flat) |
| --- | ---: | ---: |
| Per 9,500 views (one month, Static) | 160 MB | 321 MB |
| Per 25,000 views (one month, tiles) | 422 MB | 845 MB |
| Per 100,000 views | 1.7 GB | 3.4 GB |

The RAM figure matters for the Mac: the API already holds about 3.6 GB.
`faiss.IndexScalarQuantizer` with fp16 would halve it, but that should be measured first.

### 9. Map

**Pending:** `data/gsv/gap_plan.html` (planned panos coloured by priority) is generated
by `report` once the metadata run has produced `panos.json`.

## Finding that needs a decision before Part 5: EEA "No use With any Map"

Sweden is in the EEA. Since 8 July 2025, the EEA terms for both the Street View Static
API and Street View tiles say "Customer may not use any Google Maps Content from the
Street View Static API With any Map". Here, "With any Map" means "to display Google Maps
Content on, next to, or in a manner that is visually associated with any map, including
a Google map, or to link Google Maps Content to any map".

This goes beyond the indexing risk already accepted:
- **The demo:** Part 5.3 shows a live Street View image next to the Leaflet/OSM map.
  That is the prohibited case as written.
- **The core output:** the system itself shows a location derived from Street View
  content on a map. Whether that counts is a legal question I can't settle.

Options for the demo:
- show Street View matches with no map on screen (a separate view);
- or show only the attribution, capture date and pano link, not the image.

This applies only if the project is linked to an EEA billing account and was created,
or changed, after 8 July 2025.

## Reproduce

```bash
cd swespy
.venv-benchmark/bin/python -m eval.gsv_plan clip-osm --pbf <sweden-latest.osm.pbf>
.venv-benchmark/bin/python -m eval.gsv_plan points
.venv-benchmark/bin/python -m eval.gsv_plan metadata     # needs the key; free requests
.venv-benchmark/bin/python -m eval.gsv_plan report       # writes gap_plan.json + .html
.venv-benchmark/bin/python -m eval.gsv_plan fp16-check
```

Tests: `tests/unit/test_gsv.py` (10 tests). They cover:
- gap classification, priority order, cell dedup;
- pano dedup and the copyright filter;
- headings and 20 m sampling;
- the monthly cap: exactly one more request at 9,499, and tiles blocked;
- the pull-log schema with a planted fake key;
- a mocked metadata run that leaks no key and never sends a covered point;
- attribution, and the float16 round trip.
