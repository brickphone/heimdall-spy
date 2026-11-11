# Local demo: try the system on your own photos

Localhost only. Nothing is published, deployed or bound to other interfaces.

## Start and stop (Mac, native, fastest)

```bash
cd ~/Documents/GitHub/heimdall-spy/swespy
scripts/run_local.sh            # then open http://127.0.0.1:3000
```

Stop both servers with **Ctrl-C**. The script builds the web app, starts the API in the
background, waits until its `/ready` answers (~70 s), then starts the web server:

| Process | Address | What it does |
| --- | --- | --- |
| API (FastAPI/uvicorn) | 127.0.0.1:8000 | All inference. Loads model and gallery once (~60 s; prints nothing until `Application startup complete`) |
| Web (Next.js 16, TypeScript, Tailwind 4, MapLibre GL) | 127.0.0.1:3000 | Long-running Node server (`output: "standalone"`, `node server.js`). Serves the page and forwards `/api/*` unchanged to the API. No API routes, server actions or serverless functions |

The web server starts once the API's `/ready` answers. MegaLoc and ALIKED run on the
Apple GPU (LightGlue on CPU), with re-ranking on (K = 5, 1,024 keypoints). Measured:
**3–6 s per photo**.

Web development: `cd web && npm run dev` (hot reload on 127.0.0.1:3000, API still needed).
Checks: `npm run typecheck && npm run lint && npm run check:tokens && npm run check:contrast`.
End-to-end (with the demo running): `node scripts/e2e.mjs <expected.json>`.

## What the page does ("Heimdall" workspace)

A light, map-first workspace. All inference stays in FastAPI; the page calls only the
public endpoints (`POST /predict/stream`, `GET /references/{id}/image`).

### Flow: photo, region, then Locate

1. **Choose or drop a photo:** use the button, or drop it anywhere on the map, which shows
   a full-map "Drop to add photo" target. The photo appears with region selection active.
   **Nothing is sent until Locate is pressed.**
2. **Optionally drag a box** around the part that shows the place (keyboard: arrows move
   it, Shift with arrows resizes). "Use whole photo" clears it.
3. **Locate.** After a result, "Change region" lets you adjust the box and locate again.

### The locating sequence (the map is the loading state)

Driven by the streamed pipeline events, not by timers. Each phase holds a minimum time so
fast answers still read clearly.

| Event from `/predict/stream` | Map | Inspector |
| --- | --- | --- |
| Locate pressed (min 900 ms) | Dims (paper at 55%, soft vignette); frames the Linköping polygon, tilts to 35°, rotates about 4°/s. Centre label "Searching Linköping" with Cancel | Stage list: "Search reference photos" running |
| `candidates` (top 50) | Rotation continues; 50 dots fade in, strongest first, over ~600 ms. Label: "Comparing the 5 closest matches" | Search done (seconds); "Compare closest matches" running |
| `stage: verification` | Rotation stops; ~1.2 s flight to the top 5; the other 45 dots fade out; wide dashed rings | |
| `verified` × 5 | Each candidate's rings ease to their final radius (500 ms): tight for strong matches, wide for weak | |
| `result` | Dim lifts; flight to the best match at zoom 16, pitch 45°, 3D buildings; prediction rings draw in, inner to outer; callout fades in | Sections fill; the stage list collapses to "Located in 4.1 s" (expand for per-stage times) |

- **Cancel:** aborts the request, which stops the work on the server; lifts the dim;
  returns the camera; keeps the photo and region.
- **Errors:** the label becomes the message plus "Try again".
- **Reduced motion:** no flights or rotation (camera jumps), states crossfade, rings
  appear at their final size.
- **Candidate selection after the result:** flies between candidates at pitch 45° in
  ~800 ms.

**Reference coverage** is no longer in the workspace. It's a static, non-interactive map
on `/about` with a sentence explaining it. The workspace never requests coverage or
gallery points; the only reference image it loads is the matched one, in the compare
view after a result.

### Streaming API

`POST /predict/stream` (same form fields as `/predict`) returns NDJSON
(`application/x-ndjson`). Events, in order:
- `{"type":"stage","stage":"retrieval"}`
- `{"type":"candidates","items":[{id,lat,lng,similarity}×50]}`: locations only, never images or sources
- `{"type":"stage","stage":"verification","count":5}`
- `{"type":"verified","id","inliers","score"}`, one per candidate as it completes
- `{"type":"result", …exactly /predict's response…}`
- `{"type":"error","message"}` on failure

How it's built and verified:
- **One code path:** both routes run `GeoLocator.locate_events`; `/predict` returns its
  final result, so the two cannot differ. On the running stack the streamed result was
  byte-identical to `/predict` for 3 dev photos, and both match the frozen evaluation.
- **Cancellation:** a client disconnect sets a flag the pipeline checks between steps
  (tested: work stops within one step).
- **Proxy:** the Next.js proxy passes the stream through unbuffered. Measured: candidates
  at 1.3 s, each verification ~0.35 s apart, result at 4.2 s.

### Region, place names, About

- **Region:** the form field `region` = JSON `{"x","y","width","height"}`, normalised to
  the upright photo; retrieval and re-ranking both see only that part.
- **Place names:** offline reverse geocoding from OpenStreetMap street and district names
  (`public/data/places.json`).
- **About:** the dev results with 95% intervals, read from
  `eval/results/dev_historic_rerank.json` at build time; the coverage map; failure modes
  with a real manual-crop example; data and licences.

### Checks (2026-11-11)

| Check | Result |
| --- | --- |
| Python suites | 173 + 178 passed |
| Stream tests | order and schema; same result as `/predict`; disconnect cancels work (`tests/integration/test_predict_stream.py`) |
| Real model, 3 dev photos | stream result = `/predict`, exactly |
| Playwright (`web/scripts/e2e.mjs`), 1440×900 and 390×844 | 32 / 33 |
| Nothing sent before Locate; region active by default; drawn region sent | pass |
| Status label follows the stream; "Located in …" with stage timings | pass |
| API-equals: UI top 5, details and state = eval, 3 dev photos | pass |
| Cancel mid-verification (photo and region kept, Locate offered); error label with Try again | pass |
| Network: no coverage or gallery-point requests in the workspace | pass |
| Keyboard; focus ring 2 px cobalt, 2 px offset; contrast; tokens; type check; lint | pass |
| **Frame budget: no long task > 100 ms** | **Not fully met**, see below |

**Frame budget on the M1.**
- With screenshots off, most runs have no long task at all. About 1 run in 9 has a single
  frame of 109–181 ms, always while the API is still computing.
- The API's feature extractor (ALIKED) runs on the same Apple GPU that renders the map, so
  that is likely contention, though not proven.
- Seen once: one 95 ms frame at Locate, when the tilted camera loads many tiles.
- Hiding labels while the map is dimmed did not remove the slow frames; the label hiding
  was kept.
- Moving verification to the CPU would avoid the overlap but makes that step about 4×
  slower.

### Screenshots

**Locating sequence** (1440×900, fixed times after Locate):

| | |
| --- | --- |
| ![](screenshots/sequence-01-0.3s.png) 0.3 s | ![](screenshots/sequence-02-0.9s.png) 0.9 s |
| ![](screenshots/sequence-03-1.5s.png) 1.5 s | ![](screenshots/sequence-04-2.1s.png) 2.1 s |
| ![](screenshots/sequence-05-2.7s.png) 2.7 s | ![](screenshots/sequence-06-3.3s.png) 3.3 s |
| ![](screenshots/sequence-07-4.0s.png) 4.0 s | ![](screenshots/sequence-08-4.8s.png) 4.8 s |
| ![](screenshots/sequence-09-5.8s.png) 5.8 s | ![](screenshots/sequence-10-7.5s.png) 7.5 s |

**Other states:**

| | |
| --- | --- |
| Empty state (polygon outline, no coverage) | ![](screenshots/desktop-1-empty.png) |
| Photo chosen, region active, Locate | ![](screenshots/desktop-2-ready.png) |
| Result | ![](screenshots/desktop-3-result.png) |
| Result with a drawn region | ![](screenshots/desktop-4-region-result.png) |
| Candidate 3 selected | ![](screenshots/desktop-5-candidate3.png) |
| Compare | ![](screenshots/desktop-6-compare.png) |
| Cancel during verification: before / after | ![](screenshots/cancel-1-before.png) ![](screenshots/cancel-2-after.png) |
| Error | ![](screenshots/error.png) |
| Reduced motion | ![](screenshots/reduced-1.png) ![](screenshots/reduced-2.png) ![](screenshots/reduced-3-result.png) |
| About, with the coverage map | ![](screenshots/desktop-7-about.png) |
| Mobile 390×844 | ![](screenshots/mobile-1-empty.png) ![](screenshots/mobile-2-result.png) ![](screenshots/mobile-3-compare.png) |

## Privacy

- `/predict` processes the upload in memory and writes nothing. A test checks that no
  file appears; logs contain no image hash or coordinates.
- **No location metadata is used.** Every upload goes through
  `swespy/model/imageio.py:decode_upload`, which applies EXIF orientation and then
  copies only the RGB pixels, so EXIF/GPS/XMP never reach the model. A test checks that
  a GPS-tagged JPEG gives exactly the same pixels as an untagged one.

## iPhone photos

- HEIC is decoded server-side by `pillow-heif` (optional `heic` extra, installed in the
  local `.venv`). EXIF orientation is applied, so portrait photos are upright. Chrome
  cannot preview HEIC, so "Your photo" shows a placeholder there; Safari previews it.
- A HEIC *converted* from a JPEG is a re-compressed image: on borderline photos the top
  candidates are within 0.01 cosine, so the answer can change. Checked on one dev photo:
  the HEIC and a lossless PNG of its decoded pixels gave identical results, and a JPEG
  re-saved at quality 80 already moved the inlier counts. The decoder is faithful; the
  model is sensitive to compression on hard photos.
- Licence: `pillow-heif`'s binary wheels are **GPL-2.0** (they bundle the x265
  encoder; only the LGPL decoders are used). It is therefore optional and **not in the
  default Docker image**. Build with `--build-arg EXTRAS=heic` to include it. Decide
  before distributing that image (GPL source-offer obligations).

## Docker (CPU, for parity with a server)

The compose file runs the API only; the Next.js app is not containerised yet.

```bash
cd ~/Documents/GitHub/heimdall-spy/swespy
docker compose up --build       # http://127.0.0.1:8000/demo ; stop: Ctrl-C, then `docker compose down`
```

| | Measured 2026-11-09 (Docker Desktop, Apple M1, linux/arm64) |
| --- | --- |
| Image | multi-stage; **2.06 GB** unpacked (458 MB compressed); no weights or gallery inside |
| Mounted read-only | `gallery-index-historic.json`, `gallery_v2/images` (4.8 GB), `megaloc` (3.2 GB), `huggingface` (0.9 GB), `torch` (48 MB); `field_photos` read-write |
| Startup to `/ready` | **75 s** |
| Peak memory | **5.4 GiB** (Docker VM has 7.65 GB) |
| `/predict` | 20–22 s (CPU only: Docker on a Mac cannot use the Apple GPU) |
| Correctness | 2 dev photos: top-1, confidence state and inliers identical to the evaluation |

Fixes found while testing the container:

- **OOM at startup:** the gallery loader made several full copies of 3.1 GB of
  descriptors. It now streams them into FAISS in 4,096-row chunks.
- **faiss-cpu 1.9 on ARM64 Linux crashed** probing SVE via `numpy.distutils` (absent on
  Python 3.12). `FAISS_OPT_LEVEL=generic` skips the probe.
- **Read-only mounts:** the descriptor cache folder is now created only when written to.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness |
| `GET /ready` | 200 + versions once model and gallery are loaded, else 503 |
| `POST /predict` | Upload → top 5 candidates, scores, inliers, confidence state, versions |
| `GET /references/{image_id}/image` | Stored 640 px reference image (for attribution display) |
| `POST /field-photos` | Opt-in save of a field photo + true location (future evaluation only) |
