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
| Web (Next.js 16, TypeScript, Tailwind 4, Leaflet) | 127.0.0.1:3000 | Long-running Node server (`output: "standalone"`, `node server.js`). Serves the page and forwards `/api/*` unchanged to the API. No API routes, server actions or serverless functions |

The page shows "Waiting for the API…" and keeps Locate disabled until `GET /ready` says
the model is loaded. MegaLoc and ALIKED run on the Apple GPU (LightGlue on CPU), with
re-ranking on (K = 5, 1,024 keypoints). Measured: **3–6 s per photo**.

Web development: `cd web && npm run dev` (hot reload on 127.0.0.1:3000, API still needed).
Checks: `npm run typecheck && npm run lint`.

## What the page does

- Upload a JPEG, PNG or HEIC photo. The page calls the public `POST /predict`, exactly
  as any client would; there is no demo-specific inference logic. Code:
  `swespy/web/src/components/Demo.tsx`, `ResultMap.tsx`, `src/lib/api.ts` (types mirror
  `swespy/api/schemas.py`).
- **Map** (Leaflet + OpenStreetMap): the predicted location is the main pin;
  candidates 2–5 are small pins with their inlier counts.
- **Confidence label:** "Confident" or "Low confidence", with the reason (too few
  inliers, or another verified candidate far away). On the dev set, "confident" was
  right within 500 m for 92% of photos and "low" for 21%.
- **Matched reference** next to your photo, with credit: source link (Mapillary /
  Wikimedia Commons / KartaView), contributor, licence (e.g. CC BY-SA 4.0) and
  "resized". `GET /references/{image_id}/image` serves only images in the loaded
  gallery, looked up by ID.
- **Processing time** (server and round trip) and the **versions** block: MegaLoc
  revision and weights hash, gallery file, stage and index hash, re-ranking settings.
- **"Save this photo for later evaluation"** (off by default): after the result, click
  the map at the true location and press Save. The page then calls `POST /field-photos`,
  which writes `data/field_photos/<time>-<hash>/photo.jpg` (pixels only, no EXIF/GPS)
  and `record.json` (true location, prediction, versions). Field photos are a future
  evaluation set only: never added to the gallery, dev set or any tuning.

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
