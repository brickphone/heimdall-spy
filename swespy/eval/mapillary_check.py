"""Automated, sequence-held-out street sanity check; never a private-photo baseline.

Run from swespy/: python -m eval.mapillary_check
Artifacts and third-party query photos stay under ignored data/mapillary_eval/.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from loguru import logger
from PIL import Image, ImageOps

from eval.dataset import Photo
from eval.metrics import distance_m, observation, print_table, summarize
from eval.run_eval import index_fingerprint, write_new
from swespy.config import get_settings
from swespy.data.mapillary import MapillaryClient
from swespy.vector_store.qdrant_client import VectorStore

LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
FIELDS = "id,sequence,captured_at,geometry,creator,thumb_2048_url"


def record(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Require explicit sequence, attribution and finite coordinates; fail closed."""
    try:
        image_id = str(raw["id"])
        sequence = raw["sequence"]
        creator = raw["creator"]["username"]
        lng, lat = raw["geometry"]["coordinates"]
        lat, lng = float(lat), float(lng)
        captured_at = int(raw["captured_at"])
        if not (
            image_id.isascii()
            and image_id.isdecimal()
            and isinstance(sequence, str)
            and sequence
            and isinstance(creator, str)
            and creator
            and math.isfinite(lat)
            and math.isfinite(lng)
            and -90 <= lat <= 90
            and -180 <= lng <= 180
            and captured_at > 0
        ):
            return None
        return {
            "image_id": image_id,
            "sequence": sequence,
            "creator": creator,
            "captured_at": captured_at,
            "lat": lat,
            "lng": lng,
            "source_url": f"https://www.mapillary.com/app/?pKey={image_id}&focus=photo",
            "license_url": LICENSE,
            "thumb_url": raw.get("thumb_2048_url"),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def signature(path: Path) -> dict[str, Any]:
    """Hash exact bytes and low-frequency structure for conservative duplicate screening."""
    contents = path.read_bytes()
    with Image.open(io.BytesIO(contents)) as original:
        image = ImageOps.exif_transpose(original).convert("L")
        small = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS))
        dhash_bits = (small[:, 1:] > small[:, :-1]).ravel()
        pixels = np.asarray(
            image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
        )
    basis = np.cos(np.pi / 32 * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None])
    dct = (basis @ pixels @ basis.T).ravel()[1:]  # Ignore brightness/DC coefficient.

    def bits(values: np.ndarray) -> int:
        result = 0
        for bit in values:
            result = (result << 1) | int(bit)
        return result

    return {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "dhash": bits(dhash_bits),
        "phash": bits(dct > np.median(dct)),
    }


def duplicate(sig: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    """Reject exact copies or close perceptual hashes; this is not a proof of independence."""
    return any(
        sig["sha256"] == other["sha256"]
        or (sig["phash"] ^ other["phash"]).bit_count() <= 6
        or (sig["dhash"] ^ other["dhash"]).bit_count() <= 5
        for other in others
    )


def eligible(
    item: dict[str, Any], reference_ids: set[str], reference_sequences: set[str],
    bbox: tuple[float, float, float, float],
) -> bool:
    """Only accept known, separate sequences within the fixed pilot area."""
    west, south, east, north = bbox
    return (
        item["image_id"] not in reference_ids
        and item["sequence"] not in reference_sequences
        and west <= item["lng"] <= east
        and south <= item["lat"] <= north
    )


def reference_points(store: VectorStore) -> list[dict[str, Any]]:
    """Read all indexed reference payloads without upserting or changing the collection."""
    points: list[dict[str, Any]] = []
    offset = None
    while True:
        page, offset = store.client.scroll(
            collection_name=store.collection_name, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for point in page:
            payload = point.payload or {}
            if not all(k in payload for k in ("image_id", "lat", "lng")):
                raise RuntimeError("Reference point is missing required provenance")
            points.append({**payload, "image_id": str(payload["image_id"])})
        if offset is None:
            break
    if not points:
        raise RuntimeError("Reference collection is empty; rebuild it separately")
    return points


async def prepare(root: Path, target: int, candidate_limit: int) -> Path:
    """Freeze an independently selected street query set before any predictions are run."""
    settings = get_settings()
    if settings.testing or not settings.mapillary_access_token.strip():
        raise RuntimeError("Real index and a private Mapillary token are required")
    bbox = settings.sweden_bbox
    west, south, east, north = bbox
    if not (15.50 <= west < east <= 15.75 and 58.30 <= south < north <= 58.50):
        raise ValueError("Evaluation acquisition must stay within the Linkoping pilot")
    store = VectorStore(settings)
    try:
        before = index_fingerprint(store)
        references = reference_points(store)
        ids = [r["image_id"] for r in references]
        metadata: dict[str, dict[str, Any]] = {}
        cache = root / "reference_metadata.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved["index_sha256"] == before:
                metadata = {r["image_id"]: r for r in saved["references"]}
        async with MapillaryClient(settings) as api:
            if not metadata:
                for start in range(0, len(ids), 40):
                    batch = ids[start:start + 40]
                    payload = await api._request(
                        "/", {"ids": ",".join(batch), "fields": FIELDS}
                    )
                    for image_id in batch:
                        item = record(payload.get(image_id, {}))
                        if item is None or item["image_id"] != image_id:
                            raise RuntimeError(
                                "Reference sequence/attribution lookup incomplete; cannot exclude "
                                "shared sequences safely"
                            )
                        metadata[image_id] = {k: v for k, v in item.items() if k != "thumb_url"}
                    print(
                        f"Reference provenance {min(start + 40, len(ids))}/{len(ids)}", flush=True
                    )
                root.mkdir(parents=True, exist_ok=True)
                if cache.exists():
                    cache = root / f"reference_metadata-{before[:16]}.json"
                if not cache.exists():
                    write_new(
                        cache, {"index_sha256": before, "references": list(metadata.values())}
                    )
            if set(metadata) != set(ids):
                raise RuntimeError("Cached reference provenance does not match index")
            reference_sequences = {r["sequence"] for r in metadata.values()}
            excluded_ids = set(ids) | {p.stem for p in Path("data/raw").glob("*.jpg")}
            print(f"Excluding {len(excluded_ids)} image IDs and "
                  f"{len(reference_sequences)} reference sequences", flush=True)
            candidates: dict[str, dict[str, Any]] = {}
            seen_count = 0
            for i in range(6):
                for j in range(6):
                    box = (
                        west + (east - west) * i / 6, south + (north - south) * j / 6,
                        west + (east - west) * (i + 1) / 6,
                        south + (north - south) * (j + 1) / 6,
                    )
                    limit = min(math.ceil(candidate_limit / 36), candidate_limit - seen_count)
                    if limit <= 0:
                        break
                    payload = await api._request(
                        "/images", {"bbox": ",".join(map(str, box)), "fields": FIELDS,
                                    "limit": limit}
                    )
                    page = payload.get("data", [])[:limit]
                    seen_count += len(page)
                    for raw in page:
                        item = record(raw)
                        if item and eligible(item, excluded_ids, reference_sequences, bbox):
                            candidates[item["image_id"]] = item
                    print(f"Grid {i * 6 + j + 1}/36: {len(candidates)} eligible candidates",
                          flush=True)
            ordered = sorted(candidates.values(), key=lambda r: r["image_id"])
            random.Random(42).shuffle(ordered)
            reference_signatures = [signature(p) for p in sorted(Path("data/raw").glob("*.jpg"))]
            run = root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            photos = run / "photos"
            photos.mkdir(parents=True)
            selected: list[dict[str, Any]] = []
            selected_signatures: list[dict[str, Any]] = []
            counts: Counter[str] = Counter()
            rejections: Counter[str] = Counter()
            downloads = 0
            for item in ordered:
                if len(selected) >= target or downloads >= target * 3:
                    break
                if counts[item["sequence"]] >= 3:
                    rejections["sequence_cap"] += 1
                    continue
                if any(distance_m(item["lat"], item["lng"], s["lat"], s["lng"]) < 100
                       for s in selected):
                    rejections["query_spacing"] += 1
                    continue
                if not item["thumb_url"]:
                    rejections["missing_url"] += 1
                    continue
                dest = photos / f"{item['image_id']}.jpg"
                downloads += 1
                await api.download_image(item["thumb_url"], dest)
                sig = signature(dest)
                if duplicate(sig, reference_signatures + selected_signatures):
                    rejections["near_duplicate"] += 1
                    dest.unlink()  # Only the newly acquired, rejected evaluation file.
                    continue
                selected.append({
                    **{k: v for k, v in item.items() if k != "thumb_url"},
                    **sig, "filename": dest.name, "viewpoint_type": "street",
                })
                selected_signatures.append(sig)
                counts[item["sequence"]] += 1
                print(f"Accepted test photo {len(selected)}/{target}", flush=True)
            if index_fingerprint(store) != before:
                raise RuntimeError("Reference index changed during acquisition")
            manifest = {
                "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
                "kind": "mapillary_sequence_holdout_sanity_check",
                "index_sha256": before, "reference_count": len(references),
                "reference_sequences": sorted(reference_sequences),
                "bbox": bbox, "requested_count": target,
                "metadata_candidate_limit": candidate_limit, "metadata_examined": seen_count,
                "eligible_candidates": len(ordered), "downloads_attempted": downloads,
                "rejections": dict(rejections), "query_sequence_count": len(counts),
                "selection": "seed 42; >=100m query spacing; <=3 queries per sequence",
                "near_duplicate_screen": "exact SHA256; 63-bit pHash <=6 OR 64-bit dHash <=5",
                "limitations": [
                    "Same-provider street-only sanity check, not independent user-photo baseline",
                    "Sequence exclusion and perceptual hashes cannot prove visual independence",
                    "Mapillary GPS labels may contain error; no manual label verification",
                    "Sparse references limit achievable geographic recall",
                ],
                "queries": selected,
            }
            path = run / "manifest.json"
            write_new(path, manifest)
            print(
                f"Saved {len(selected)} queries from {len(counts)} sequences to {path}", flush=True
            )
            if len(selected) < target:
                print("Target not reached under fixed exclusion rules; no rules were relaxed.",
                      flush=True)
            return path
    finally:
        store.client.close()


def verify_manifest(manifest: dict[str, Any], folder: Path) -> None:
    """Recheck sequence exclusions, hashes and spacing before any evaluation."""
    queries = manifest["queries"]
    if not queries:
        raise ValueError("No independent query photos available under the selection rules")
    sequences = set(manifest["reference_sequences"])
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for i, item in enumerate(queries):
        name = item["filename"]
        if Path(name).name != name or not name.endswith(".jpg"):
            raise ValueError("Unsafe query filename")
        if item["sequence"] in sequences or item["image_id"] in seen:
            raise ValueError("Reference sequence leakage or repeated query ID")
        seen.add(item["image_id"])
        counts[item["sequence"]] += 1
        if counts[item["sequence"]] > 3:
            raise ValueError("Query sequence cap exceeded")
        if any(distance_m(item["lat"], item["lng"], q["lat"], q["lng"]) < 100
               for q in queries[:i]):
            raise ValueError("Query spacing violated")
        if signature(folder / "photos" / name)["sha256"] != item["sha256"]:
            raise ValueError("Frozen query bytes changed")


def thumbnail(path: Path) -> str:
    """Embed a small preview so the local report does not require an image server."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((360, 220))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def render_map(path: Path, report: dict[str, Any]) -> None:
    """Write a local interactive Leaflet report with attribution and a table fallback."""
    payload = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linköping street retrieval check</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{font:15px system-ui;margin:24px;color:#14213d;background:#f7f8fa}
h1{font-size:25px}#map{height:510px;border-radius:12px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px}
article{background:white;padding:16px;border:1px solid #ddd;border-radius:12px}
img{width:100%;height:180px;object-fit:contain;background:#eef}button{padding:8px;cursor:pointer}
.muted{color:#526174}a{color:#165ca3}.badge{font-weight:600}.legend{margin:12px 0}
</style><h1 id="title">Linköping street retrieval check</h1><p id="summary"></p>
<p class="muted" id="context-note">Mapillary sequence holdout · street-level only ·
similarity is uncalibrated.
Same-provider results may be optimistic. Hash screening cannot prove independence.</p>
<div class="legend">🟢 Camera label &nbsp; 🔴 <span id="prediction-label">Averaged prediction</span>
&nbsp; 🔵 Ranked candidates</div>
<div id="map"></div><div class="grid" id="cards"></div>
<p class="muted" id="attribution-note">Previews are resized Mapillary images,
attributed individually under CC BY-SA 4.0.
The map needs internet for Leaflet and OpenStreetMap tiles; photo cards remain self-contained.</p>
<script id="report" type="application/json">REPORT_DATA</script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const r=JSON.parse(document.getElementById('report').textContent),m=r.metrics.overall;
if(r.model_label){document.getElementById('title').textContent=r.model_label;
 document.title=r.model_label;}
if(r.prediction_label)document.getElementById('prediction-label').textContent=r.prediction_label;
if(r.context_note)document.getElementById('context-note').textContent=r.context_note;
if(r.attribution_note)document.getElementById('attribution-note').textContent=
 r.attribution_note+' Map tiles and library require internet.';
document.getElementById('summary').textContent=`${m.count} queries from
${r.query_sequence_count} ${r.query_group_label ?? 'sequences'}.
Median pin error: ${m.median_error_m.toFixed(0)} m. Within 100 m: ${m.pct_within_100m.toFixed(1)}%.
Within 500 m: ${m.pct_within_500m.toFixed(1)}%.
Top-${r.top_k ?? 5} within ${r.hit_radius_m ?? 100} m: ${m.top_k_hit_pct.toFixed(1)}%.`;
const map=typeof L!=='undefined'?L.map('map'):null, bounds=[];
if(map)L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
attribution:'© OpenStreetMap contributors',maxZoom:19}).addTo(map);
const detail=map?L.layerGroup().addTo(map):null;
function pin(lat,lng,color,label,group){return L.circleMarker([lat,lng],{
radius:6,color,fillOpacity:.8}).addTo(group||map).bindTooltip(label)}
function focus(q){if(!map)return;detail.clearLayers();
 q.prediction.top_matches.forEach((c,i)=>pin(c.lat,c.lng,'#1269b0',`Candidate ${i+1}`,detail));
 L.circle([q.lat,q.lng],{radius:100,color:'#16834a',fillOpacity:.06}).addTo(detail);
 const pts=[[q.lat,q.lng],[q.prediction.predicted_lat,q.prediction.predicted_lng],
 ...q.prediction.top_matches.map(c=>[c.lat,c.lng])];map.fitBounds(pts,{padding:[35,35],maxZoom:17});
 document.getElementById('map').scrollIntoView({behavior:'smooth'});}
r.queries.forEach((q,i)=>{const p=q.prediction,actual=[q.lat,q.lng];
 const guess=[p.predicted_lat,p.predicted_lng];
 if(map){bounds.push(actual,guess);
 pin(...actual,'#16834a',`Query ${i+1}: camera`).on('click',()=>focus(q));
 pin(...guess,'#cf3030',`Query ${i+1}: prediction`).on('click',()=>focus(q));
 L.polyline([actual,guess],{color:'#999',weight:1}).addTo(map)}
 const card=document.createElement('article'),img=document.createElement('img');
 img.src=q.preview;img.alt=`Query ${i+1}`;card.append(img);
 const h=document.createElement('h3');
 h.textContent=`Query ${i+1} · ${q.error_m.toFixed(0)} m pin error`;
 card.append(h);const text=document.createElement('p');
 text.textContent=`Nearest reference: ${q.nearest_reference_m.toFixed(0)} m.
 Best of ${r.top_k ?? 5}: ${q.best_candidate_error_m?.toFixed(0) ?? 'n/a'} m.
 Top-ranked: ${q.top_1_error_m?.toFixed(0) ?? 'n/a'} m.
 Camera: ${q.lat.toFixed(6)}, ${q.lng.toFixed(6)}.
 Captured ${new Date(q.captured_at).toISOString().slice(0,10)}.`;
 card.append(text);const source=document.createElement('a');source.href=q.source_url;
 source.textContent=`${q.provider ?? 'Mapillary'} photo by ${q.creator}`;
 source.target='_blank';source.rel='noopener';
 card.append(source,document.createTextNode(' · '));const license=document.createElement('a');
 license.href=q.license_url;license.textContent=q.license_name ?? 'CC BY-SA 4.0';
 card.append(license);
 const button=document.createElement('button');button.textContent='Inspect candidates on map';
 button.onclick=()=>focus(q);card.append(document.createElement('p'),button);
 document.getElementById('cards').append(card)});
if(map&&bounds.length)map.fitBounds(bounds,{padding:[20,20]});
if(!map)document.getElementById('map').textContent=
 'Map library unavailable. Photo results are below.';
</script></html>"""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(page.replace("REPORT_DATA", payload))


async def evaluate(manifest_path: Path) -> Path:
    """Call the live API, verify a frozen index, and save a separate sanity-check report."""
    manifest = json.loads(manifest_path.read_text())
    folder = manifest_path.parent
    verify_manifest(manifest, folder)
    settings = get_settings()
    if settings.testing:
        raise RuntimeError("TESTING is forbidden")
    store = VectorStore(settings)
    try:
        before = index_fingerprint(store)
        if before != manifest["index_sha256"]:
            raise RuntimeError("Reference index changed since query selection")
        references = reference_points(store)
        ids = {r["image_id"] for r in references}
        if any(q["image_id"] in ids for q in manifest["queries"]):
            raise RuntimeError("Query image is present in reference index")
        rows: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=180) as client:
            (await client.get("/health")).raise_for_status()
            for i, item in enumerate(manifest["queries"], start=1):
                if i > 1:
                    await asyncio.sleep(60 / settings.api_rate_limit_per_minute + 0.05)
                photo_path = folder / "photos" / item["filename"]
                response = await client.post(
                    "/predict", files={"file": ("query.jpg", photo_path.read_bytes(), "image/jpeg")}
                )
                response.raise_for_status()
                prediction = response.json()
                photo = Photo(item["filename"], photo_path, item["lat"], item["lng"],
                              "street", item["sha256"])
                measured = observation(photo, prediction, 100)
                observations.append(measured)
                distances = [distance_m(item["lat"], item["lng"], c["lat"], c["lng"])
                             for c in prediction["top_matches"]]
                nearest = min(distance_m(item["lat"], item["lng"], r["lat"], r["lng"])
                              for r in references)
                rows.append({
                    **item, **measured, "prediction": prediction,
                    "best_candidate_error_m": min(distances) if distances else None,
                    "top_1_error_m": distances[0] if distances else None,
                    "nearest_reference_m": nearest, "preview": thumbnail(photo_path),
                })
                print(f"Evaluated {i}/{len(manifest['queries'])}: "
                      f"pin error {measured['error_m']:.0f}m", flush=True)
        if index_fingerprint(store) != before:
            raise RuntimeError("Index changed during evaluation; results not saved")
        revision_file = Path("data/huggingface/hub") / (
            "models--" + settings.clip_model_name.replace("/", "--")
        ) / "refs/main"
        report = {
            **{k: v for k, v in manifest.items() if k != "queries"},
            "evaluated_at": datetime.now(UTC).isoformat(), "metrics": summarize(observations),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "coverage": {
                f"pct_queries_with_reference_within_{radius}m":
                100 * sum(q["nearest_reference_m"] <= radius for q in rows) / len(rows)
                for radius in (100, 500, 2000)
            },
            "model": settings.clip_model_name,
            "cached_model_revision": revision_file.read_text().strip(),
            "inference_source_sha256": hashlib.sha256(
                b"".join(p.read_bytes() for p in sorted(Path("swespy").rglob("*.py")))
            ).hexdigest(),
            "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
            "top_k": settings.top_k_results, "hit_radius_m": 100,
            "confidence_status": "uncalibrated mean visual similarity",
            "queries": rows,
        }
        output = folder / datetime.now(UTC).strftime("report-%Y%m%dT%H%M%S.%fZ.json")
        write_new(output, report)
        render_map(output.with_suffix(".html"), report)
        print_table(report["metrics"])
        print("Coverage:", json.dumps(report["coverage"]))
        print("Saved report:", output, flush=True)
        return output
    finally:
        store.client.close()


def main() -> None:
    """Prepare and evaluate, or evaluate a frozen manifest without downloading again."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/mapillary_eval"))
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--candidate-limit", type=int, default=2000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if not 1 <= args.target <= 50 or not 1 <= args.candidate_limit <= 3000:
        parser.error("Target must be 1–50 and candidate limit 1–3000")
    logger.remove()
    logger.add(sys.stderr, level=get_settings().log_level)

    async def run() -> None:
        manifest = args.manifest or await prepare(args.root, args.target, args.candidate_limit)
        if not args.prepare_only:
            await evaluate(manifest)

    asyncio.run(run())


if __name__ == "__main__":
    main()
