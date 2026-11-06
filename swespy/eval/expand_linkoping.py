"""Acquire a resumable city-wide reference gallery without modifying existing indexes/tests."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from eval.dense_linkoping import FIELDS, OLD_MANIFEST, formatted_link, normalized
from eval.mapillary_check import duplicate, signature
from eval.megaloc_compare import file_hash
from eval.run_eval import write_new
from swespy.config import get_settings
from swespy.data.mapillary import MapillaryClient, valid_image

BBOX = (15.50, 58.34, 15.75, 58.48)
GRID = (10, 8)
PER_TILE = 600
CENTRAL = Path("data/linkoping_dense/central_v1")
DEFAULT_ROOT = Path("data/linkoping_dense/city_v1")


def tiles() -> list[tuple[float, float, float, float]]:
    west, south, east, north = BBOX
    nx, ny = GRID
    return [(west + (east-west)*x/nx, south + (north-south)*y/ny,
             west + (east-west)*(x+1)/nx, south + (north-south)*(y+1)/ny)
            for x in range(nx) for y in range(ny)]


def select(
    items: list[dict[str, Any]], protected: set[str], existing_ids: set[str], cap: int,
) -> list[dict[str, Any]]:
    """Newest per 20m/45-degree bin, then round-robin through 1km cells."""
    bins: dict[tuple[int, int, int], dict[str, Any]] = {}
    for item in sorted(items, key=lambda r: (-r["captured_at"], r["image_id"])):
        if item["sequence"] in protected or item["image_id"] in existing_ids:
            continue
        heading = int(item["heading"] // 45) if item["heading"] is not None else -1
        key = (int(item["easting"] // 20), int(item["northing"] // 20), heading)
        bins.setdefault(key, item)
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in bins.values():
        cells[(int(item["easting"] // 1000), int(item["northing"] // 1000))].append(item)
    for values in cells.values():
        values.sort(key=lambda r: hashlib.sha256(r["image_id"].encode()).hexdigest())
    keys = sorted(cells)
    result = []
    for turn in range(max((len(v) for v in cells.values()), default=0)):
        for cell in keys:
            if turn < len(cells[cell]):
                result.append(cells[cell][turn])
                if len(result) >= cap:
                    return result
    return result


async def metadata(api: MapillaryClient, root: Path) -> list[dict[str, Any]]:
    directory = root / "metadata_tiles"
    directory.mkdir(exist_ok=True)
    items: dict[str, dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(4)
    completed = 0

    async def fetch(i: int, box: tuple[float, float, float, float]) -> None:
        nonlocal completed
        async with semaphore:
            await fetch_tile(i, box)

    async def fetch_tile(i: int, box: tuple[float, float, float, float]) -> None:
        nonlocal completed
        path = directory / f"{i:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["bbox"] != list(box) or saved["budget"] != PER_TILE:
                raise ValueError("Cached tile differs from acquisition protocol")
            rows = saved["items"]
        else:
            rows = []
            examined = 0
            cursor = None
            cursors: set[str] = set()
            while examined < PER_TILE:
                params: dict[str, Any] = {
                    "bbox": ",".join(map(str, box)), "fields": FIELDS,
                    "limit": min(200, PER_TILE-examined),
                }
                if cursor:
                    params["after"] = cursor
                payload = await api._request("/images", params)
                page = payload.get("data", [])[:params["limit"]]
                examined += len(page)
                for raw in page:
                    accepted = normalized(raw, BBOX)
                    if accepted is not None:
                        rows.append(accepted)
                cursor = payload.get("paging", {}).get("cursors", {}).get("after")
                if not page or not cursor or cursor in cursors:
                    break
                cursors.add(cursor)
            write_new(path, {"bbox": box, "budget": PER_TILE, "examined": examined,
                             "budget_reached": examined == PER_TILE, "items": rows})
        for item in rows:
            items[item["image_id"]] = item
        completed += 1
        print(f"Metadata tiles {completed}/{len(tiles())}: "
              f"{len(items)} unique candidates (saved tile {i+1})", flush=True)

    await asyncio.gather(*(fetch(i, box) for i, box in enumerate(tiles())))
    return sorted(items.values(), key=lambda r: r["image_id"])


def coverage(references: list[dict[str, Any]]) -> dict[str, Any]:
    """Report occupied cells and camera bounds, without implying continuous coverage."""
    return {
        "occupied_1km_cells": len({(int(r["easting"]//1000), int(r["northing"]//1000))
                                  for r in references}),
        "occupied_100m_cells": len({(int(r["easting"]//100), int(r["northing"]//100))
                                   for r in references}),
        "capture_sequences": len({r["sequence"] for r in references}),
        "creators": len({r["creator"] for r in references}),
        "camera_bbox": [min(r["lng"] for r in references), min(r["lat"] for r in references),
                        max(r["lng"] for r in references), max(r["lat"] for r in references)],
    }


def render_report(root: Path) -> None:
    """Map actual downloaded camera positions, with attributed source/photo links."""
    saved = json.loads((root/"manifest.json").read_text())
    summary = saved["coverage"]
    points = [{k: r[k] for k in ("lat", "lng", "image_id", "creator", "source_url", "path")}
              for r in saved["references"]]
    encoded = json.dumps(points).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Linköping reference gallery coverage</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{font:16px system-ui;margin:24px;color:#17253c;background:#f8f9fb}
#map{height:72vh;border-radius:12px}p{max-width:1000px;line-height:1.6}
.leaflet-popup-content img{max-width:280px;max-height:180px}a{color:#165ca3}</style>
<h1>Linköping reference gallery</h1>
<p>__SUMMARY__</p>
<p>Dots show downloaded reference camera positions, not guaranteed continuous coverage.
Click a dot to inspect the photo and its attribution. The old central gallery is preserved;
held-out test capture sequences and near-duplicate test photos are excluded.
This is reference acquisition; model and API indexes have not been changed.</p>
<div id="map"></div>
<p>Mapillary imagery: <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.
Map and library require an internet connection. Coordinates are provider labels and have
not been independently verified. <a href="coverage.geojson">Download camera GeoJSON</a>.</p>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const points=__POINTS__;
const map=L.map('map',{preferCanvas:true});
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
 attribution:'&copy; OpenStreetMap contributors',maxZoom:19}).addTo(map);
for(const p of points){
 L.circleMarker([p.lat,p.lng],{radius:3,color:'#165ca3',weight:1,fillOpacity:.55})
 .addTo(map).bindPopup(()=>{
   const box=document.createElement('div');
   const img=document.createElement('img'); img.src=p.path; img.alt='Reference '+p.image_id;
   box.append(img,document.createElement('br'));
   const a=document.createElement('a'); a.href=p.source_url; a.target='_blank';
   a.rel='noopener'; a.textContent='Mapillary photo by '+p.creator; box.append(a);
   return box;
 });
}
map.fitBounds(points.map(p=>[p.lat,p.lng]));
window.galleryPointCount=points.length;
</script></html>"""
    text = (f"{saved['reference_count']:,} reference photos; "
            f"{summary['capture_sequences']:,} capture sequences; "
            f"{summary['creators']:,} contributors; "
            f"{summary['occupied_1km_cells']} occupied 1km grid cells. "
            f"{saved['preserved_central_count']} original central references retained.")
    page = page.replace("__SUMMARY__", text).replace("__POINTS__", encoded)
    path = root/"coverage.html"
    if not path.exists():
        path.write_text(page)


async def prepare(root: Path, cap: int) -> None:
    settings = get_settings()
    if not settings.mapillary_access_token.strip():
        raise ValueError("Mapillary token is required")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        saved = json.loads((root / "manifest.json").read_text())
        for row in saved["references"]:
            if file_hash(root / row["path"]) != row["sha256"]:
                raise ValueError("Frozen gallery bytes changed")
        render_report(root)
        print("Validated existing gallery:", len(saved["references"]), flush=True)
        return
    central = json.loads((CENTRAL / "manifest.json").read_text())
    old = json.loads(OLD_MANIFEST.read_text())
    protected = set(central["reserved_query_sequences"])
    protected.update(q["sequence"] for q in central["queries"] + old["queries"])
    protected.update(central["protected_previous_query_sequences"])
    query_signatures = central["queries"] + old["queries"]
    protocol = {"bbox": BBOX, "grid": GRID, "per_tile_metadata_budget": PER_TILE,
                "gallery_cap": cap, "central_manifest_sha256": file_hash(CENTRAL/"manifest.json"),
                "previous_manifest_sha256": file_hash(OLD_MANIFEST),
                "protected_sequences": sorted(protected)}
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("Existing acquisition protocol differs; use a new root")
    else:
        write_new(protocol_path, protocol)
    originals = root / "originals"
    originals.mkdir(exist_ok=True)
    async with MapillaryClient(settings) as api:
        items = await metadata(api, root)
        central_refs = central["references"]
        candidates = select(items, protected, {r["image_id"] for r in central_refs},
                            cap-len(central_refs))
        print(f"Selected {len(candidates)} additions + {len(central_refs)} preserved references",
              flush=True)
        refs = []
        for row in central_refs:
            if row["sequence"] in protected:
                raise ValueError("Central reference conflicts with protected test sequences")
            original = originals / f"{row['image_id']}.jpg"
            if not original.exists():
                os.link(CENTRAL/row["path"], original)
            if file_hash(original) != row["sha256"]:
                raise ValueError("Preserved reference bytes changed")
            refs.append({**row, "path": formatted_link(root, row, "database")})
        semaphore = asyncio.Semaphore(6)
        rejected: Counter[str] = Counter()
        completed = 0
        successful: dict[str, dict[str, Any]] = {}
        cache = {r["image_id"]: CENTRAL/r["path"]
                 for r in central_refs}

        async def worker(item: dict[str, Any]) -> None:
            nonlocal completed
            async with semaphore:
                if shutil.disk_usage(root).free < 4*1024**3:
                    raise RuntimeError("Stopped to preserve 4 GiB free disk space; rerun to resume")
                path = originals / f"{item['image_id']}.jpg"
                try:
                    local = cache.get(item["image_id"], Path("data/raw")/path.name)
                    if not path.exists() and local.is_file() and valid_image(local):
                        os.link(local, path)
                    try:
                        await api.download_image(item["thumb_url"], path)
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code not in (401, 403, 404):
                            raise
                        fresh = await api._request(f"/{item['image_id']}", {"fields": FIELDS})
                        await api.download_image(fresh["thumb_1024_url"], path)
                    sig = await asyncio.to_thread(signature, path)
                    if duplicate(sig, query_signatures):
                        rejected["similar_to_protected_query"] += 1
                    else:
                        successful[item["image_id"]] = {
                            **{k: v for k, v in item.items() if k != "thumb_url"}, **sig,
                        }
                except (httpx.HTTPError, OSError, ValueError, SyntaxError) as exc:
                    # Do not log signed URLs or request credentials.
                    rejected[f"download_or_decode_{type(exc).__name__}"] += 1
                completed += 1
                if completed % 100 == 0 or completed == len(candidates):
                    print(f"Photos {completed}/{len(candidates)}: "
                          f"{len(successful)+len(refs)} validated; "
                          f"{sum(rejected.values())} rejected",
                          flush=True)

        await asyncio.gather(*(worker(item) for item in candidates))
        known_hashes = {r["sha256"] for r in refs}
        for item in candidates:
            row = successful.get(item["image_id"])
            if row is None:
                continue
            if row["sha256"] in known_hashes:
                rejected["exact_gallery_duplicate"] += 1
                continue
            known_hashes.add(row["sha256"])
            refs.append({**row, "path": formatted_link(root, row, "database")})
        if any(key.startswith("download_or_decode_") for key in rejected):
            write_new(root/f"attempt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json",
                      {"accepted": len(refs), "rejected": dict(rejected)})
            raise RuntimeError("Downloads failed; completed files retained. "
                               "Rerun before publishing")
        if {r["sequence"] for r in refs} & protected:
            raise RuntimeError("Protected sequence leaked into gallery")
        manifest = {
            "kind": "linkoping_city_reference_gallery", "created_at": datetime.now(UTC).isoformat(),
            **protocol, "reference_count": len(refs), "preserved_central_count": len(central_refs),
            "candidate_count": len(items), "selected_additions": len(candidates),
            "rejections": dict(rejected), "coverage": coverage(refs),
            "selection": "20m XY x 45-degree heading bins; newest per bin; "
                         "1km cell round robin; SHA256 image-ID order within cells",
            "duplicate_screen": "Protected queries: SHA256/pHash<=6/dHash<=5; "
                                "gallery: exact SHA256",
            "metadata_tile_sha256": {p.name: file_hash(p)
                                     for p in sorted((root/"metadata_tiles").glob("*.json"))},
            "references": refs, "queries": [],
            "limitations": ["Bounded first pass, not complete or continuous city coverage",
                            "Mapillary availability and metadata budgets limit coverage",
                            "GPS, headings and capture dates not independently verified",
                            "Reference acquisition only; no training or production index update"],
        }
        write_new(root/"manifest.json", manifest)
        geojson = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lng"], r["lat"]]},
             "properties": {k: r[k] for k in ("image_id", "creator", "sequence", "source_url")}}
            for r in refs]}
        write_new(root/"coverage.geojson", geojson)
        render_report(root)
        print(json.dumps({"reference_count": len(refs), "coverage": manifest["coverage"],
                          "rejections": dict(rejected)}, indent=2), flush=True)
        print("CITY_GALLERY_MANIFEST", root/"manifest.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cap", type=int, default=10000)
    args = parser.parse_args()
    if not 1000 <= args.cap <= 15000:
        parser.error("cap must be between 1,000 and 15,000")
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    asyncio.run(prepare(args.root, args.cap))


if __name__ == "__main__":
    main()
