"""Street View gap plan for Linköping (Part 1). Metadata only: no image requests.

Stages (each resumable, outputs in data/gsv/, which is git-ignored):

  clip-osm     Geofabrik Sweden extract -> camera-travelable ways near Linköping
               (data/osm/linkoping-ways.json). The 1 GB extract can be deleted after.
  points       Sample every 20 m along those ways inside the polygon and classify each
               point against the historic gallery: empty / old-only / covered.
  metadata     Street View metadata (free, uncapped) for empty and old-only points only,
               radius 25 m, source=outdoor; keep Google-captured panos; dedup by pano_id.
               Every batch is logged in data/gsv/pull_log.jsonl. Needs the API key.
  report       Counts per class / way type / priority, headings, image requests, months
               at the monthly cap; map data/gsv/gap_plan.html.
  fp16-check   Does storing gallery descriptors as float16 change dev top-5 retrieval?
  zoom-check   Does zoom-2 detail (512 px per 90° view) change dev top-5 retrieval?

Covered points never reach the Street View API; that is asserted, not assumed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval.densify import log, polygon_bbox, read_json, write_json
from swespy import gsv
from swespy.config import LINKOPING_AREA, in_area

ROOT = Path("data/gsv")
OSM_WAYS = Path("data/osm/linkoping-ways.json")
GALLERY_INDEX = Path("data/gallery_v2/gallery-index-historic.json")
DEV_ROOT = Path("data/public_eval/recent_linkoping_streets_v1")
GEOFABRIK_URL = "https://download.geofabrik.de/europe/sweden-latest.osm.pbf"

SAMPLE_STEP_M = 20.0
GAP_RADIUS_M = 25.0
OLD_BEFORE_YEAR = 2018
METADATA_RADIUS_M = 25
CELL_DEDUP_M = 10.0  # One metadata call per 10 m cell and priority (sidewalk + road twins).
BBOX_MARGIN_DEG = 0.01

ROAD = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
}
NON_ROAD = {"pedestrian", "footway", "cycleway", "path", "track", "steps", "bridleway"}
TRAVELABLE = ROAD | NON_ROAD
FOV, SIZE, PITCH = 90, 640, 0
M_PER_DEG_LAT = 111_320.0


# ---------------------------------------------------------------- geometry helpers


def m_per_deg_lng(lat: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Initial compass bearing from a to b, degrees in [0, 360)."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlng = math.radians(b[1] - a[1])
    x = math.sin(dlng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def segment_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat = (a[0] + b[0]) / 2
    return math.hypot((b[0] - a[0]) * M_PER_DEG_LAT, (b[1] - a[1]) * m_per_deg_lng(lat))


def sample_line(
    geometry: list[tuple[float, float]], step: float = SAMPLE_STEP_M
) -> list[tuple[float, float, float]]:
    """(lat, lng, bearing) every ``step`` metres along a polyline, starting at its start."""
    out: list[tuple[float, float, float]] = []
    carry = 0.0  # Distance along the current segment where the next sample falls.
    for a, b in zip(geometry, geometry[1:], strict=False):
        length = segment_m(a, b)
        if length == 0:
            continue
        heading = bearing(a, b)
        t = carry
        while t <= length:
            f = t / length
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), heading))
            t += step
        carry = t - length
    return out


def headings(way_bearing: float | None, priority: int) -> list[float]:
    """4 headings (priorities 1-2) or 2 sideways ones (priority 3), relative to the way."""
    base = way_bearing if way_bearing is not None else 0.0
    offsets = (0, 90, 180, 270) if priority in (1, 2) else (90, 270)
    return [round((base + o) % 360.0, 1) for o in offsets]


# ---------------------------------------------------------------- classification


class RefGrid:
    """Gallery references bucketed in a metric grid for fast radius queries."""

    def __init__(self, refs: list[dict[str, Any]], cell_m: float = GAP_RADIUS_M):
        self.cell = cell_m
        self.lat0 = 58.41
        self.kx = m_per_deg_lng(self.lat0)
        self.buckets: dict[tuple[int, int], list[tuple[float, float, int | None]]] = (
            defaultdict(list))
        for r in refs:
            self.buckets[self._key(r["lat"], r["lng"])].append(
                (r["lat"], r["lng"], r.get("capture_year")))

    def _key(self, lat: float, lng: float) -> tuple[int, int]:
        return (int(math.floor(lat * M_PER_DEG_LAT / self.cell)),
                int(math.floor(lng * self.kx / self.cell)))

    def near(self, lat: float, lng: float, radius: float) -> list[int | None]:
        """Capture years of references within ``radius`` metres."""
        ky, kx = self._key(lat, lng)
        reach = int(math.ceil(radius / self.cell))
        years = []
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                for rlat, rlng, year in self.buckets.get((ky + dy, kx + dx), ()):
                    d = math.hypot((rlat - lat) * M_PER_DEG_LAT, (rlng - lng) * self.kx)
                    if d <= radius:
                        years.append(year)
        return years


def classify(years: list[int | None]) -> str:
    """empty: nothing within 25 m; old-only: all known-dated and before 2018; else covered.

    A reference with an unknown capture year counts as covering the point, so budget is
    never spent on places that may already be covered.
    """
    if not years:
        return "empty"
    if all(y is not None and y < OLD_BEFORE_YEAR for y in years):
        return "old-only"
    return "covered"


def priority_of(gap: str, highway: str) -> int | None:
    if gap == "empty":
        return 1 if highway in NON_ROAD else 2
    if gap == "old-only":
        return 3
    return None


# ---------------------------------------------------------------- stage: clip-osm


def clip_osm(pbf: Path, out: Path = OSM_WAYS) -> None:
    """Two passes over the extract: node locations inside the bbox, then highway ways."""
    import osmium

    west, south, east, north = polygon_bbox()
    west, south = west - BBOX_MARGIN_DEG, south - BBOX_MARGIN_DEG
    east, north = east + BBOX_MARGIN_DEG, north + BBOX_MARGIN_DEG
    nodes: dict[int, tuple[float, float]] = {}
    for obj in osmium.FileProcessor(str(pbf), osmium.osm.NODE):
        loc = obj.location
        if loc.valid() and west <= loc.lon <= east and south <= loc.lat <= north:
            nodes[obj.id] = (loc.lat, loc.lon)
    ways = []
    for obj in osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(
            osmium.filter.KeyFilter("highway")):
        highway = obj.tags.get("highway")
        if highway not in TRAVELABLE or obj.tags.get("area") == "yes":
            continue
        # Split where the way leaves the bbox (missing nodes) so no false segments appear.
        part: list[tuple[float, float]] = []
        for ref in obj.nodes:
            p = nodes.get(ref.ref)
            if p is None:
                if len(part) >= 2:
                    ways.append({"id": obj.id, "highway": highway, "geometry": part})
                part = []
            else:
                part.append(p)
        if len(part) >= 2:
            ways.append({"id": obj.id, "highway": highway, "geometry": part})
    write_json(out, {
        "source": GEOFABRIK_URL, "extract_sha256": file_sha256(pbf),
        "clipped_at": datetime.now(UTC).isoformat(),
        "bbox": [west, south, east, north], "licence": "ODbL 1.0, © OpenStreetMap contributors",
        "ways": ways,
    })
    log("clip-osm", ways=len(ways), nodes_in_bbox=len(nodes))


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# ---------------------------------------------------------------- stage: points


def gallery_refs() -> list[dict[str, Any]]:
    refs = read_json(GALLERY_INDEX)["references_in_index_order"]
    return [r for r in refs if in_area(r["lng"], r["lat"], LINKOPING_AREA)]


def build_points(ways: list[dict[str, Any]], grid: RefGrid) -> list[dict[str, Any]]:
    points = []
    for way in ways:
        for lat, lng, way_bearing in sample_line([tuple(p) for p in way["geometry"]]):
            if not in_area(lng, lat, LINKOPING_AREA):
                continue
            gap = classify(grid.near(lat, lng, GAP_RADIUS_M))
            points.append({
                "lat": round(lat, 7), "lng": round(lng, 7), "bearing": round(way_bearing, 1),
                "highway": way["highway"], "way_id": way["id"], "gap": gap,
                "priority": priority_of(gap, way["highway"]),
            })
    return points


def points_stage(root: Path) -> None:
    ways = read_json(OSM_WAYS)["ways"]
    points = build_points(ways, RefGrid(gallery_refs()))
    write_json(root / "sample_points.json", {"step_m": SAMPLE_STEP_M, "points": points})
    log("points", total=len(points), **Counter(p["gap"] for p in points))


def metadata_queue(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gap points to query, in priority order, one per 10 m cell and priority."""
    seen: set[tuple[int, int, int]] = set()
    queue = []
    for p in sorted((p for p in points if p["priority"] is not None),
                    key=lambda p: p["priority"]):
        assert p["gap"] in ("empty", "old-only"), "covered points never reach Google"
        cell = (int(p["lat"] * M_PER_DEG_LAT // CELL_DEDUP_M),
                int(p["lng"] * m_per_deg_lng(p["lat"]) // CELL_DEDUP_M), p["priority"])
        if cell not in seen:
            seen.add(cell)
            queue.append(p)
    return queue


# ---------------------------------------------------------------- stage: metadata


def merge_panos(
    panos: dict[str, dict[str, Any]], point: dict[str, Any], meta: dict[str, Any]
) -> str:
    """Add one metadata answer; keep Google-captured panos, dedup by pano_id.

    A pano reached from several points keeps the best (lowest) priority and that point's
    way type and bearing. Returns what happened, for counting.
    """
    if meta.get("status") != "OK":
        return str(meta.get("status", "error")).lower()
    if not gsv.is_google_captured(meta.get("copyright")):
        return "not_google"
    pano_id = meta["pano_id"]
    old = panos.get(pano_id)
    if old is not None and old["priority"] <= point["priority"]:
        return "duplicate"
    panos[pano_id] = {
        "pano_id": pano_id, "lat": meta["location"]["lat"], "lng": meta["location"]["lng"],
        "capture_date": meta.get("date"), "priority": point["priority"],
        "highway": point["highway"], "bearing": point["bearing"],
        "headings": headings(point["bearing"], point["priority"]),
    }
    return "duplicate" if old is not None else "new"


async def metadata_stage(root: Path, concurrency: int = 16, batch: int = 500) -> None:
    import httpx

    key = gsv.api_key()
    points = read_json(root / "sample_points.json")["points"]
    queue = metadata_queue(points)
    state_path = root / "metadata_state.json"
    state = read_json(state_path) if state_path.exists() else {"done": 0, "panos": {}}
    panos: dict[str, dict[str, Any]] = state["panos"]
    pull_log = gsv.PullLog(root / "pull_log.jsonl")
    outcomes: Counter[str] = Counter(state.get("outcomes", {}))
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=30) as client:

        async def one(p: dict[str, Any]) -> dict[str, Any]:
            params = {"location": f"{p['lat']},{p['lng']}", "radius": METADATA_RADIUS_M,
                      "source": "outdoor", "key": key}
            async with semaphore:
                for attempt in range(5):
                    try:
                        r = await client.get(gsv.METADATA_URL, params=params)
                    except httpx.HTTPError as exc:
                        error = type(exc).__name__  # No message: it can contain the URL.
                    else:
                        if r.status_code == 200:
                            return r.json()
                        error = f"http_{r.status_code}"
                    await asyncio.sleep(2 ** attempt)
            return {"status": error}

        for start in range(state["done"], len(queue), batch):
            chunk = queue[start:start + batch]
            answers = await asyncio.gather(*[one(p) for p in chunk])
            for p, meta in zip(chunk, answers, strict=True):
                if meta.get("status") == "REQUEST_DENIED":
                    raise gsv.GsvRequestError(
                        "Metadata request denied: check that the key allows the Street View "
                        "Static API")  # The response message is not shown: it may echo input.
                outcomes[merge_panos(panos, p, meta)] += 1
            lats = [p["lat"] for p in chunk]
            lngs = [p["lng"] for p in chunk]
            ok = sum(1 for m in answers if m.get("status") in ("OK", "ZERO_RESULTS"))
            pull_log.append(request_type="metadata", request_count=len(chunk), success_count=ok,
                            bbox=(min(lngs), min(lats), max(lngs), max(lats)),
                            priority=min(p["priority"] for p in chunk))
            state = {"done": start + len(chunk), "panos": panos, "outcomes": dict(outcomes)}
            write_json(state_path, state)
            log("metadata", done=state["done"], of=len(queue), panos=len(panos),
                **dict(outcomes))
    write_json(root / "panos.json", {"created_at": datetime.now(UTC).isoformat(),
                                     "panos": sorted(panos.values(),
                                                     key=lambda p: p["priority"])})


# ---------------------------------------------------------------- stage: report


def report_stage(root: Path, cap: int = gsv.MONTHLY_IMAGE_CAP) -> dict[str, Any]:
    points = read_json(root / "sample_points.json")["points"]
    by_class = Counter(p["gap"] for p in points)
    by_way = defaultdict(Counter)
    for p in points:
        by_way[p["highway"]][p["gap"]] += 1
    queue = metadata_queue(points)
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "sample_points": {"total": len(points), **by_class},
        "by_way_type": {k: dict(v) for k, v in sorted(by_way.items())},
        "metadata_queue": {"total": len(queue),
                           **{f"priority_{k}": v for k, v in
                              sorted(Counter(p["priority"] for p in queue).items())}},
    }
    panos_path = root / "panos.json"
    if panos_path.exists():
        panos = read_json(panos_path)["panos"]
        per_priority = Counter(p["priority"] for p in panos)
        images = Counter()
        for p in panos:
            images[p["priority"]] += len(p["headings"])
        years = Counter((p["capture_date"] or "unknown")[:4] for p in panos)
        total = sum(images.values())
        report["panos"] = {f"priority_{k}": v for k, v in sorted(per_priority.items())}
        report["capture_years"] = dict(sorted(years.items()))
        report["image_requests"] = {f"priority_{k}": v for k, v in sorted(images.items())}
        report["image_requests_total"] = total
        report["months_at_cap"] = round(total / cap, 2)
        gap_map(root, panos)
    write_json(root / "gap_plan.json", report)
    return report


COLOURS = {1: "#d7301f", 2: "#fc8d59", 3: "#4575b4"}


def gap_map(root: Path, panos: list[dict[str, Any]]) -> None:
    """Planned panos coloured by priority (Leaflet + OSM; canvas for many markers)."""
    data = [[round(p["lat"], 6), round(p["lng"], 6), p["priority"]] for p in panos]
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Street View gap plan</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0;font:14px system-ui}}#map{{height:100vh}}
.legend{{background:#fff;padding:8px 10px;border-radius:6px;line-height:1.6}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}</style>
</head><body><div id="map"></div><script>
const panos = {json.dumps(data)};
const colours = {json.dumps(COLOURS)};
const map = L.map("map", {{preferCanvas: true}}).setView([58.41, 15.62], 13);
L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
  {{maxZoom: 19, attribution: "© OpenStreetMap contributors"}}).addTo(map);
const area = {json.dumps([[lat, lng] for lng, lat in LINKOPING_AREA])};
L.polygon(area, {{color: "#333", weight: 1, fill: false}}).addTo(map);
for (const [lat, lng, p] of panos)
  L.circleMarker([lat, lng], {{radius: 3, weight: 0, fillOpacity: 0.8, fillColor: colours[p]}})
    .addTo(map);
const legend = L.control({{position: "topright"}});
legend.onAdd = () => {{ const d = L.DomUtil.create("div", "legend"); d.innerHTML =
  `<b>${{panos.length}} planned panos</b><br>` +
  `<i style="background:${{colours[1]}}"></i>1 empty, non-road<br>` +
  `<i style="background:${{colours[2]}}"></i>2 empty, road<br>` +
  `<i style="background:${{colours[3]}}"></i>3 old-only (before 2018)`; return d; }};
legend.addTo(map);
</script></body></html>"""
    (root / "gap_plan.html").write_text(html)


# ---------------------------------------------------------------- stage: fp16-check


def fp16_check(root: Path, top_k: int = 5, chunk: int = 4096) -> dict[str, Any]:
    """Exact search over the historic gallery with fp32 vs fp16-stored descriptors.

    Compute stays fp32: fp16 descriptors are cast back to fp32 before the dot product.
    Descriptors are L2-normalised, so ranking by dot product equals FAISS L2 ranking.
    Streams the gallery in chunks; memory stays small.
    """
    index = read_json(GALLERY_INDEX)
    refs = index["references_in_index_order"]
    keep = [i for i, r in enumerate(refs) if in_area(r["lng"], r["lat"], LINKOPING_AREA)]
    folder = Path(index["descriptor_dir"])
    queries = read_json(DEV_ROOT / "manifest.json")["queries"]
    dev_cache = next(d for d in Path("data/megaloc/descriptors").iterdir()
                     if all((d / f"{q['sha256']}.npy").exists() for q in queries))
    q = np.stack([np.load(dev_cache / f"{x['sha256']}.npy") for x in queries]).astype(np.float32)
    best = {name: (np.full((len(q), 0), -np.inf), np.zeros((len(q), 0), dtype=np.int64))
            for name in ("fp32", "fp16")}
    max_err = 0.0
    for start in range(0, len(keep), chunk):
        ids = keep[start:start + chunk]
        g32 = np.stack([np.load(folder / f"{refs[i]['sha256']}.npy") for i in ids]).astype(
            np.float32)
        g16 = g32.astype(np.float16).astype(np.float32)
        max_err = max(max_err, float(np.abs(g32 - g16).max()))
        for name, g in (("fp32", g32), ("fp16", g16)):
            scores = q @ g.T
            all_s = np.concatenate([best[name][0], scores], axis=1)
            all_i = np.concatenate([best[name][1], np.broadcast_to(np.array(ids), scores.shape)],
                                   axis=1)
            order = np.argsort(-all_s, axis=1, kind="stable")[:, :top_k]
            best[name] = (np.take_along_axis(all_s, order, 1), np.take_along_axis(all_i, order, 1))
    a, b = best["fp32"][1], best["fp16"][1]
    result = {
        "queries": len(q), "gallery": len(keep), "top_k": top_k,
        "top1_identical": int((a[:, 0] == b[:, 0]).sum()),
        "top5_order_identical": int((a == b).all(axis=1).sum()),
        "top5_set_identical": int(sum(set(x) == set(y) for x, y in zip(a, b, strict=True))),
        "max_abs_descriptor_error": max_err,
        "max_abs_score_diff_top5": float(np.abs(best["fp32"][0] - best["fp16"][0]).max()),
        "bytes_per_descriptor": {"fp32": 8448 * 4, "fp16": 8448 * 2},
    }
    write_json(root / "fp16_check.json", result)
    return result


def exact_top(q: np.ndarray, refs: list[dict[str, Any]], keep: list[int], folder: Path,
              k: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k (scores, gallery indices) over stored fp32 descriptors, streamed."""
    best_s = np.full((len(q), 0), -np.inf)
    best_i = np.zeros((len(q), 0), dtype=np.int64)
    for start in range(0, len(keep), chunk):
        ids = keep[start:start + chunk]
        g = np.stack([np.load(folder / f"{refs[i]['sha256']}.npy") for i in ids]).astype(
            np.float32)
        s_ = np.concatenate([best_s, q @ g.T], axis=1)
        i_ = np.concatenate([best_i, np.broadcast_to(np.array(ids), (len(q), len(ids)))], axis=1)
        order = np.argsort(-s_, axis=1, kind="stable")[:, :k]
        best_s, best_i = np.take_along_axis(s_, order, 1), np.take_along_axis(i_, order, 1)
    return best_s, best_i


def zoom_check(root: Path, device: str = "mps", depth: int = 100, top_k: int = 5,
               variants: tuple[tuple[float, bool], ...] = ((512 / 640, True),)
               ) -> dict[str, Any]:
    """Offline: does zoom-2 detail (512 px per 90° view) change dev top-5 retrieval?

    Zoom 3 gives >= 640 px of real detail in a 640x640 view; zoom 2 gives 512 px (416 px
    on older 3328-px panoramas). Each variant is (detail, enlarge): shrink gallery images
    by ``detail`` (antialiased), optionally enlarge back to the original size (a 640 px
    render of 512 px detail), and encode. MegaLoc resizes every input to 322x322. Both
    conditions are re-encoded on the same device. Only each dev photo's exact
    top-``depth`` references are re-encoded; the report checks whether anything outside
    them could reach the top 5.
    """
    from PIL import Image

    from swespy.model.megaloc import MegaLocEmbedder

    index = read_json(GALLERY_INDEX)
    refs = index["references_in_index_order"]
    keep = [i for i, r in enumerate(refs) if in_area(r["lng"], r["lat"], LINKOPING_AREA)]
    folder = Path(index["descriptor_dir"])
    gallery_root = GALLERY_INDEX.parent
    queries = read_json(DEV_ROOT / "manifest.json")["queries"]
    dev_cache = next(d for d in Path("data/megaloc/descriptors").iterdir()
                     if all((d / f"{x['sha256']}.npy").exists() for x in queries))
    q = np.stack([np.load(dev_cache / f"{x['sha256']}.npy") for x in queries]).astype(np.float32)
    stored_s, stored_i = exact_top(q, refs, keep, folder, depth + 1)
    union = sorted({int(i) for i in stored_i[:, :depth].ravel()})
    embedder = MegaLocEmbedder(Path("data/megaloc"), device)

    def encode(ids: list[int], shrink: bool, detail: float = 1.0,
               enlarge: bool = True) -> np.ndarray:
        out = []
        for start in range(0, len(ids), 32):
            images = []
            for i in ids[start:start + 32]:
                with Image.open(gallery_root / refs[i]["path"]) as im:
                    im = im.convert("RGB")
                    if shrink:
                        w, h = im.size
                        small = im.resize((round(w * detail), round(h * detail)),
                                          Image.Resampling.LANCZOS)
                        im = small.resize((w, h), Image.Resampling.BILINEAR) if enlarge \
                            else small
                    images.append(im.copy())
            out.append(embedder.batch_images(images))
        return np.concatenate(out)

    z3 = encode(union, shrink=False)
    stored = np.stack([np.load(folder / f"{refs[i]['sha256']}.npy") for i in union])

    def rank(desc: np.ndarray) -> np.ndarray:
        scores = q @ desc.T  # Only references in the union can be in the top 5.
        return np.array(union)[np.argsort(-scores, axis=1, kind="stable")[:, :top_k]]

    top3 = rank(z3)
    cos_stored = np.sum(z3 * stored, axis=1)
    result: dict[str, Any] = {
        "queries": len(q), "references_re_encoded": len(union), "depth": depth,
        "device": device,
        "descriptor_cosine_reencoded_vs_stored": {
            "mean": float(cos_stored.mean()), "min": float(cos_stored.min())},
        "variants": []}
    for detail, enlarge in variants:
        z2 = encode(union, shrink=True, detail=detail, enlarge=enlarge)
        top2 = rank(z2)
        cos_z = np.sum(z3 * z2, axis=1)
        shift = float(np.linalg.norm(z2 - stored, axis=1).max())
        fifth_new = np.sort(q @ z2.T, axis=1)[:, -top_k]
        result["variants"].append({
            "detail": detail, "px_per_90deg_view": round(640 * detail),
            "enlarged_back_to_640": enlarge,
            "descriptor_cosine_vs_zoom3": {"mean": float(cos_z.mean()),
                                           "p1": float(np.percentile(cos_z, 1)),
                                           "min": float(cos_z.min())},
            "top1_identical": int((top3[:, 0] == top2[:, 0]).sum()),
            "top5_set_identical": int(sum(set(a) == set(b)
                                          for a, b in zip(top3, top2, strict=True))),
            "top5_order_identical": int((top3 == top2).all(axis=1).sum()),
            "queries_where_outside_refs_provably_cannot_enter_top5":
                int(np.sum(stored_s[:, depth] + shift < fifth_new)),
            "changed_queries": [
                {"query": queries[n]["image_id"],
                 "zoom3_top5": [refs[i]["image_id"] for i in top3[n]],
                 "zoom2_top5": [refs[i]["image_id"] for i in top2[n]]}
                for n in range(len(q)) if set(top3[n]) != set(top2[n])]})
    write_json(root / "zoom_check.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["clip-osm", "points", "metadata", "report",
                                          "fp16-check", "zoom-check"])
    parser.add_argument("--pbf", type=Path, help="Geofabrik extract (clip-osm)")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage == "clip-osm":
        clip_osm(args.pbf)
    elif args.stage == "points":
        points_stage(args.root)
    elif args.stage == "metadata":
        asyncio.run(metadata_stage(args.root))
    elif args.stage == "report":
        print(json.dumps(report_stage(args.root), indent=1))
    elif args.stage == "zoom-check":
        result = zoom_check(args.root, depth=30,
                            variants=((512 / 640, False), (416 / 640, False)))
        print(json.dumps({**result, "variants": [
            {k: v for k, v in x.items() if k != "changed_queries"}
            for x in result["variants"]]}, indent=1))
    else:
        print(json.dumps(fp16_check(args.root), indent=1))


if __name__ == "__main__":
    main()
