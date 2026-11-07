"""Build reference gallery v2 inside the project polygon (Part 4).

Stages are resumable and idempotent; rerun any stage after an interruption.

    python -m eval.densify mapillary-meta   # vector tiles + Graph API metadata (no images)
    python -m eval.densify commons-meta     # Commons geosearch + imageinfo
    python -m eval.densify kartaview-meta   # KartaView nearby-photo search
    python -m eval.densify plan             # thin, quality-by-metadata, leakage guards
    python -m eval.densify download         # images (checkpointed, rate-limited, disk guard)
    python -m eval.densify crops            # panorama -> perspective crops
    python -m eval.densify filter           # blur / sky-ground / near-duplicate rules
    python -m eval.densify encode           # MegaLoc descriptors (hash-keyed cache)
    python -m eval.densify index            # gallery-index.json for swespy.model.locator
    python -m eval.densify coverage         # heatmap + dev nearest_reference_m

Requirements: docs/GALLERY_V2_SPEC.md. Licences: docs/LICENSES.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from eval import leakage, panorama, provenance, quality
from eval.common import signature
from eval.geo_eval import haversine_m
from swespy.config import LINKOPING_AREA, get_settings, in_area

ROOT = Path("data/gallery_v2")
LEGACY_GALLERY = Path("data/linkoping_dense/city_v1/manifest.json")
TILE_ZOOM = 14
THIN_DISTANCE_M = 10.0  # User decision 2026-11-07 (3 m did not fit on disk).
THIN_HEADING_DEG = 20.0
MIN_FREE_BYTES = 3 * 2**30
IMAGE_MAX_SIDE = 640  # Stored size; MegaLoc itself resizes to 322 px.
JPEG_QUALITY = 88


def build_parameters() -> dict[str, Any]:
    """Every setting that determines the gallery, recorded in each build artifact."""
    return {
        "polygon_lnglat": [list(p) for p in LINKOPING_AREA],
        "mapillary_tile_zoom": TILE_ZOOM,
        "thinning": {
            "distance_m": THIN_DISTANCE_M,
            "heading_deg": THIN_HEADING_DEG,
            "scope": "within one sequence, capture order, vs already-kept frames",
            "panoramas": "always kept",
        },
        "image_storage": {
            "max_side_px": IMAGE_MAX_SIDE,
            "jpeg_quality": JPEG_QUALITY,
            "panorama_source": "thumb_2048_url, kept at source size for cropping",
        },
        "panorama_crops": {
            "count": panorama.CROP_COUNT,
            "fov_deg": panorama.FOV_DEG,
            "size_px": panorama.CROP_SIZE,
        },
        "quality": {
            "blur_effect_threshold": quality.BLUR_EFFECT_THRESHOLD,
            "edge_share_threshold": quality.EDGE_SHARE_THRESHOLD,
            "duplicate_radius_m": quality.DUPLICATE_RADIUS_M,
            "same_spot_m": quality.SAME_SPOT_M,
        },
        "leakage_guard": {
            "radius_m": leakage.RADIUS_M,
            "window_minutes": leakage.WINDOW.total_seconds() / 60,
            "session": "same photographer, same Europe/Stockholm day",
        },
        "commons_grid_m": 700,
        "kartaview_grid_m": 700,
    }


def store_resized(data: bytes, max_side: int = IMAGE_MAX_SIDE) -> bytes:
    """Re-encode to the stored gallery size (EXIF orientation applied, metadata dropped)."""
    import io

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


MAPILLARY_FIELDS = (
    "id,computed_geometry,geometry,compass_angle,computed_compass_angle,"
    "captured_at,sequence,creator,is_pano,camera_type,width,height"
)
CC_BY_SA_4 = "https://creativecommons.org/licenses/by-sa/4.0/"
UA = (
    "HeimdallSpyResearchBot/0.1 (https://github.com/brickphone/heimdall-spy; "
    "reference gallery build) httpx/" + httpx.__version__
)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


# ---------------------------------------------------------------- small helpers


def write_json(path: Path, data: Any) -> None:
    """Atomic write so an interrupted stage never leaves a half-written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def log(stage: str, **data: Any) -> None:
    print(
        json.dumps(
            {"stage": stage, "at": datetime.now(UTC).isoformat(), **data}, ensure_ascii=False
        ),
        flush=True,
    )


def year_of(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, UTC).year


def angle_diff(a: float | None, b: float | None) -> float:
    """Smallest absolute heading difference; unknown headings count as similar."""
    if a is None or b is None:
        return 0.0
    d = abs(a - b) % 360
    return min(d, 360 - d)


def polygon_bbox() -> tuple[float, float, float, float]:
    lngs = [p[0] for p in LINKOPING_AREA]
    lats = [p[1] for p in LINKOPING_AREA]
    return min(lngs), min(lats), max(lngs), max(lats)


def grid_centres(spacing_m: float) -> list[tuple[float, float]]:
    """Uniform grid over the polygon; a centre is kept if its cell touches the polygon."""
    west, south, east, north = polygon_bbox()
    dlat = spacing_m / 111_320
    dlng = spacing_m / (111_320 * math.cos(math.radians((south + north) / 2)))
    centres = []
    lat = south + dlat / 2
    while lat < north + dlat / 2:
        lng = west + dlng / 2
        while lng < east + dlng / 2:
            corners = [
                (lng + sx * dlng / 2, lat + sy * dlat / 2) for sx in (-1, 0, 1) for sy in (-1, 0, 1)
            ]
            if any(in_area(x, y) for x, y in corners):
                centres.append((lat, lng))
            lng += dlng
        lat += dlat
    return centres


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


async def get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    attempts: int = 6,
) -> httpx.Response:
    """GET with exponential backoff on 429/5xx and network errors (never on 401/403)."""
    for attempt in range(attempts):
        try:
            response = await client.get(url, params=params, headers=headers)
        except (httpx.TransportError, httpx.TimeoutException):
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(2**attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
            retry = response.headers.get("retry-after")
            await asyncio.sleep(float(retry) if retry and retry.isdigit() else 2**attempt)
            continue
        return response
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- Mapillary metadata


def tile_range() -> list[tuple[int, int]]:
    west, south, east, north = polygon_bbox()
    n = 2**TILE_ZOOM

    def tx(lng: float) -> int:
        return int((lng + 180) / 360 * n)

    def ty(lat: float) -> int:
        return int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

    return [(x, y) for x in range(tx(west), tx(east) + 1) for y in range(ty(north), ty(south) + 1)]


def decode_tile(content: bytes, x: int, y: int) -> list[dict[str, Any]]:
    import mapbox_vector_tile

    n = 2**TILE_ZOOM
    rows = []
    layer = mapbox_vector_tile.decode(content).get("image", {})
    extent = layer.get("extent", 4096)
    for feature in layer.get("features", []):
        px, py = feature["geometry"]["coordinates"]
        lng = (x + px / extent) / n * 360 - 180
        yy = y + 1 - py / extent  # Decoder returns y-up tile coordinates.
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        p = feature["properties"]
        rows.append(
            {
                "id": str(p["id"]),
                "sequence": p.get("sequence_id"),
                "captured_at": p.get("captured_at"),
                "is_pano": bool(p.get("is_pano")),
                "tile_lat": lat,
                "tile_lng": lng,
            }
        )
    return rows


async def mapillary_meta(root: Path) -> None:
    token = get_settings().mapillary_access_token.strip()
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN is required")
    tiles_dir, ent_dir = root / "mapillary" / "tiles", root / "mapillary" / "entities"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    ent_dir.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    async with httpx.AsyncClient(timeout=60) as client:
        for x, y in tile_range():
            path = tiles_dir / f"{TILE_ZOOM}_{x}_{y}.json"
            if not path.exists():
                r = await get_with_backoff(
                    client,
                    f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{TILE_ZOOM}/{x}/{y}",
                    params={"access_token": token},
                )
                r.raise_for_status()
                write_json(path, decode_tile(r.content, x, y))
                await asyncio.sleep(0.2)  # Tile limit is 50,000/day; we need ~160.
            # Keep a 150 m margin so computed positions can move images into the polygon.
            for row in read_json(path):
                lat, lng = row["tile_lat"], row["tile_lng"]
                if in_area(lng, lat) or any(
                    in_area(lng + dx, lat + dy)
                    for dx in (-0.0026, 0.0026)
                    for dy in (-0.0014, 0.0014)
                ):
                    ids.add(row["id"])
        log("mapillary-meta", tiles=len(tile_range()), candidate_ids=len(ids))
        ordered = sorted(ids)
        headers = {"Authorization": f"OAuth {token}"}
        semaphore = asyncio.Semaphore(8)  # ~6 s per 100-ID request; far below rate limits.
        done = 0

        async def fetch(batch: list[str]) -> None:
            nonlocal done
            path = ent_dir / f"{hashlib.sha256(','.join(batch).encode()).hexdigest()[:24]}.json"
            if not path.exists():
                async with semaphore:
                    r = await get_with_backoff(
                        client,
                        "https://graph.mapillary.com/images",
                        params={"image_ids": ",".join(batch), "fields": MAPILLARY_FIELDS},
                        headers=headers,
                    )
                r.raise_for_status()
                write_json(path, r.json().get("data", []))
            done += len(batch)
            if done % 10_000 < len(batch):
                log("mapillary-meta", entities_done=done, of=len(ordered))

        await asyncio.gather(*[fetch(ordered[i : i + 100]) for i in range(0, len(ordered), 100)])
    log("mapillary-meta", state="complete", entities=len(ordered))


def mapillary_rows(root: Path) -> list[dict[str, Any]]:
    """Normalise entity metadata to gallery rows (computed geometry preferred)."""
    rows = []
    for path in sorted((root / "mapillary" / "entities").glob("*.json")):
        for e in read_json(path):
            computed = (e.get("computed_geometry") or {}).get("coordinates")
            raw = (e.get("geometry") or {}).get("coordinates")
            if computed:
                (lng, lat), location_source = computed, "mapillary_computed"
                heading = e.get("computed_compass_angle", e.get("compass_angle"))
            elif raw:
                (lng, lat), location_source = raw, "mapillary_raw"
                heading = e.get("compass_angle")
            else:
                continue
            if e.get("captured_at") is None or not e.get("sequence"):
                continue
            heading = None if heading is None else float(heading) % 360
            rows.append(
                {
                    "source": "mapillary",
                    "source_id": str(e["id"]),
                    "ref_id": f"mapillary-{e['id']}",
                    "sequence": e["sequence"],
                    "creator": (e.get("creator") or {}).get("username"),
                    "captured_at": int(e["captured_at"]),
                    "capture_year": year_of(e["captured_at"]),
                    "lat": float(lat),
                    "lng": float(lng),
                    "location_source": location_source,
                    "heading": heading,
                    "is_pano": bool(e.get("is_pano")),
                    "camera_type": e.get("camera_type"),
                    "license_url": CC_BY_SA_4,
                    "source_url": f"https://www.mapillary.com/app/?pKey={e['id']}&focus=photo",
                }
            )
    return rows


def thin_sequences(
    rows: list[dict[str, Any]],
    distance_m: float = THIN_DISTANCE_M,
    heading_deg: float = THIN_HEADING_DEG,
) -> list[dict[str, Any]]:
    """Per sequence, in capture order, drop frames within ``distance_m`` and
    ``heading_deg`` of an already-kept frame. Panoramas are always kept."""
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_seq[r["sequence"]].append(r)
    kept = []
    for seq_rows in by_seq.values():
        cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for r in sorted(seq_rows, key=lambda r: (r["captured_at"], r["source_id"])):
            if r.get("is_pano"):
                kept.append(r)
                continue
            cy = int(r["lat"] * 111_320 // distance_m)
            cx = int(r["lng"] * 111_320 * math.cos(math.radians(r["lat"])) // distance_m)
            near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for k in cells[(cx + dx, cy + dy)]]
            if any(
                float(
                    haversine_m(r["lat"], r["lng"], np.array([k["lat"]]), np.array([k["lng"]]))[0]
                )
                <= distance_m
                and angle_diff(r["heading"], k["heading"]) <= heading_deg
                for k in near
            ):
                continue
            cells[(cx, cy)].append(r)
            kept.append(r)
    return kept


# ---------------------------------------------------------------- Commons metadata

COMMONS_TIME = re.compile(r"(\d{4})[-:](\d{2})[-:](\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?")


def commons_time(value: str) -> tuple[int, str] | None:
    """(captured_at ms, precision) from DateTimeOriginal; local time assumed Stockholm."""
    m = COMMONS_TIME.search(re.sub(r"<[^>]+>", "", value))
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    try:
        moment = datetime(
            int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0), tzinfo=leakage.LOCAL
        )
    except ValueError:
        return None
    return int(moment.timestamp() * 1000), ("exact" if hh else "day")


def commons_row(page: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Normalise one Commons file page; returns (row, rejection reason)."""
    import html as html_lib

    def plain(v: str) -> str:
        return html_lib.unescape(re.sub(r"<[^>]+>", "", v)).strip()

    try:
        info = page["imageinfo"][0]
        ext = {k: str(v["value"]) for k, v in info.get("extmetadata", {}).items()}
    except (KeyError, IndexError):
        return None, "no_metadata"
    if info.get("mime") != "image/jpeg":
        return None, "not_jpeg"
    url = ext.get("LicenseUrl", "").replace("http://", "https://")
    url = "https:" + url if url.startswith("//") else url
    if not provenance.license_allowed(url):
        return None, "license_not_commercial"
    if "GPSLatitude" not in ext:
        return None, "no_camera_gps"
    taken = commons_time(ext.get("DateTimeOriginal", ""))
    if taken is None:
        return None, "no_capture_time"
    lat, lng = float(ext["GPSLatitude"]), float(ext["GPSLongitude"])
    if not in_area(lng, lat):
        return None, "outside_polygon"
    title, description = page["title"], plain(ext.get("ImageDescription", ""))
    if quality.indoor_text(title, description, ext.get("Categories", "")):
        return None, "indoor"
    creator = plain(ext.get("Artist", ""))
    return {
        "source": "wikimedia_commons",
        "source_id": str(page["pageid"]),
        "ref_id": f"commons-{page['pageid']}",
        "sequence": f"commons-{creator}-{taken[0] // 86_400_000}",
        "creator": creator,
        "captured_at": taken[0],
        "capture_time_precision": taken[1],
        "capture_year": year_of(taken[0]),
        "lat": lat,
        "lng": lng,
        "location_source": "commons_exif_low_precision",
        "heading": None,
        "is_pano": False,
        "license_url": url,
        "license_name": plain(ext.get("LicenseShortName", "")),
        "source_url": info["descriptionurl"],
        "title": title,
        "download_url": info.get("thumburl", info["url"]),
    }, "ok"


async def commons_request(client: httpx.AsyncClient, params: dict[str, Any]) -> dict[str, Any]:
    """Commons Action API call; transient overload errors back off and retry, 403 never."""
    for attempt in range(7):
        started = time.monotonic()
        r = await get_with_backoff(
            client, COMMONS_API, params={"action": "query", "format": "json", "maxlag": 5, **params}
        )
        r.raise_for_status()  # Never retry or evade 403.
        payload = r.json()
        code = payload.get("error", {}).get("code")
        if code in ("maxlag", "cirrussearch-too-busy-error", "ratelimited") and attempt < 6:
            await asyncio.sleep(min(120, 5 * 2**attempt))
            continue
        if code:
            raise ValueError(f"Commons API error: {code}")
        await asyncio.sleep(5 if time.monotonic() - started > 1 else 0.25)
        return payload
    raise RuntimeError("unreachable")


async def commons_meta(root: Path) -> None:
    folder = root / "commons"
    (folder / "geosearch").mkdir(parents=True, exist_ok=True)
    (folder / "info").mkdir(parents=True, exist_ok=True)
    pageids: set[str] = set()
    async with httpx.AsyncClient(
        headers={"User-Agent": UA}, timeout=60, follow_redirects=True
    ) as client:
        queue = [(lat, lng, 500) for lat, lng in grid_centres(700)]
        while queue:
            lat, lng, radius = queue.pop()
            path = folder / "geosearch" / f"{lat:.5f}_{lng:.5f}_{radius}.json"
            if path.exists():
                found = read_json(path)
            else:
                found = (
                    (
                        await commons_request(
                            client,
                            {
                                "list": "geosearch",
                                "gscoord": f"{lat}|{lng}",
                                "gsradius": radius,
                                "gsnamespace": 6,
                                "gslimit": 500,
                            },
                        )
                    )
                    .get("query", {})
                    .get("geosearch", [])
                )
                write_json(path, found)
            if len(found) >= 500 and radius > 100:  # Saturated: split into four.
                off = radius / 2 / 111_320
                offl = off / math.cos(math.radians(lat))
                queue += [
                    (lat + sy * off, lng + sx * offl, radius // 2)
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                ]
            pageids |= {str(item["pageid"]) for item in found}
        ordered = sorted(pageids)
        log("commons-meta", pages=len(ordered))
        for start in range(0, len(ordered), 50):
            batch = ordered[start : start + 50]
            path = (
                folder
                / "info"
                / f"{hashlib.sha256('|'.join(batch).encode()).hexdigest()[:24]}.json"
            )
            if not path.exists():
                data = await commons_request(
                    client,
                    {
                        "pageids": "|".join(batch),
                        "prop": "imageinfo",
                        "iiprop": "url|mime|size|extmetadata",
                        "iiurlwidth": 1024,
                    },
                )
                write_json(path, list(data.get("query", {}).get("pages", {}).values()))
    log("commons-meta", state="complete")


def commons_rows(root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows, reasons = [], Counter()
    seen = set()
    for path in sorted((root / "commons" / "info").glob("*.json")):
        for page in read_json(path):
            if str(page.get("pageid")) in seen:
                continue
            seen.add(str(page.get("pageid")))
            row, reason = commons_row(page)
            reasons[reason] += 1
            if row:
                rows.append(row)
    return rows, reasons


# ---------------------------------------------------------------- KartaView metadata


async def kartaview_meta(root: Path) -> None:
    folder = root / "kartaview" / "nearby"
    folder.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers={"User-Agent": UA}, timeout=60) as client:
        for lat, lng in grid_centres(700):
            for page in range(1, 50):
                path = folder / f"{lat:.5f}_{lng:.5f}_p{page}.json"
                if path.exists():
                    result = read_json(path)
                else:
                    r = await get_with_backoff(
                        client,
                        "https://api.kartaview.org/2.0/photo/",
                        params={
                            "lat": lat,
                            "lng": lng,
                            "radius": 500,
                            "itemsPerPage": 100,
                            "page": page,
                        },
                    )
                    r.raise_for_status()
                    result = r.json().get("result", {})
                    write_json(path, result)
                    await asyncio.sleep(0.5)
                if not result.get("hasMoreData"):
                    break
    log("kartaview-meta", state="complete")


def kartaview_rows(root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows, reasons, seen = [], Counter(), set()
    for path in sorted((root / "kartaview" / "nearby").glob("*.json")):
        for p in read_json(path).get("data", []):
            pid = str(p.get("id"))
            if pid in seen:
                continue
            seen.add(pid)
            try:
                lat, lng = float(p["lat"]), float(p["lng"])
                shot = datetime.fromisoformat(p["shotDate"]).replace(tzinfo=UTC)
            except (KeyError, TypeError, ValueError):
                reasons["bad_metadata"] += 1
                continue
            if not in_area(lng, lat):
                reasons["outside_polygon"] += 1
                continue
            if str(p.get("projection", "PLANE")).upper() not in ("PLANE", ""):
                reasons["not_perspective"] += 1
                continue
            url = p.get("imageLthUrl") or p.get("fileurlLTh") or p.get("imageProcUrl")
            if not url:
                reasons["no_image_url"] += 1
                continue
            try:
                heading = float(p["heading"]) % 360 if p.get("heading") not in (None, "") else None
            except ValueError:
                heading = None
            captured = int(shot.timestamp() * 1000)
            reasons["ok"] += 1
            rows.append(
                {
                    "source": "kartaview",
                    "source_id": pid,
                    "ref_id": f"kartaview-{pid}",
                    "sequence": f"kartaview-{p.get('sequenceId')}",
                    "creator": p.get("user"),
                    "captured_at": captured,
                    "capture_year": year_of(captured),
                    "lat": lat,
                    "lng": lng,
                    "location_source": "kartaview_gps",
                    "heading": heading,
                    "gps_accuracy_m": float(p["gpsAccuracy"]) if p.get("gpsAccuracy") else None,
                    "is_pano": False,
                    "license_url": CC_BY_SA_4,
                    "source_url": f"https://kartaview.org/details/{p.get('sequenceId')}/"
                    f"{p.get('sequenceIndex')}",
                    "download_url": url,
                }
            )
    return rows, reasons


# ---------------------------------------------------------------- plan


def protected_sequences() -> set[str]:
    """Earlier street-eval query sequences stay out of every gallery."""
    if not LEGACY_GALLERY.exists():
        return set()
    return set(read_json(LEGACY_GALLERY).get("protected_sequences", []))


def plan(root: Path) -> None:
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "build_parameters": build_parameters(),
    }
    mly = mapillary_rows(root)
    inside = [r for r in mly if in_area(r["lng"], r["lat"])]
    protected = protected_sequences()
    unprotected = [r for r in inside if r["sequence"] not in protected]
    thinned = thin_sequences(unprotected)
    per_year = lambda rows: dict(sorted(Counter(r["capture_year"] for r in rows).items()))  # noqa: E731
    report["mapillary"] = {
        "metadata_rows": len(mly),
        "inside_polygon": len(inside),
        "protected_eval_sequences_excluded": len(inside) - len(unprotected),
        "before_thinning": len(unprotected),
        "after_thinning": len(thinned),
        "panoramas_kept": sum(r["is_pano"] for r in thinned),
        "before_thinning_by_year": per_year(unprotected),
        "after_thinning_by_year": per_year(thinned),
        "location_source": dict(Counter(r["location_source"] for r in inside)),
        "thinning_rule": f"same sequence, <= {THIN_DISTANCE_M} m and <= {THIN_HEADING_DEG} deg "
        "of an already-kept frame (capture order); panoramas always kept",
    }
    commons, commons_reasons = (
        commons_rows(root) if (root / "commons").exists() else ([], Counter())
    )
    karta, karta_reasons = (
        kartaview_rows(root) if (root / "kartaview").exists() else ([], Counter())
    )
    report["commons"] = {"accepted": len(commons), "decisions": dict(commons_reasons)}
    report["kartaview"] = {"accepted": len(karta), "decisions": dict(karta_reasons)}
    walks = []
    for approved in sorted((root / "self_collected").glob("*/approved.json")):
        walks += read_json(approved)["references"]
    if any(not r.get("blurred") for r in walks):
        raise ValueError("Self-collected frames must be blurred before they enter the gallery")
    report["self_collected"] = {"approved_frames": len(walks)}
    candidates = thinned + commons + karta + walks
    bad = [r for r in candidates if provenance.problems(r)]
    if bad:
        raise ValueError(
            f"{len(bad)} candidates violate the provenance contract, e.g. "
            f"{provenance.problems(bad[0])}"
        )
    kept, dev_report = leakage.guard_dev(candidates)
    write_json(root / "leakage-dev.json", dev_report)
    kept, sealed_counts = leakage.guard_sealed(kept)
    report["leakage_guard"] = {
        "dev": {k: v for k, v in dev_report.items() if k != "exclusions"},
        "test_sealed_counts_only": sealed_counts,
    }
    report["planned"] = {
        "total": len(kept),
        "by_source": dict(Counter(r["source"] for r in kept)),
        "by_location_source": dict(Counter(r["location_source"] for r in kept)),
    }
    write_json(root / "plan.json", {"references": kept})
    write_json(root / "plan-report.json", report)
    log("plan", **{k: v for k, v in report.items() if k != "created_at"})


# ---------------------------------------------------------------- download


def image_path(root: Path, ref: dict[str, Any]) -> Path:
    return root / "images" / ref["source"] / f"{ref['source_id']}.jpg"


def legacy_images() -> dict[str, Path]:
    """Previously downloaded Mapillary thumbnails, reused instead of re-downloading."""
    found = {}
    for folder in Path("data/linkoping_dense").glob("*/originals"):
        for p in folder.glob("*.jpg"):
            found[p.stem] = p
    return found


def valid_jpeg(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - any decode failure means re-download.
        return False


async def download(root: Path, concurrency: int = 12) -> None:
    token = get_settings().mapillary_access_token.strip()
    refs = read_json(root / "plan.json")["references"]
    legacy = legacy_images()
    todo = [r for r in refs if r["source"] != "self_collected" and not image_path(root, r).exists()]
    log("download", planned=len(refs), remaining=len(todo))
    failures: Counter[str] = Counter()
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        headers={"User-Agent": UA}, timeout=90, follow_redirects=True
    ) as client:

        async def fetch(ref: dict[str, Any], url: str) -> None:
            path = image_path(root, ref)
            async with semaphore:
                r = await get_with_backoff(client, url)
            if r.status_code != 200 or not r.content:
                failures[f"http_{r.status_code}"] += 1
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            try:
                tmp.write_bytes(r.content if ref.get("is_pano") else store_resized(r.content))
            except OSError:
                failures["undecodable"] += 1
                return
            if valid_jpeg(tmp):
                tmp.replace(path)
            else:
                tmp.unlink(missing_ok=True)
                failures["invalid_image"] += 1

        async def mapillary_batch(batch: list[dict[str, Any]]) -> None:
            fresh = []
            for ref in batch:
                old = legacy.get(ref["source_id"])
                if old and not ref["is_pano"]:
                    image_path(root, ref).parent.mkdir(parents=True, exist_ok=True)
                    image_path(root, ref).write_bytes(store_resized(old.read_bytes()))
                else:
                    fresh.append(ref)
            if not fresh:
                return
            # Signed thumbnail URLs expire, so fetch them just before downloading.
            r = await get_with_backoff(
                client,
                "https://graph.mapillary.com/images",
                params={
                    "image_ids": ",".join(f["source_id"] for f in fresh),
                    "fields": "id,thumb_1024_url,thumb_2048_url",
                },
                headers={"Authorization": f"OAuth {token}"},
            )
            r.raise_for_status()
            urls = {d["id"]: d for d in r.json().get("data", [])}
            jobs = []
            for ref in fresh:
                found = urls.get(ref["source_id"], {})
                url = (found.get("thumb_2048_url") if ref["is_pano"] else None) or found.get(
                    "thumb_1024_url"
                )
                if url:
                    jobs.append(fetch(ref, url))
                else:
                    failures["no_url"] += 1
            await asyncio.gather(*jobs)

        mly = [r for r in todo if r["source"] == "mapillary"]
        batches = [mly[i : i + 100] for i in range(0, len(mly), 100)]
        # Four URL batches in flight hide the ~6 s Graph API latency per batch.
        for start in range(0, len(batches), 4):
            if free_bytes(root) < MIN_FREE_BYTES:
                log("download", state="stopped_low_disk", free_gb=free_bytes(root) / 2**30)
                return
            await asyncio.gather(*[mapillary_batch(b) for b in batches[start : start + 4]])
            if start % 40 == 0:
                log(
                    "download",
                    mapillary_batches_done=min(start + 4, len(batches)),
                    of=len(batches),
                    failures=dict(failures),
                    free_gb=round(free_bytes(root) / 2**30, 1),
                )
        others = [r for r in todo if r["source"] in ("wikimedia_commons", "kartaview")]
        for start in range(0, len(others), 20):
            if free_bytes(root) < MIN_FREE_BYTES:
                log("download", state="stopped_low_disk")
                return
            await asyncio.gather(*[fetch(r, r["download_url"]) for r in others[start : start + 20]])
            await asyncio.sleep(1)  # Commons/KartaView: gentle pacing.
    log("download", state="complete", failures=dict(failures))


# ---------------------------------------------------------------- panorama crops


def crops(root: Path) -> None:
    from PIL import Image

    refs = read_json(root / "plan.json")["references"]
    out = []
    for ref in refs:
        if not ref.get("is_pano"):
            continue
        src = image_path(root, ref)
        if not src.exists():
            continue
        folder = root / "images" / "mapillary_pano_crops"
        folder.mkdir(parents=True, exist_ok=True)
        targets = [folder / f"{ref['source_id']}_{k}.jpg" for k in range(panorama.CROP_COUNT)]
        headings = panorama.crop_headings(ref["heading"])
        if not all(t.exists() for t in targets):
            with Image.open(src) as pano:
                for target, (crop, _, _) in zip(
                    targets, panorama.split(pano, ref["heading"]), strict=True
                ):
                    crop.save(target, quality=92)
        for k, (target, (offset, heading)) in enumerate(zip(targets, headings, strict=True)):
            out.append(
                {
                    **ref,
                    "ref_id": f"{ref['ref_id']}-crop{k}",
                    "is_pano": False,
                    "pano_crop": {
                        "index": k,
                        "yaw_offset_deg": offset,
                        "fov_deg": panorama.FOV_DEG,
                    },
                    "heading": heading,
                    "heading_source": "derived_from_panorama",
                    "path": str(target.relative_to(root)),
                }
            )
    write_json(root / "pano-crops.json", {"references": out})
    log("crops", panoramas=sum(r.get("is_pano", False) for r in refs), crops=len(out))


# ---------------------------------------------------------------- quality filter


def filter_stage(root: Path) -> None:
    from PIL import Image

    refs = [r for r in read_json(root / "plan.json")["references"] if not r.get("is_pano")]
    for r in refs:
        if r["source"] != "self_collected":
            r["path"] = str(image_path(root, r).relative_to(root))
    refs += (
        read_json(root / "pano-crops.json")["references"]
        if (root / "pano-crops.json").exists()
        else []
    )
    cache_path = root / "quality-cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    dropped: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    # Newest first, so a near-duplicate pair keeps the more recent capture.
    for i, r in enumerate(sorted(refs, key=lambda r: (-r["captured_at"], r["ref_id"]))):
        path = root / r["path"]
        if not path.exists():
            dropped["not_downloaded"] += 1
            continue
        if r["ref_id"] not in cache:
            with Image.open(path) as im:
                cache[r["ref_id"]] = {**signature(path), **quality.measures(im)}
            if i % 2000 == 0:
                write_json(cache_path, cache)
        q = cache[r["ref_id"]]
        rule = quality.image_rule(q)
        if rule:
            dropped[rule] += 1
            continue
        cy = int(r["lat"] * 111_320 // quality.DUPLICATE_RADIUS_M)
        cx = int(
            r["lng"] * 111_320 * math.cos(math.radians(r["lat"])) // quality.DUPLICATE_RADIUS_M
        )
        near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for k in cells[(cx + dx, cy + dy)]]
        sig = {
            **{k: q[k] for k in ("sha256", "phash", "dhash")},
            "source": r["source"],
            "lat": r["lat"],
            "lng": r["lng"],
        }
        if quality.is_duplicate(sig, near):
            dropped["near_duplicate"] += 1
            continue
        row = {**r, **sig, **{k: q[k] for k in ("laplacian_var", "edge_share")}}
        cells[(cx, cy)].append(row)
        kept.append(row)
    write_json(cache_path, cache)
    # Pixel-level guard against eval photos (hash metadata only; sealed for the test set).
    kept, dev_pixels = leakage.guard_dev(kept)
    kept, sealed_pixels = leakage.guard_sealed(kept)
    report = {
        "input": len(refs),
        "kept": len(kept),
        "dropped_by_rule": dict(dropped),
        "build_parameters": build_parameters(),
        "indoor_rule": "Commons keyword rule applied at plan stage; no image classifier",
        "post_download_leakage": {
            "dev": {k: v for k, v in dev_pixels.items() if k != "exclusions"},
            "test_sealed_counts_only": sealed_pixels,
        },
    }
    write_json(root / "filtered.json", {"references": kept})
    write_json(root / "filter-report.json", report)
    log("filter", **report)


# ---------------------------------------------------------------- encode + index


def encode(root: Path, device: str, batch_size: int) -> None:
    from swespy.model.megaloc import MegaLocEmbedder

    refs = read_json(root / "filtered.json")["references"]
    embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), device)
    chunk = 512
    for start in range(0, len(refs), chunk):  # The descriptor cache is the checkpoint.
        part = refs[start : start + chunk]
        embedder.embed([root / r["path"] for r in part], [r["sha256"] for r in part], batch_size)
        log("encode", done=start + len(part), of=len(refs))
    write_json(root / "encoder-identity.json", embedder.identity)


def index(root: Path) -> None:
    from swespy.model.megaloc import file_hash

    refs = [{**r, "image_id": r["ref_id"]} for r in read_json(root / "filtered.json")["references"]]
    identity = read_json(root / "encoder-identity.json")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(get_settings().megaloc_root) / "descriptors" / key
    missing = [r for r in refs if not (cache / f"{r['sha256']}.npy").exists()]
    if missing:
        raise ValueError(f"{len(missing)} references are not encoded yet; run encode")
    digest = hashlib.sha256("\n".join(r["sha256"] for r in refs).encode()).hexdigest()
    bad = [r for r in refs if provenance.problems(r)]
    if bad:
        raise ValueError(f"{len(bad)} references violate the provenance contract")
    write_json(
        root / "gallery-index.json",
        {
            "format": "descriptor_cache_v1",
            "dimension": 8448,
            "descriptor_dir": str(cache),
            "references_in_index_order": refs,
            "index_sha256": digest,
            "model_identity": identity,
            "created_at": datetime.now(UTC).isoformat(),
            "filter_report_sha256": file_hash(root / "filter-report.json"),
            "plan_report_sha256": file_hash(root / "plan-report.json"),
            "build_parameters": build_parameters(),
        },
    )
    log("index", references=len(refs))


# ---------------------------------------------------------------- coverage


def coverage(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    refs = read_json(root / "gallery-index.json")["references_in_index_order"]
    lats, lngs = np.array([r["lat"] for r in refs]), np.array([r["lng"] for r in refs])
    dev = read_json(leakage.DEV_MANIFEST)["queries"]
    old = read_json(Path(get_settings().gallery_index))["references_in_index_order"]
    old = [r for r in old if in_area(r["lng"], r["lat"])]
    olats, olngs = np.array([r["lat"] for r in old]), np.array([r["lng"] for r in old])
    nearest_new = [float(haversine_m(q["lat"], q["lng"], lats, lngs).min()) for q in dev]
    nearest_old = [float(haversine_m(q["lat"], q["lng"], olats, olngs).min()) for q in dev]

    def dist(values: list[float]) -> dict[str, Any]:
        s = sorted(values)
        return {
            "median_m": statistics.median(s),
            "p25_m": s[len(s) // 4],
            "p75_m": s[3 * len(s) // 4],
            **{f"within_{r}m": sum(v <= r for v in s) for r in (25, 50, 100, 500)},
        }

    # 250 m density grid over the polygon (cells outside the polygon are masked).
    west, south, east, north = polygon_bbox()
    cell = 250
    dlat = cell / 111_320
    dlng = cell / (111_320 * math.cos(math.radians((south + north) / 2)))
    nx, ny = int((east - west) / dlng) + 1, int((north - south) / dlat) + 1
    grid = np.zeros((ny, nx))
    for la, ln in zip(lats, lngs, strict=True):
        grid[int((la - south) / dlat), int((ln - west) / dlng)] += 1
    mask = np.array(
        [
            [not in_area(west + (i + 0.5) * dlng, south + (j + 0.5) * dlat) for i in range(nx)]
            for j in range(ny)
        ]
    )
    inside_cells = int((~mask).sum())
    empty = int(((grid == 0) & ~mask).sum())
    fig, ax = plt.subplots(figsize=(10, 9))
    shown = np.ma.array(np.log10(grid + 1), mask=mask)
    im = ax.imshow(
        shown,
        origin="lower",
        extent=(west, east, south, north),
        cmap="viridis",
        aspect=1 / math.cos(math.radians(58.41)),
    )
    poly = list(LINKOPING_AREA) + [LINKOPING_AREA[0]]
    ax.plot([p[0] for p in poly], [p[1] for p in poly], color="crimson", lw=1.5)
    fig.colorbar(im, ax=ax, label="log10(references + 1) per 250 m cell")
    ax.set_title(
        f"Gallery v2 reference density ({len(refs):,} references; "
        f"{empty}/{inside_cells} cells empty)"
    )
    fig.savefig(root / "coverage.png", dpi=130, bbox_inches="tight")
    write_json(
        root / "density-grid.json",
        {
            "cell_m": cell,
            "west": west,
            "south": south,
            "dlat": dlat,
            "dlng": dlng,
            "counts": grid.astype(int).tolist(),
            "outside_mask": mask.tolist(),
        },
    )
    report = {
        "references": len(refs),
        "by_source": dict(Counter(r["source"] for r in refs)),
        "by_location_source": dict(Counter(r["location_source"] for r in refs)),
        "by_capture_year": dict(sorted(Counter(r["capture_year"] for r in refs).items())),
        "pre_2018": sum(r["capture_year"] < 2018 for r in refs),
        "cells_250m_inside": inside_cells,
        "cells_250m_empty": empty,
        "dev_nearest_reference": {
            "gallery_v1_in_polygon": dist(nearest_old),
            "gallery_v2": dist(nearest_new),
        },
    }
    write_json(root / "coverage-report.json", report)
    log("coverage", **report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stage",
        choices=[
            "mapillary-meta",
            "commons-meta",
            "kartaview-meta",
            "plan",
            "download",
            "crops",
            "filter",
            "encode",
            "index",
            "coverage",
        ],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    if stage == "mapillary-meta":
        asyncio.run(mapillary_meta(args.root))
    elif stage == "commons-meta":
        asyncio.run(commons_meta(args.root))
    elif stage == "kartaview-meta":
        asyncio.run(kartaview_meta(args.root))
    elif stage == "plan":
        plan(args.root)
    elif stage == "download":
        asyncio.run(download(args.root))
    elif stage == "crops":
        crops(args.root)
    elif stage == "filter":
        filter_stage(args.root)
    elif stage == "encode":
        encode(args.root, args.device, args.batch_size)
    elif stage == "index":
        index(args.root)
    else:
        coverage(args.root)


if __name__ == "__main__":
    main()
