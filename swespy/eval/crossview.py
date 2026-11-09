"""Cross-view pair feasibility in the training cities (metadata only; no training images).

    python -m eval.crossview meta --city Norrköping      # Mapillary tiles + per-sequence camera
    python -m eval.crossview commons --city Norrköping   # geotagged Commons (allowed licences)
    python -m eval.crossview estimate                    # viewpoint classes + candidate pairs
    python -m eval.crossview review                      # 256 px thumbnail sheets (rule check)

Training cities and polygons: swespy/cities.py (SCB tätorter 2023). Nothing inside the
Linköping polygon, and nothing within 1 km of a dev or test photo (test: sealed, counts
only). Rules: docs/experiments/08_crossview_pairs.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from eval import densify, leakage
from eval.geo_eval import haversine_m
from swespy.config import LINKOPING_AREA, TRAINING_CITIES, get_settings, in_area

ROOT = Path("data/crossview")
EXCLUSION_RADIUS_M = 1000.0
CLASSES = ("car", "bike", "pedestrian", "panorama", "unknown")
CAR_MEDIAN_MPS = 7.0
CAR_P90_MPS = 12.0
BIKE_MIN_MPS = 2.5
MIN_STEPS = 5
PAIR_RADIUS_M = 25.0
PAIR_HEADING_DEG = 60.0
COMMONS_RADIUS_M = 2000
COMMONS_SPACING_M = 2800
PHONE_MAKES = (
    "apple",
    "samsung",
    "google",
    "oneplus",
    "xiaomi",
    "huawei",
    "sony",
    "motorola",
    "nokia",
    "oppo",
    "fairphone",
    "lg",
    "realme",
    "vivo",
    "honor",
    "nothing",
)
SPHERICAL_TYPES = ("equirectangular", "spherical")


# ---------------------------------------------------------------- viewpoint rules


def sequence_speeds(frames: list[dict[str, Any]]) -> list[float]:
    """Speeds (m/s) between consecutive frames in capture order; skips gaps > 60 s."""
    ordered = sorted(frames, key=lambda f: f["captured_at"])
    speeds = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        dt = (b["captured_at"] - a["captured_at"]) / 1000
        if 0 < dt <= 60:
            d = float(
                haversine_m(a["lat"], a["lng"], np.array([b["lat"]]), np.array([b["lng"]]))[0]
            )
            speeds.append(d / dt)
    return speeds


def classify(frames: list[dict[str, Any]], camera: dict[str, Any] | None = None) -> dict[str, Any]:
    """Viewpoint class of one sequence from metadata only (rules fixed before review)."""
    camera = camera or {}
    make = str(camera.get("make") or "").strip().lower()
    camera_type = str(camera.get("camera_type") or "").lower()
    speeds = sequence_speeds(frames)
    median = statistics.median(speeds) if speeds else None
    p90 = float(np.percentile(speeds, 90)) if speeds else None
    info = {
        "frames": len(frames),
        "steps": len(speeds),
        "median_mps": median,
        "p90_mps": p90,
        "make": make or None,
        "camera_type": camera_type or None,
    }
    if any(f.get("is_pano") for f in frames) or camera_type in SPHERICAL_TYPES:
        return {**info, "class": "panorama", "rule": "panorama flag or spherical camera"}
    if len(speeds) < MIN_STEPS:
        return {**info, "class": "unknown", "rule": f"fewer than {MIN_STEPS} timed steps"}
    if median > CAR_MEDIAN_MPS or p90 > CAR_P90_MPS:
        return {
            **info,
            "class": "car",
            "rule": f"median > {CAR_MEDIAN_MPS} or p90 > {CAR_P90_MPS} m/s",
        }
    if median > BIKE_MIN_MPS:
        return {**info, "class": "bike", "rule": f"{BIKE_MIN_MPS} < median <= {CAR_MEDIAN_MPS} m/s"}
    return {
        **info,
        "class": "pedestrian",
        "rule": f"median <= {BIKE_MIN_MPS} m/s"
        + (" (phone make)" if make.startswith(PHONE_MAKES) else ""),
    }


# ---------------------------------------------------------------- guards


def outside_linkoping(lat: float, lng: float) -> bool:
    return not in_area(lng, lat, LINKOPING_AREA)


def guard_eval_radius(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop rows within 1 km of any dev photo (reported) or test photo (sealed counts)."""

    def near(photos: list[dict[str, Any]]) -> set[int]:
        if not rows or not photos:
            return set()
        lats = np.array([r["lat"] for r in rows])
        lngs = np.array([r["lng"] for r in rows])
        hit: set[int] = set()
        for p in photos:
            hit |= set(
                np.where(haversine_m(p["lat"], p["lng"], lats, lngs) <= EXCLUSION_RADIUS_M)[
                    0
                ].tolist()
            )
        return hit

    dev = leakage.eval_photos(json.loads(leakage.DEV_MANIFEST.read_text()))
    test_manifest = json.loads(leakage.TEST_MANIFEST.read_text())
    test = leakage.eval_photos(test_manifest, set(test_manifest["fresh_commons_ids"]))
    dev_hits, test_hits = near(dev), near(test)
    kept = [r for i, r in enumerate(rows) if i not in dev_hits and i not in test_hits]
    return kept, {
        "dev_excluded": len(dev_hits),
        "test_sealed_excluded_count_only": len(test_hits - dev_hits),
    }


# ---------------------------------------------------------------- Mapillary metadata


def city_dir(city: str) -> Path:
    return ROOT / hashlib.sha1(city.encode()).hexdigest()[:8]


def city_tiles(polygon: tuple[tuple[float, float], ...]) -> list[tuple[int, int]]:
    lngs, lats = [p[0] for p in polygon], [p[1] for p in polygon]
    n = 2**densify.TILE_ZOOM

    def tx(lng: float) -> int:
        return int((lng + 180) / 360 * n)

    def ty(lat: float) -> int:
        return int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

    return [
        (x, y)
        for x in range(tx(min(lngs)), tx(max(lngs)) + 1)
        for y in range(ty(max(lats)), ty(min(lats)) + 1)
    ]


async def meta(city: str) -> None:
    polygon = TRAINING_CITIES[city]["polygon"]
    token = get_settings().mapillary_access_token.strip()
    folder = city_dir(city)
    (folder / "tiles").mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps({"city": city}))
    rows = []
    async with httpx.AsyncClient(timeout=60) as client:
        for x, y in city_tiles(polygon):  # type: ignore[arg-type]
            path = folder / "tiles" / f"{densify.TILE_ZOOM}_{x}_{y}.json"
            if not path.exists():
                r = await densify.get_with_backoff(
                    client,
                    f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
                    f"{densify.TILE_ZOOM}/{x}/{y}",
                    params={"access_token": token},
                )
                r.raise_for_status()
                densify.write_json(path, densify.decode_tile(r.content, x, y, extra=True))
                await asyncio.sleep(0.2)
            for row in densify.read_json(path):
                if in_area(row["tile_lng"], row["tile_lat"], polygon) and outside_linkoping(
                    row["tile_lat"], row["tile_lng"]
                ):
                    rows.append(row)
        by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_seq[r["sequence"]].append(r)
        # One camera lookup per sequence (the middle frame), batched 100 per request.
        sample = {
            seq: sorted(f, key=lambda r: r["captured_at"])[len(f) // 2]["id"]
            for seq, f in by_seq.items()
        }
        cam_path = folder / "cameras.json"
        cameras = densify.read_json(cam_path) if cam_path.exists() else {}
        todo = [s for s in sample if s not in cameras]
        headers = {"Authorization": f"OAuth {token}"}
        sem = asyncio.Semaphore(8)

        async def lookup(batch: list[str]) -> None:
            async with sem:
                r = await densify.get_with_backoff(
                    client,
                    "https://graph.mapillary.com/images",
                    params={
                        "image_ids": ",".join(sample[s] for s in batch),
                        "fields": "id,make,model,camera_type,creator",
                    },
                    headers=headers,
                )
            r.raise_for_status()
            found = {d["id"]: d for d in r.json().get("data", [])}
            for s in batch:
                d = found.get(sample[s], {})
                cameras[s] = {
                    "make": d.get("make"),
                    "model": d.get("model"),
                    "camera_type": d.get("camera_type"),
                    "creator": (d.get("creator") or {}).get("username"),
                }

        for i in range(0, len(todo), 800):
            await asyncio.gather(
                *[lookup(todo[j : j + 100]) for j in range(i, min(i + 800, len(todo)), 100)]
            )
            densify.write_json(cam_path, cameras)
    densify.write_json(folder / "images.json", rows)
    densify.log("crossview-meta", city=city, images=len(rows), sequences=len(by_seq))


# ---------------------------------------------------------------- Commons


async def commons(city: str) -> None:
    polygon = TRAINING_CITIES[city]["polygon"]
    folder = city_dir(city) / "commons"
    (folder / "geosearch").mkdir(parents=True, exist_ok=True)
    (folder / "info").mkdir(parents=True, exist_ok=True)
    lngs, lats = [p[0] for p in polygon], [p[1] for p in polygon]
    pageids: set[str] = set()
    async with httpx.AsyncClient(
        headers={"User-Agent": densify.UA}, timeout=60, follow_redirects=True
    ) as client:
        # Coarse grid (2 km radius, 2.8 km spacing); saturated searches split into four,
        # so counts stay complete with far fewer requests than a 500 m grid.
        dlat = COMMONS_SPACING_M / 111_320
        dlng = COMMONS_SPACING_M / (111_320 * math.cos(math.radians(statistics.fmean(lats))))
        queue = []
        lat = min(lats) + dlat / 2
        while lat < max(lats) + dlat / 2:
            lng = min(lngs) + dlng / 2
            while lng < max(lngs) + dlng / 2:
                if any(
                    in_area(lng + sx * dlng / 2, lat + sy * dlat / 2, polygon)
                    for sx in (-1, 0, 1)
                    for sy in (-1, 0, 1)
                ):
                    queue.append((lat, lng, COMMONS_RADIUS_M))
                lng += dlng
            lat += dlat
        while queue:
            lat, lng, radius = queue.pop()
            path = folder / "geosearch" / f"{lat:.5f}_{lng:.5f}_{radius}.json"
            if path.exists():
                found = densify.read_json(path)
            else:
                found = (
                    (
                        await densify.commons_request(
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
                densify.write_json(path, found)
            if len(found) >= 500 and radius > 100:
                off = radius / 2 / 111_320
                offl = off / math.cos(math.radians(lat))
                queue += [
                    (lat + sy * off, lng + sx * offl, radius // 2)
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                ]
            pageids |= {str(i["pageid"]) for i in found}
        ordered = sorted(pageids)
        for start in range(0, len(ordered), 50):
            batch = ordered[start : start + 50]
            path = (
                folder
                / "info"
                / f"{hashlib.sha256('|'.join(batch).encode()).hexdigest()[:24]}.json"
            )
            if not path.exists():
                data = await densify.commons_request(
                    client,
                    {
                        "pageids": "|".join(batch),
                        "prop": "imageinfo",
                        "iiprop": "url|mime|size|extmetadata",
                        "iiurlwidth": 640,
                    },
                )
                densify.write_json(path, list(data.get("query", {}).get("pages", {}).values()))
    densify.log("crossview-commons", city=city, pages=len(ordered))


def commons_rows(city: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    polygon = TRAINING_CITIES[city]["polygon"]
    rows, reasons, seen = [], Counter(), set()
    for path in sorted((city_dir(city) / "commons" / "info").glob("*.json")):
        for page in densify.read_json(path):
            pid = str(page.get("pageid"))
            if pid in seen:
                continue
            seen.add(pid)
            row, reason = densify.commons_row(page, polygon=polygon)
            reasons[reason] += 1
            if row:
                rows.append(row)
    return rows, reasons


# ---------------------------------------------------------------- estimate


def angle_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    d = abs(a - b) % 360
    return min(d, 360 - d)


def estimate_city(city: str) -> dict[str, Any]:
    folder = city_dir(city)
    images = densify.read_json(folder / "images.json")
    cameras = densify.read_json(folder / "cameras.json")
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in images:
        by_seq[r["sequence"]].append({**r, "lat": r["tile_lat"], "lng": r["tile_lng"]})
    seq_class = {s: classify(f, cameras.get(s)) for s, f in by_seq.items()}
    rows = [{**f, "class": seq_class[s]["class"]} for s, fs in by_seq.items() for f in fs]
    rows, guard = guard_eval_radius(rows)
    cars = [r for r in rows if r["class"] == "car"]
    cell = PAIR_RADIUS_M / 111_320
    grid: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for c in cars:
        grid[
            (int(c["lat"] / cell), int(c["lng"] / (cell / math.cos(math.radians(c["lat"])))))
        ].append(c)

    def car_neighbours(r: dict[str, Any], need_heading: bool) -> int:
        cy = int(r["lat"] / cell)
        cx = int(r["lng"] / (cell / math.cos(math.radians(r["lat"]))))
        n = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for c in grid.get((cy + dy, cx + dx), []):
                    if c["sequence"] == r.get("sequence"):
                        continue
                    d = float(
                        haversine_m(r["lat"], r["lng"], np.array([c["lat"]]), np.array([c["lng"]]))[
                            0
                        ]
                    )
                    if d > PAIR_RADIUS_M:
                        continue
                    if need_heading:
                        h = angle_diff(r.get("compass_angle"), c.get("compass_angle"))
                        if h is None or h >= PAIR_HEADING_DEG:
                            continue
                    n += 1
        return n

    pairs: Counter[str] = Counter()
    images_with_pair: Counter[str] = Counter()
    for r in rows:
        if r["class"] in ("pedestrian", "bike"):
            n = car_neighbours(r, need_heading=True)
        elif r["class"] == "panorama":
            n = car_neighbours(r, need_heading=False)
        else:
            continue
        pairs[r["class"]] += n
        images_with_pair[r["class"]] += n > 0
    com, reasons = commons_rows(city) if (folder / "commons").exists() else ([], Counter())
    com = [c for c in com if outside_linkoping(c["lat"], c["lng"])]
    com, com_guard = guard_eval_radius(com)
    commons_with_pair = sum(car_neighbours(c, need_heading=False) > 0 for c in com)
    seq_counts = Counter(v["class"] for v in seq_class.values())
    img_counts = Counter(r["class"] for r in rows)
    return {
        "city": city,
        "images": len(rows),
        "sequences": len(seq_class),
        "sequences_by_class": {c: seq_counts.get(c, 0) for c in CLASSES},
        "images_by_class": {c: img_counts.get(c, 0) for c in CLASSES},
        "candidate_pairs": dict(pairs),
        "non_car_images_with_a_car_pair": dict(images_with_pair),
        "commons_allowed": len(com),
        "commons_with_car_within_25m": commons_with_pair,
        "commons_decisions": dict(reasons),
        "eval_guard": {"mapillary": guard, "commons": com_guard},
    }


def estimate() -> Path:
    out = {
        "rules": {
            "car": f"median > {CAR_MEDIAN_MPS} m/s or p90 > {CAR_P90_MPS} m/s",
            "bike": f"{BIKE_MIN_MPS} < median <= {CAR_MEDIAN_MPS} m/s",
            "pedestrian": f"median <= {BIKE_MIN_MPS} m/s",
            "panorama": "is_pano or spherical/equirectangular camera",
            "unknown": f"< {MIN_STEPS} timed steps (gaps > 60 s ignored)",
            "pair": f"non-car image with a car image (other sequence) within "
            f"{PAIR_RADIUS_M:.0f} m and heading diff < {PAIR_HEADING_DEG:.0f} deg; "
            "panorama and Commons: distance only (no reliable heading)",
        },
        "cities": [
            estimate_city(c) for c in TRAINING_CITIES if (city_dir(c) / "images.json").exists()
        ],
    }
    path = ROOT / "estimate.json"
    densify.write_json(path, out)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return path


# ---------------------------------------------------------------- reliability review


def review(per_class: int = 50, seed: int = 0) -> list[Path]:
    """Contact sheets of 256 px thumbnails for random sequences per class (review only).

    Thumbnails are fetched in memory and embedded in the sheet; no image is stored for
    training. One middle frame per sequence.
    """
    import io

    from PIL import Image, ImageDraw

    token = get_settings().mapillary_access_token.strip()
    pool: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for city in TRAINING_CITIES:
        folder = city_dir(city)
        if not (folder / "images.json").exists():
            continue
        cameras = densify.read_json(folder / "cameras.json")
        by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in densify.read_json(folder / "images.json"):
            by_seq[r["sequence"]].append({**r, "lat": r["tile_lat"], "lng": r["tile_lng"]})
        for s, f in by_seq.items():
            c = classify(f, cameras.get(s))["class"]
            mid = sorted(f, key=lambda r: r["captured_at"])[len(f) // 2]["id"]
            pool[c].append((city, s, mid))
    sheets = []
    for cls in ("car", "bike", "pedestrian", "panorama"):
        chosen = random.Random(seed).sample(pool[cls], min(per_class, len(pool[cls])))
        ids = [m for _, _, m in chosen]
        urls = {}
        for i in range(0, len(ids), 100):
            r = httpx.get(
                "https://graph.mapillary.com/images",
                params={"image_ids": ",".join(ids[i : i + 100]), "fields": "id,thumb_256_url"},
                headers={"Authorization": f"OAuth {token}"},
                timeout=60,
            )
            r.raise_for_status()
            urls.update({d["id"]: d.get("thumb_256_url") for d in r.json().get("data", [])})
        sheet = Image.new("RGB", (10 * 256, 5 * 200), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (_city, _seq, mid) in enumerate(chosen):
            if not urls.get(mid):
                continue
            with Image.open(io.BytesIO(httpx.get(urls[mid], timeout=60).content)) as im:
                im = im.convert("RGB")
                im.thumbnail((256, 192))
                sheet.paste(im, ((k % 10) * 256, (k // 10) * 200))
            draw.text(((k % 10) * 256 + 3, (k // 10) * 200 + 3), str(k), fill="yellow")
        path = ROOT / f"review_{cls}.jpg"
        sheet.save(path, quality=80)
        densify.write_json(
            ROOT / f"review_{cls}.json",
            [{"n": k, "city": c, "sequence": s, "image": m} for k, (c, s, m) in enumerate(chosen)],
        )
        sheets.append(path)
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["meta", "commons", "estimate", "review"])
    parser.add_argument("--city", choices=list(TRAINING_CITIES))
    args = parser.parse_args()
    if args.action in ("meta", "commons") and not args.city:
        parser.error("--city is required")
    if args.action == "meta":
        asyncio.run(meta(args.city))
    elif args.action == "commons":
        asyncio.run(commons(args.city))
    elif args.action == "estimate":
        estimate()
    else:
        print(review())


if __name__ == "__main__":
    main()
