"""Static data for the demo web app (swespy/web/public/data). Run when the gallery changes.

- coverage.geojson  street stretches with a gallery reference within 25 m (from the gap
                    plan's 20 m sample points), for the "reference coverage" layer.
- places.json       reverse-geocoding index: points every ~25 m along named OSM ways plus
                    district names (one Overpass request), both ODbL, © OpenStreetMap.
- library.json      reference counts per source and the area polygon.

No photo, coordinate of a user upload or Street View data is involved.
Usage: .venv-benchmark/bin/python scripts/build_web_data.py
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import httpx

from eval.gsv_plan import M_PER_DEG_LAT, m_per_deg_lng, sample_line
from swespy.config import LINKOPING_AREA, in_area

OUT = Path("web/public/data")
GALLERY_INDEX = Path("data/gallery_v2/gallery-index-historic.json")
SAMPLE_POINTS = Path("data/gsv/sample_points.json")
NAMED_WAYS = Path("data/gallery_v2/osm-walkable.json")
OVERPASS = ("https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter")
CHAIN_MAX_M = 25.0
PLACE_STEP_M = 25.0


def dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot((a[0] - b[0]) * M_PER_DEG_LAT, (a[1] - b[1]) * m_per_deg_lng(a[0]))


def coverage() -> dict:
    points = json.loads(SAMPLE_POINTS.read_text())["points"]
    lines, current, prev = [], [], None
    for p in points:  # Stored in way order, 20 m apart along each way.
        referenced = p["gap"] in ("covered", "old-only")
        here = (p["lat"], p["lng"])
        joined = (prev is not None and prev["way_id"] == p["way_id"]
                  and dist_m((prev["lat"], prev["lng"]), here) <= CHAIN_MAX_M)
        if referenced and joined and current:
            current.append(here)
        else:
            if len(current) >= 2:
                lines.append(current)
            current = [here] if referenced else []
        prev = p
    if len(current) >= 2:
        lines.append(current)
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "MultiLineString",
                     "coordinates": [[[round(lng, 5), round(lat, 5)] for lat, lng in line]
                                     for line in lines]}}]}


def districts() -> list[list]:
    lngs = [p[0] for p in LINKOPING_AREA]
    lats = [p[1] for p in LINKOPING_AREA]
    bbox = f"{min(lats) - 0.02},{min(lngs) - 0.02},{max(lats) + 0.02},{max(lngs) + 0.02}"
    query = (f'[out:json][timeout:60];node["place"~"^(suburb|quarter|neighbourhood|village|'
             f'hamlet)$"]["name"]({bbox});out;')
    for attempt in range(6):  # Public Overpass servers are often briefly overloaded.
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = httpx.post(url, data={"data": query},
                           headers={"User-Agent": "heimdall-spy/0.1 (research)"}, timeout=90)
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    return [[round(e["lat"], 5), round(e["lon"], 5), e["tags"]["name"], e["tags"]["place"]]
            for e in r.json()["elements"]]


def places() -> dict:
    ways = json.loads(NAMED_WAYS.read_text())
    names: list[str] = []
    index: dict[str, int] = {}
    pts = []
    seen: set[tuple[int, int, int]] = set()
    for w in ways:
        name = w.get("name")
        if not name:
            continue
        k = index.setdefault(name, len(names))
        if k == len(names):
            names.append(name)
        geom = [tuple(p) for p in w["geometry"]]
        samples = sample_line(geom, PLACE_STEP_M) if len(geom) > 1 else [(*geom[0], 0.0)]
        for lat, lng, _ in samples:
            if not in_area(lng, lat, LINKOPING_AREA):
                continue
            key = (round(lat * 4000), round(lng * 2000), k)  # ~25 m dedup per name.
            if key in seen:
                continue
            seen.add(key)
            pts.append([round(lat, 5), round(lng, 5), k])
    return {"licence": "ODbL 1.0, © OpenStreetMap contributors", "names": names,
            "streets": pts, "districts": districts()}


def library() -> dict:
    refs = json.loads(GALLERY_INDEX.read_text())["references_in_index_order"]
    refs = [r for r in refs if in_area(r["lng"], r["lat"], LINKOPING_AREA)]
    return {"references": len(refs),
            "by_source": dict(Counter(r.get("source") for r in refs).most_common()),
            "area_polygon": [[lng, lat] for lng, lat in LINKOPING_AREA]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("coverage.geojson", coverage()), ("places.json", places()),
                       ("library.json", library())):
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        print(name, f"{(OUT / name).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
