"""Propose self-collected walk routes for the reference gallery.

Two modes, both using ONLY the gallery and the OpenStreetMap street network; evaluation
photo locations are never read. Output: data/gallery_v2/walk-routes.{json,geojson,html}.

- ``pedestrian`` (default): pedestrian-only streets (footway, pedestrian, path, steps,
  living_street) whose points have no gallery reference within ``SERVED_RADIUS_M``.
  The gallery is almost entirely car-mounted Mapillary imagery, so these are the
  places with no pedestrian viewpoint at all.
- ``density``: 250 m cells with walkable streets but <= 2 references.

    python -m eval.walk_routes --routes 5 [--mode density]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from eval.densify import ROOT, UA, polygon_bbox, read_json, write_json
from eval.geo_eval import haversine_m
from swespy.config import LINKOPING_AREA, in_area

OVERPASS = "https://overpass-api.de/api/interpreter"
WALKABLE = (
    "residential|living_street|pedestrian|footway|path|cycleway|tertiary|"
    "secondary|primary|unclassified|service|steps"
)
PEDESTRIAN = ("footway", "pedestrian", "path", "steps", "living_street")
SERVED_RADIUS_M = 15.0  # A pedestrian-street point is served if a reference is this close.
SAMPLE_STEP_M = 10.0
BLOCK_SEPARATION_M = 1500.0  # Proposed walk areas are at least this far apart.
GAP_MAX_REFS = 2  # A 250 m cell with <= 2 references counts as a gap.
MIN_STREET_M = 150  # Ignore cells with almost no walkable street.


def streets(root: Path) -> list[dict[str, Any]]:
    """Walkable OSM ways inside the polygon bbox (cached; one Overpass request)."""
    cache = root / "osm-walkable.json"
    if cache.exists():
        return read_json(cache)
    west, south, east, north = polygon_bbox()
    query = (
        f'[out:json][timeout:180];way["highway"~"^({WALKABLE})$"]'
        f"({south},{west},{north},{east});out tags geom;"
    )
    r = httpx.post(OVERPASS, data={"data": query}, headers={"User-Agent": UA}, timeout=240)
    r.raise_for_status()
    ways = [
        {
            "name": w.get("tags", {}).get("name"),
            "highway": w["tags"]["highway"],
            "geometry": [(p["lat"], p["lon"]) for p in w.get("geometry", [])],
        }
        for w in r.json()["elements"]
    ]
    write_json(cache, ways)
    return ways


def propose(root: Path, count: int) -> dict[str, Any]:
    grid = read_json(
        root
        / (
            "density-grid-historic.json"
            if (root / "density-grid-historic.json").exists()
            else "density-grid.json"
        )
    )
    counts, mask = grid["counts"], grid["outside_mask"]
    west, south, dlat, dlng = grid["west"], grid["south"], grid["dlat"], grid["dlng"]

    def cell(lat: float, lng: float) -> tuple[int, int]:
        return int((lat - south) / dlat), int((lng - west) / dlng)

    street_m: dict[tuple[int, int], float] = defaultdict(float)
    names: dict[tuple[int, int], set[str]] = defaultdict(set)
    segments: dict[tuple[int, int], list[list[tuple[float, float]]]] = defaultdict(list)
    for way in streets(root):
        g = way["geometry"]
        for (a_lat, a_lng), (b_lat, b_lng) in zip(g, g[1:], strict=False):
            m_lat, m_lng = (a_lat + b_lat) / 2, (a_lng + b_lng) / 2
            if not in_area(m_lng, m_lat):
                continue
            j, i = cell(m_lat, m_lng)
            if not (0 <= j < len(counts) and 0 <= i < len(counts[0])):
                continue
            street_m[(j, i)] += float(
                haversine_m(a_lat, a_lng, np.array([b_lat]), np.array([b_lng]))[0]
            )
            segments[(j, i)].append([(a_lat, a_lng), (b_lat, b_lng)])
            if way["name"]:
                names[(j, i)].add(way["name"])
    gaps = {
        c
        for c, m in street_m.items()
        if m >= MIN_STREET_M and not mask[c[0]][c[1]] and counts[c[0]][c[1]] <= GAP_MAX_REFS
    }
    # Connected gap areas (8-neighbourhood), ranked by unwalked street length.
    seen, clusters = set(), []
    for start in sorted(gaps):
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            c = stack.pop()
            members.append(c)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    n = (c[0] + dj, c[1] + di)
                    if n in gaps and n not in seen:
                        seen.add(n)
                        stack.append(n)
        clusters.append(members)
    clusters.sort(key=lambda m: -sum(street_m[c] for c in m))
    routes = []
    for rank, members in enumerate(clusters[:count], 1):
        lat = float(np.mean([south + (c[0] + 0.5) * dlat for c in members]))
        lng = float(np.mean([west + (c[1] + 0.5) * dlng for c in members]))
        routes.append(
            {
                "rank": rank,
                "centre": {"lat": round(lat, 5), "lng": round(lng, 5)},
                "gap_cells_250m": len(members),
                "street_length_km": round(sum(street_m[c] for c in members) / 1000, 2),
                "existing_references": sum(counts[c[0]][c[1]] for c in members),
                "streets": sorted(set().union(*(names[c] for c in members)))[:25],
                "cells": [[south + c[0] * dlat, west + c[1] * dlng] for c in members],
                "segments": [s for c in members for s in segments[c]],
            }
        )
    report = {
        "rule": f"250 m cells inside the polygon with <= {GAP_MAX_REFS} gallery "
        f"references and >= {MIN_STREET_M} m of walkable OSM street; "
        "adjacent cells merged; ranked by street length. Eval photo "
        "locations not used.",
        "gap_cells": len(gaps),
        "gap_areas": len(clusters),
        "routes": [{k: v for k, v in r.items() if k not in ("cells", "segments")} for r in routes],
    }
    write_json(root / "walk-routes.json", report)
    features = [
        {
            "type": "Feature",
            "properties": {"rank": r["rank"]},
            "geometry": {
                "type": "MultiLineString",
                "coordinates": [[[p[1], p[0]] for p in s] for s in r["segments"]],
            },
        }
        for r in routes
    ]
    write_json(root / "walk-routes.geojson", {"type": "FeatureCollection", "features": features})
    render(root, routes, grid, dlat, dlng)
    return report


def propose_pedestrian(root: Path, index: Path, count: int) -> dict[str, Any]:
    """Areas with the most pedestrian-only street length lacking any nearby reference."""
    import math

    refs = read_json(index)["references_in_index_order"]
    cell_deg = SERVED_RADIUS_M / 111_320
    occupied: set[tuple[int, int]] = set()
    for r in refs:
        occupied.add(
            (
                int(r["lat"] / cell_deg),
                int(r["lng"] / (cell_deg / math.cos(math.radians(r["lat"])))),
            )
        )

    def served(lat: float, lng: float) -> bool:
        cy, cx = int(lat / cell_deg), int(lng / (cell_deg / math.cos(math.radians(lat))))
        return any((cy + dy, cx + dx) in occupied for dy in (-1, 0, 1) for dx in (-1, 0, 1))

    west, south, east, north = polygon_bbox()
    cell, dlat = 250, 250 / 111_320
    dlng = cell / (111_320 * math.cos(math.radians((south + north) / 2)))
    unserved: dict[tuple[int, int], float] = defaultdict(float)
    total: dict[tuple[int, int], float] = defaultdict(float)
    names: dict[tuple[int, int], set[str]] = defaultdict(set)
    segments: dict[tuple[int, int], list[list[tuple[float, float]]]] = defaultdict(list)
    for way in streets(root):
        if way["highway"] not in PEDESTRIAN:
            continue
        g = way["geometry"]
        for (a_lat, a_lng), (b_lat, b_lng) in zip(g, g[1:], strict=False):
            length = float(haversine_m(a_lat, a_lng, np.array([b_lat]), np.array([b_lng]))[0])
            steps = max(1, int(length // SAMPLE_STEP_M))
            for k in range(steps):
                t = (k + 0.5) / steps
                lat, lng = a_lat + t * (b_lat - a_lat), a_lng + t * (b_lng - a_lng)
                if not in_area(lng, lat):
                    continue
                c = (int((lat - south) / dlat), int((lng - west) / dlng))
                total[c] += length / steps
                if not served(lat, lng):
                    unserved[c] += length / steps
                    segments[c].append([(a_lat, a_lng), (b_lat, b_lng)])
                    if way["name"]:
                        names[c].add(way["name"])
    # Unserved paths form one city-wide network, so connected areas are not walkable
    # routes. Instead score 500 m blocks (2x2 cells) and pick the best blocks greedily,
    # at least BLOCK_SEPARATION_M apart, so routes spread across the city.
    blocks: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for c in unserved:
        blocks[(c[0] // 2, c[1] // 2)].append(c)
    ranked_blocks = sorted(blocks.items(), key=lambda kv: -sum(unserved[c] for c in kv[1]))
    clusters, centres = [], []
    for _, members in ranked_blocks:
        lat = float(np.mean([south + (c[0] + 0.5) * dlat for c in members]))
        lng = float(np.mean([west + (c[1] + 0.5) * dlng for c in members]))
        if any(
            float(haversine_m(lat, lng, np.array([a]), np.array([b]))[0]) < BLOCK_SEPARATION_M
            for a, b in centres
        ):
            continue
        clusters.append(members)
        centres.append((lat, lng))
        if len(clusters) == count:
            break
    routes = []
    for rank, members in enumerate(clusters[:count], 1):
        segs = {tuple(map(tuple, s)) for c in members for s in segments[c]}
        routes.append(
            {
                "rank": rank,
                "centre": {
                    "lat": round(float(np.mean([south + (c[0] + 0.5) * dlat for c in members])), 5),
                    "lng": round(float(np.mean([west + (c[1] + 0.5) * dlng for c in members])), 5),
                },
                "cells_250m": len(members),
                "unserved_pedestrian_km": round(sum(unserved[c] for c in members) / 1000, 2),
                "pedestrian_km_in_area": round(sum(total[c] for c in members) / 1000, 2),
                "streets": sorted(set().union(*(names[c] for c in members)))[:25],
                "cells": [[south + c[0] * dlat, west + c[1] * dlng] for c in members],
                "segments": [list(s) for s in segs],
            }
        )
    report = {
        "mode": "pedestrian",
        "rule": f"OSM {', '.join(PEDESTRIAN)} sampled every {SAMPLE_STEP_M:.0f} m; a point is "
        f"unserved if no gallery reference lies within {SERVED_RADIUS_M:.0f} m; 500 m "
        f"blocks ranked by unserved length, picked greedily >= {BLOCK_SEPARATION_M:.0f} m "
        "apart. Eval photo locations not used.",
        "gallery": str(index),
        "references": len(refs),
        "pedestrian_km_total": round(sum(total.values()) / 1000, 1),
        "pedestrian_km_unserved": round(sum(unserved.values()) / 1000, 1),
        "blocks_with_unserved_paths": len(blocks),
        "routes": [{k: v for k, v in r.items() if k not in ("cells", "segments")} for r in routes],
    }
    write_json(root / "walk-routes.json", report)
    write_json(
        root / "walk-routes.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"rank": r["rank"]},
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [[[p[1], p[0]] for p in s] for s in r["segments"]],
                    },
                }
                for r in routes
            ],
        },
    )
    for r in routes:
        r["street_length_km"] = r["unserved_pedestrian_km"]
    render(root, routes, {}, dlat, dlng)
    return report


def render(
    root: Path, routes: list[dict[str, Any]], grid: dict[str, Any], dlat: float, dlng: float
) -> None:
    colours = ["#d32f2f", "#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a"]
    data = {
        "poly": [[p[1], p[0]] for p in LINKOPING_AREA],
        "routes": [
            {
                "rank": r["rank"],
                "segments": r["segments"],
                "cells": r["cells"],
                "km": r["street_length_km"],
                "streets": r["streets"][:8],
            }
            for r in routes
        ],
        "dlat": dlat,
        "dlng": dlng,
    }
    page = f"""<!doctype html><meta charset="utf-8"><title>Walk route proposals</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{{height:100%;margin:0}}#k{{position:absolute;z-index:999;top:10px;
right:10px;background:#fff;padding:8px 12px;font:13px system-ui;border-radius:6px;
max-width:320px}}</style><div id="m"></div><div id="k"></div>
<script>const D={json.dumps(data)};const C={json.dumps(colours)};
const m=L.map('m').setView([58.41,15.62],12);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
{{attribution:'© OpenStreetMap contributors'}}).addTo(m);
L.polygon(D.poly,{{color:'#555',fill:false}}).addTo(m);let k='<b>Coverage-gap walks</b>';
D.routes.forEach((r,i)=>{{r.cells.forEach(c=>L.rectangle([c,[c[0]+D.dlat,c[1]+D.dlng]],
{{color:C[i],weight:0,fillOpacity:.12}}).addTo(m));
r.segments.forEach(s=>L.polyline(s,{{color:C[i],weight:3}}).addTo(m));
k+=`<br><span style="color:${{C[i]}}">■</span> ${{r.rank}}: ${{r.km}} km — `
 +r.streets.join(', ')}});
document.getElementById('k').innerHTML=k;</script>"""
    (root / "walk-routes.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--routes", type=int, default=5)
    parser.add_argument("--mode", choices=["pedestrian", "density"], default="pedestrian")
    parser.add_argument(
        "--index", type=Path, default=Path("data/gallery_v2/gallery-index-historic.json")
    )
    args = parser.parse_args()
    if not 3 <= args.routes <= 5:
        parser.error("Propose 3-5 routes")
    if args.mode == "density":
        result = propose(args.root, args.routes)
    else:
        result = propose_pedestrian(args.root, args.index, args.routes)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
