"""Street View via the unofficial backend (streetlevel): feasibility pilot.

Recorded decision: using the unofficial Google Street View backend via the open-source
``streetlevel`` library is approved for a pilot. Risk and responsibility accepted by
Mattis Hüttenrauch on 2026-11-10.

All rules of ``swespy/gsv.py`` apply (descriptors only, never a CC label, every pull
logged, never training, never shown in the demo). This module, stage by stage:

  select   (Mac) 500 pilot points from the Part 1 gap plan: 250 priority 1 + 250
           priority 2, spread over 10 k-means areas. Chosen without looking at dev,
           field or test photos. -> data/gsv_internal/pilot_points.csv + pilot_map.html
  report   (Mac) Part 5 numbers from the copied-back pilot files -> pilot_report.json

The fetch itself (Part 2, on the pod) is eval/gsv_internal_fetch.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from eval.densify import log, read_json
from eval.gsv_plan import M_PER_DEG_LAT, m_per_deg_lng, metadata_queue
from eval.gsv_plan import ROOT as PLAN_ROOT
from swespy.config import LINKOPING_AREA

ROOT = Path("data/gsv_internal")
SEED = 20261110
N_AREAS = 10
PER_PRIORITY = 250
CSV_FIELDS = ("point_id", "lat", "lng", "priority", "highway", "bearing", "area")


def areas(points: list[dict[str, Any]], k: int = N_AREAS, seed: int = SEED) -> np.ndarray:
    """Seeded k-means on metric coordinates: k disjoint areas covering the gap points."""
    from scipy.cluster.vq import kmeans2

    xy = np.array([[p["lat"] * M_PER_DEG_LAT, p["lng"] * m_per_deg_lng(58.41)]
                   for p in points])
    _, labels = kmeans2(xy, k, seed=seed, minit="++")
    return labels


def select(points: list[dict[str, Any]], per_priority: int = PER_PRIORITY,
           k: int = N_AREAS, seed: int = SEED) -> list[dict[str, Any]]:
    """``per_priority`` points of priority 1 and 2 each, as evenly as possible per area."""
    pool = [p for p in metadata_queue(points) if p["priority"] in (1, 2)]
    labels = areas(pool, k, seed)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for priority in (1, 2):
        by_area = {a: [p for p, lab in zip(pool, labels, strict=True)
                       if lab == a and p["priority"] == priority] for a in range(k)}
        for group in by_area.values():
            rng.shuffle(group)
        # Round-robin over areas, so small areas give what they have and the rest fill up.
        picked = 0
        while picked < per_priority and any(by_area.values()):
            for a in range(k):
                if by_area[a] and picked < per_priority:
                    p = by_area[a].pop()
                    chosen.append({**p, "area": int(a)})
                    picked += 1
    for i, p in enumerate(chosen):
        p["point_id"] = f"p{p['priority']}-{i:03d}"
    return chosen


def write_csv(path: Path, chosen: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for p in chosen:
            w.writerow({k: p[k] for k in CSV_FIELDS})


def read_points(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [{**r, "lat": float(r["lat"]), "lng": float(r["lng"]),
                 "priority": int(r["priority"]), "bearing": float(r["bearing"]),
                 "area": int(r["area"])} for r in csv.DictReader(f)]


def pilot_map(path: Path, chosen: list[dict[str, Any]]) -> None:
    data = [[p["lat"], p["lng"], p["priority"], p["area"], p["highway"]] for p in chosen]
    area = [[lat, lng] for lng, lat in LINKOPING_AREA]
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Street View pilot points</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0;font:14px system-ui}}#map{{height:100vh}}
.legend{{background:#fff;padding:8px 10px;border-radius:6px;line-height:1.6}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}</style>
</head><body><div id="map"></div><script>
const pts = {json.dumps(data)};
const map = L.map("map").setView([58.41, 15.62], 12);
L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
  {{maxZoom: 19, attribution: "© OpenStreetMap contributors"}}).addTo(map);
L.polygon({json.dumps(area)}, {{color: "#333", weight: 1, fill: false}}).addTo(map);
const col = {{1: "#d7301f", 2: "#2c7fb8"}};
for (const [lat, lng, p, a, hw] of pts)
  L.circleMarker([lat, lng], {{radius: 5, weight: 1, color: "#fff", fillOpacity: 0.9,
    fillColor: col[p]}}).bindTooltip(`priority ${{p}} · area ${{a}} · ${{hw}}`).addTo(map);
const lg = L.control({{position: "topright"}});
lg.onAdd = () => {{ const d = L.DomUtil.create("div", "legend"); d.innerHTML =
  `<b>${{pts.length}} pilot points, 10 areas</b><br>` +
  `<i style="background:${{col[1]}}"></i>priority 1: empty, non-road<br>` +
  `<i style="background:${{col[2]}}"></i>priority 2: empty, road`; return d; }};
lg.addTo(map);
</script></body></html>""")


def select_stage(root: Path) -> dict[str, Any]:
    points = read_json(PLAN_ROOT / "sample_points.json")["points"]
    chosen = select(points)
    write_csv(root / "pilot_points.csv", chosen)
    pilot_map(root / "pilot_map.html", chosen)
    summary = {
        "points": len(chosen),
        "per_priority": dict(sorted(Counter(p["priority"] for p in chosen).items())),
        "per_area": {a: dict(sorted(Counter(p["priority"] for p in chosen
                                            if p["area"] == a).items()))
                     for a in sorted({p["area"] for p in chosen})},
        "per_way_type": dict(Counter(p["highway"] for p in chosen).most_common()),
        "min_spacing_m": round(min_spacing(chosen), 1),
    }
    log("select", **{k: v for k, v in summary.items() if k != "per_area"})
    return summary


def min_spacing(points: list[dict[str, Any]]) -> float:
    best = math.inf
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            d = math.hypot((a["lat"] - b["lat"]) * M_PER_DEG_LAT,
                           (a["lng"] - b["lng"]) * m_per_deg_lng(a["lat"]))
            best = min(best, d)
    return best


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x] if path.exists() else []


def report_stage(root: Path, cost_per_hr: float, pod_seconds: float | None) -> dict[str, Any]:
    """Part 5 numbers from the copied-back files (no network)."""
    folder = root / "gsv_internal_pilot"
    points = jsonl(folder / "points.jsonl")
    views = jsonl(folder / "views.jsonl")
    pull = jsonl(root / "pull_log.jsonl")
    summary = json.loads((folder / "summary.json").read_text())
    scan = json.loads((root / "disk_scan.json").read_text()) if (root / "disk_scan.json").exists() \
        else None
    covered = {"fetched", "duplicate_pano", "tile_error_http_404"}
    by_priority: dict[str, Any] = {}
    for prio in (1, 2):
        pts = [p for p in points if p["priority"] == prio]
        outcomes = Counter(p["outcome"] for p in pts)
        by_priority[f"priority_{prio}"] = {
            "points": len(pts), **dict(outcomes.most_common()),
            "points_with_official_pano": sum(1 for p in pts
                                             if p["outcome"] in covered
                                             or p["outcome"].startswith("tile_error")),
            "unique_panos_fetched": outcomes["fetched"],
        }
    by_way = {}
    for way in sorted({p["way_type"] for p in points}):
        pts = [p for p in points if p["way_type"] == way]
        has = sum(1 for p in pts if p["outcome"] in covered or p["outcome"].startswith("tile_"))
        by_way[way] = {"points": len(pts), "with_official_pano": has}
    fetched = [p for p in points if p["outcome"] == "fetched"]
    requests = sum(summary["requests"].values())
    ok = sum(summary["successful"].values())
    hours = summary["elapsed_s"] / 3600
    years = Counter((p.get("date") or "unknown")[:4] for p in fetched)
    report = {
        "stop_reason": summary["stop_reason"],
        "points_processed": len(points),
        "by_priority": by_priority,
        "by_way_type": by_way,
        "requests": summary["requests"], "requests_total": requests,
        "requests_successful": ok, "success_rate": round(ok / requests, 4) if requests else None,
        "errors": summary["errors"],
        "requests_per_fetched_pano": round(requests / len(fetched), 1) if fetched else None,
        "tiles_per_pano": summary["tiles_per_pano"],
        "throughput": {"requests_per_min": round(requests / (hours * 60), 1) if hours else None,
                       "panos_per_hour": round(len(fetched) / hours, 1) if hours else None,
                       "views_per_hour": round(len(views) / hours, 1) if hours else None,
                       "fetch_hours": round(hours, 2)},
        "requests_before_stop": requests,
        "limiter": {"max_in_flight": summary["max_in_flight"],
                    "min_spacing_s": summary["min_spacing_s"]},
        "pano_resolution": {"zooms": summary["zooms"], "sizes": summary["pano_sizes"]},
        "capture_years": dict(sorted(years.items())),
        "older_captures": {"panos_with_any": sum(1 for p in fetched if p.get("historical")),
                           "of": len(fetched),
                           "total_older_captures": sum(p.get("historical") or 0
                                                       for p in fetched)},
        "views": summary["views"], "views_stored": len(views),
        "encoder_errors": summary["encoder_errors"],
        "max_encoder_queue": summary["max_encoder_queue"],
        "pull_log_rows": len(pull),
        "disk_scan": scan,
        "pod_cost_usd": round(cost_per_hr * pod_seconds / 3600, 3) if pod_seconds else None,
        "pod_hours": round(pod_seconds / 3600, 2) if pod_seconds else None,
    }
    (root / "pilot_report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "report"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cost-per-hr", type=float, default=0.16)
    parser.add_argument("--pod-seconds", type=float)
    args = parser.parse_args()
    if args.stage == "select":
        print(json.dumps(select_stage(args.root), indent=1))
    else:
        print(json.dumps(report_stage(args.root, args.cost_per_hr, args.pod_seconds), indent=1))


if __name__ == "__main__":
    main()
