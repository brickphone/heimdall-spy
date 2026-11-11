"""Automatic region search: whole vs + people masking vs + masking + crops.

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m eval.auto_regions run --sets dev,street \
        [--named NAME=PATH ...]
    python -m eval.auto_regions run --sets field     # adds field rows to the same file
    python -m eval.auto_regions summary

Dev, street-image and field sets only (never the test set); nothing is tuned here. The
protocol, including the confidence tightening rule, was fixed beforehand in
docs/experiments/14_auto_regions.md. Every photo goes through ``GeoLocator.locate_bytes``,
the API's own path. A named photo's EXIF GPS is read only as ground truth for scoring;
the locator receives the same bytes, whose metadata ``decode_upload`` discards. Named
photos are reported on their own, never in the headline set metrics.

The run stops (keeping the rows so far) if free disk falls below ``MIN_FREE_DISK_GB`` or
swap use exceeds ``MAX_SWAP_GB``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from eval.geo_eval import haversine_m, wilson_interval
from eval.gsv_eval import load_sets

OUT = Path("data/auto_regions/eval")
CONFIGS = ("whole", "masked", "full")
CORRECT_M = 500.0
TIGHT_INLIERS = 50  # Pre-registered: applied only if dev confident-but-wrong increases.
SAME_RESULT_M = 100.0
IOU_MATCH = 0.3
MIN_FREE_DISK_GB = 4.0
MAX_SWAP_GB = 6.0


def swap_used_gb() -> float:
    """Swap in use: sysctl on macOS, /proc/meminfo on Linux."""
    import subprocess

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        kb = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
        total = int(kb["SwapTotal"].split()[0])
        free = int(kb["SwapFree"].split()[0])
        return (total - free) / 1024**2
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0]) / 1024 if "used = " in out else 0.0


def resource_problem() -> str | None:
    """Why the run must stop now, or None (free disk where results are written)."""
    import shutil

    free = shutil.disk_usage(OUT.resolve().parent if OUT.exists() else ".").free / 1e9
    if free < MIN_FREE_DISK_GB:
        return f"free disk {free:.1f} GB < {MIN_FREE_DISK_GB} GB"
    swap = swap_used_gb()
    if swap > MAX_SWAP_GB:
        return f"swap {swap:.1f} GB > {MAX_SWAP_GB} GB"
    return None


def exif_gps(path: Path) -> tuple[float, float] | None:
    """(lat, lng) from EXIF, for scoring only."""
    from PIL import Image
    from pillow_heif import register_heif_opener

    register_heif_opener()
    with Image.open(path) as im:
        gps = im.getexif().get_ifd(0x8825)
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def degrees(v: Any, ref: str) -> float:
        d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -d if ref in ("S", "W") else d

    return degrees(gps[2], gps.get(1, "N")), degrees(gps[4], gps.get(3, "E"))


def iou(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    if not a or not b:
        return 0.0
    w = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    h = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    inter = max(w, 0) * max(h, 0)
    return inter / (a["width"] * a["height"] + b["width"] * b["height"] - inter)


def build(mode: str) -> Any:
    from swespy.config import get_settings
    from swespy.model import rerank as rr
    from swespy.model.locator import Gallery, GeoLocator
    from swespy.model.megaloc import MegaLocEmbedder
    from swespy.model.variants import PersonSegmenter

    s = get_settings()
    gallery = Gallery(Path(s.gallery_index), s.area_polygon)
    embedder = MegaLocEmbedder(Path(s.megaloc_root), s.megaloc_device)
    return GeoLocator(embedder, gallery, s.top_k_results, s.retrieval_depth,
                      verifier=rr.GeometricVerifier(s.rerank_device), weight=rr.DEFAULT_WEIGHT,
                      min_inliers=rr.DEFAULT_MIN_INLIERS, rerank_k=s.rerank_k,
                      segmenter=PersonSegmenter(s.segment_device), variants_mode=mode)


def row(result: dict[str, Any], lat: float, lng: float, seconds: float) -> dict[str, Any]:
    top = result["candidates"][0]
    return {
        "error_m": float(haversine_m(lat, lng, np.array([result["predicted_lat"]]),
                                     np.array([result["predicted_lng"]]))[0]),
        "state": result.get("confidence_state"),
        "top_inliers": top.get("inliers"),
        "variant": result["searched_variant"],
        "region": result["searched_region"],
        "person_share": result["person_share"],
        "predicted": [result["predicted_lat"], result["predicted_lng"]],
        "timings_ms": result["timings_ms"],
        "seconds": round(seconds, 3),
    }


def run(set_names: list[str], named: list[str]) -> Path:
    sets = {k: v for k, v in load_sets().items() if k in set_names}
    if named:
        sets["named"] = []
    for item in named:
        name, path = item.split("=", 1)
        gps = exif_gps(Path(path).expanduser())
        if gps is None:
            raise SystemExit(f"{name}: no EXIF GPS to score against")
        sets["named"].append({"image_id": name, "path": str(Path(path).expanduser()),
                              "lat": gps[0], "lng": gps[1]})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    # Rows of sets not being re-run are kept (e.g. field rows added later).
    rows: dict[str, dict[str, Any]] = {k: v for k, v in saved.get("rows", {}).items()
                                       if v["set"] not in sets}
    locator = build("full")

    def save(status: str) -> None:
        path.write_text(json.dumps({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "status": status, "versions": locator.versions(),
                                    "rows": rows}, indent=1))

    for mode in CONFIGS:
        locator.variants_mode = mode
        for set_name, queries in sets.items():
            for q in queries:
                problem = resource_problem()
                if problem:
                    save(f"stopped: {problem}")
                    raise SystemExit(f"Stopped: {problem}. Rows so far are in {path}.")
                contents = Path(q["path"]).read_bytes()
                key = f"{set_name}/{q['image_id']}"
                rows.setdefault(key, {"set": set_name})
                start = time.perf_counter()
                result = locator.locate_bytes(contents)
                rows[key][mode] = row(result, q["lat"], q["lng"], time.perf_counter() - start)
                manual = (json.loads((Path(q["path"]).parent / "record.json").read_text())
                          .get("prediction", {}).get("region") if set_name == "field" else None)
                if manual:
                    result = locator.locate_bytes(contents, manual)
                    rows[key][f"{mode}_manual"] = {
                        **row(result, q["lat"], q["lng"], 0.0), "manual_region": manual}
                print(mode, key, rows[key][mode]["variant"],
                      round(rows[key][mode]["error_m"]), rows[key][mode]["state"], flush=True)
            save(f"running: {mode}/{set_name}")
    save("complete")
    return path


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(k, n) if n else (float("nan"), float("nan"))
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None,
            "ci95": [round(lo, 3), round(hi, 3)]}


def tightened(r: dict[str, Any]) -> str:
    automatic = r["variant"] not in ("whole", "manual")
    if r["state"] == "ok" and automatic and (r["top_inliers"] or 0) < TIGHT_INLIERS:
        return "low"
    return r["state"]


def block(rows: list[dict[str, Any]], mode: str, tight: bool = False) -> dict[str, Any]:
    rs = [r[mode] for r in rows]
    states = [tightened(r) if tight else r["state"] for r in rs]
    errs = [r["error_m"] for r in rs]
    n = len(rs)
    secs = [r["seconds"] for r in rs]
    return {
        "within_100m": rate(sum(e <= 100 for e in errs), n),
        "within_500m": rate(sum(e <= 500 for e in errs), n),
        "within_1km": rate(sum(e <= 1000 for e in errs), n),
        "median_error_m": round(float(np.median(errs)), 1) if errs else None,
        "ok": sum(s == "ok" for s in states),
        "ok_correct": rate(sum(s == "ok" and e <= CORRECT_M for s, e in zip(states, errs,
                                                                             strict=True)),
                           sum(s == "ok" for s in states)),
        "confident_but_wrong": sum(s == "ok" and e > CORRECT_M
                                   for s, e in zip(states, errs, strict=True)),
        "winners": dict(Counter(r["variant"] for r in rs)),
        "seconds_median": round(float(np.median(secs)), 2) if secs else None,
        "seconds_p90": round(float(np.percentile(secs, 90)), 2) if secs else None,
        "changed_500m_vs_whole": {
            "fixed": sum(r["whole"]["error_m"] > 500 >= r[mode]["error_m"] for r in rows),
            "broken": sum(r["whole"]["error_m"] <= 500 < r[mode]["error_m"] for r in rows)},
    }


def summary() -> dict[str, Any]:
    rows = json.loads((OUT / "rows.json").read_text())["rows"]
    by_set: dict[str, list[dict[str, Any]]] = {}
    for key, r in rows.items():
        by_set.setdefault(r["set"], []).append({"key": key, **r})
    dev = by_set.get("dev", [])
    tighten = bool(dev) and (block(dev, "full")["confident_but_wrong"]
                             > block(dev, "whole")["confident_but_wrong"])
    out: dict[str, Any] = {"tightening_rule_applied": tighten, "sets": {}}
    # Only photos with all three configurations are scored (a stopped run is partial).
    by_set = {k: [r for r in v if all(m in r for m in CONFIGS)] for k, v in by_set.items()}
    for name, rs in by_set.items():
        entry: dict[str, Any] = {"n": len(rs)}
        for mode in CONFIGS:
            entry[mode] = block(rs, mode)
        if tighten:
            entry["full_tightened"] = block(rs, "full", tight=True)
        manual = [r for r in rs if "full_manual" in r]
        if manual:
            entry["manual_region"] = {
                "n": len(manual),
                "auto_iou_ge_0_3": sum(iou(r["full"]["region"], r["full_manual"]["manual_region"])
                                       >= IOU_MATCH for r in manual),
                "same_result_100m": sum(float(haversine_m(
                    *r["full"]["predicted"], np.array([r["full_manual"]["predicted"][0]]),
                    np.array([r["full_manual"]["predicted"][1]]))[0]) <= SAME_RESULT_M
                    for r in manual),
            }
        if name == "named":
            entry["cases"] = {r["key"]: {m: {k: r[m][k] for k in (
                "variant", "region", "person_share", "error_m", "state", "top_inliers")}
                for m in CONFIGS} for r in rs}
        out["sets"][name] = entry
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summary"))
    parser.add_argument("--sets", default="dev,street", help="comma-separated: dev,street,field")
    parser.add_argument("--named", action="append", default=[])
    args = parser.parse_args()
    if args.command == "run":
        print(run(args.sets.split(","), args.named))
    print(json.dumps(summary(), indent=1))


if __name__ == "__main__":
    main()
