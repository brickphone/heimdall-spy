"""Evaluate the development photo set with CIs, per-photo diagnostics and baselines."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval import geo_eval
from eval.public_photos import ROOT
from swespy.config import LINKOPING_AREA, MAJORITY_BASELINE_LATLNG, get_settings
from swespy.model.locator import Gallery, GeoLocator
from swespy.model.megaloc import MegaLocEmbedder, file_hash

DEPTH = geo_eval.RETRIEVAL_DEPTH


def robustness(queries: list[dict[str, Any]], rows: list[dict[str, Any]],
               majority: dict[str, Any]) -> dict[str, Any]:
    """Photographer-balanced accuracy and results without the three largest photographers."""
    radius = geo_eval.TARGET_RADIUS_M
    creators = [q["creator"] for q in queries]
    base_errors = geo_eval.constant_errors(queries, majority["lat"], majority["lng"])
    top3 = [c for c, _ in Counter(creators).most_common(3)]
    kept = [i for i, c in enumerate(creators) if c not in top3]
    sub = [rows[i] for i in kept]
    return {
        f"photographer_balanced_within_{radius}m": {
            "system": geo_eval.photographer_balanced(
                creators, [r["error_m"] <= radius for r in rows]),
            "majority_location": geo_eval.photographer_balanced(
                creators, [e <= radius for e in base_errors]),
        },
        "excluding_top3_photographers": {
            "excluded_photographers": top3,
            "query_count": len(sub),
            "system_within": {f"{r}m": geo_eval.rate(sum(x["error_m"] <= r for x in sub), len(sub))
                              for r in geo_eval.RADII_M},
            "majority_within": {
                f"{r}m": geo_eval.rate(sum(base_errors[i] <= r for i in kept), len(sub))
                for r in geo_eval.RADII_M},
            "failure_type": {t: sum(x["failure_type"] == t for x in sub)
                             for t in ("success", "ranking", "retrieval", "no_coverage")},
        },
        "note": "Diagnostics only; the development set itself is not reweighted or filtered",
    }


def evaluate_rows(
    queries: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score ``GeoLocator`` outputs; ranked indices refer to ``refs``."""
    lats = np.array([r["lat"] for r in refs])
    lngs = np.array([r["lng"] for r in refs])
    rows = []
    for q, result in zip(queries, results, strict=True):
        nearest = float(geo_eval.haversine_m(q["lat"], q["lng"], lats, lngs).min())
        coords = [(refs[i]["lat"], refs[i]["lng"]) for i in result["ranked_indices"]]
        predicted = (result["predicted_lat"], result["predicted_lng"])
        row = geo_eval.diagnose(q["lat"], q["lng"], predicted, coords, nearest)
        rows.append({
            "image_id": q["image_id"], "creator": q["creator"],
            "capture_date": q["capture_date"], "lat": q["lat"], "lng": q["lng"],
            **row,
            "confidence": result["confidence"],
            "top5": [{k: c[k] for k in ("image_id", "lat", "lng", "similarity")}
                     for c in result["candidates"][:5]],
        })
    return rows


def report(
    queries: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    label: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    lat, lng = MAJORITY_BASELINE_LATLNG
    majority = {"lat": lat, "lng": lng, "source": "fixed city-centre coordinate (config)"}
    metrics = geo_eval.summarize(rows)
    baseline = {**geo_eval.constant_baseline(queries, majority["lat"], majority["lng"]),
                "location": majority}
    return {
        "label": label,
        "created_at": datetime.now(UTC).isoformat(),
        "definitions": {
            "prediction": "Location of the top-ranked reference",
            "failure_type": f"no_coverage: no reference within {geo_eval.COVERAGE_RADIUS_M} m; "
                            f"success: top-1 within {geo_eval.COVERAGE_RADIUS_M} m; ranking: "
                            f"covered and within top {DEPTH}; retrieval: covered, not in top "
                            f"{DEPTH}",
            "ci95": "Wilson score interval",
            "majority_location": "Fixed city-centre coordinate MAJORITY_BASELINE_LATLNG "
                                 "(mean of the most populated 1 km cell of the original "
                                 "gallery); independent of the evaluated gallery",
            "random_reference": "Exact expectation of predicting a uniformly random reference",
            "reranking_ceiling": "Share with a reference within 100 m in the top 50; geometric "
                                 "verification cannot recover matches beyond the same scene",
        },
        "reference_count": len(refs),
        "metrics": metrics,
        "target": geo_eval.target_check(metrics, baseline),
        "baselines": {
            "majority_location": baseline,
            "random_reference": geo_eval.random_reference_baseline(queries, refs),
        },
        "robustness": robustness(queries, rows, majority),
        "dev_set_distribution": geo_eval.distribution(queries),
        "provenance": provenance,
        "queries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path, default=Path(get_settings().gallery_index))
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--label", default="baseline_v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.dev_root / "manifest.json"
    queries = json.loads(manifest_path.read_text())["queries"]
    gallery = Gallery(args.index)
    embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), args.device)
    locator = GeoLocator(embedder, gallery, top_k=5, depth=DEPTH)
    # Same pixels as an upload; the hash-keyed cache only avoids re-encoding.
    vectors = embedder.embed([args.dev_root / q["path"] for q in queries],
                             [q["sha256"] for q in queries], batch_size=4)
    results = [locator.locate_vector(v) for v in vectors]
    refs, excluded = gallery.refs, gallery.excluded_outside_area
    rows = evaluate_rows(queries, refs, results)
    result = report(queries, refs, rows, args.label, {
        "dev_manifest_sha256": file_hash(manifest_path),
        "index_sha256": gallery.index_sha256,
        "gallery_filter": {"polygon_lnglat": [list(v) for v in LINKOPING_AREA],
                           "references_in_saved_index": len(refs) + excluded,
                           "excluded_outside_polygon": excluded,
                           "references_used": len(refs)},
        "model_identity": embedder.identity,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    m = result["metrics"]
    for radius, value in m["within"].items():
        print(f"within {radius}: {value['count']}/{value['n']} "
              f"[{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]")
    print("target:", {k: result["target"][k] for k in
                      ("a_rate_at_least_40pct", "b_ci_above_majority", "met")})
    print("excluded outside polygon:", excluded)
    print("failure types:", {k: v["count"] for k, v in m["failure_type"].items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
