"""Part 5: geometric re-ranking of MegaLoc top-K on the 66-photo DEV set (never the test set).

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m eval.rerank_dev pairs     # inliers for top-K
    python -m eval.rerank_dev tune                                     # weight + threshold
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m eval.rerank_dev latency   # MPS and CPU timings

Tuning protocol (fixed before looking at results): combined score =
cosine + w * log(1 + inliers); w chosen from a fixed grid by the number of dev photos
whose top-1 is within 100 m, ties broken by within 500 m, then by the smaller w. The
low-confidence inlier threshold is the smallest value whose confident subset reaches
>= 80% within 500 m on dev (or the largest grid value if none does).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval import dev_eval, geo_eval
from eval.public_photos import ROOT as DEV_ROOT
from swespy.config import get_settings
from swespy.model import rerank as rr
from swespy.model.locator import Gallery, GeoLocator
from swespy.model.megaloc import MegaLocEmbedder, file_hash

WORK = Path("data/gallery_v2/rerank")
WEIGHT_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
THRESHOLD_GRID = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100)
TARGET_CONFIDENT_ACCURACY = 0.80


def load(index: Path, device: str) -> tuple[list[dict[str, Any]], Gallery, Any, list[Any]]:
    queries = json.loads((DEV_ROOT / "manifest.json").read_text())["queries"]
    gallery = Gallery(index)
    embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), device)
    vectors = embedder.embed(
        [DEV_ROOT / q["path"] for q in queries], [q["sha256"] for q in queries], batch_size=4
    )
    return queries, gallery, embedder, list(vectors)


def pairs_path(k: int, keypoints: int) -> Path:
    return WORK / (
        f"dev_pairs_k{k}.json"
        if keypoints == rr.MAX_KEYPOINTS
        else f"dev_pairs_k{k}_kp{keypoints}.json"
    )


def pairs(index: Path, k: int, device: str, keypoints: int = rr.MAX_KEYPOINTS) -> Path:
    from PIL import Image

    queries, gallery, embedder, vectors = load(index, device)
    locator = GeoLocator(embedder, gallery, top_k=5, depth=k)
    feature_dir = WORK / (
        "features" if keypoints == rr.MAX_KEYPOINTS else f"features_kp{keypoints}"
    )
    verifier = rr.GeometricVerifier(device, cache_dir=feature_dir, max_keypoints=keypoints)
    out_path = pairs_path(k, keypoints)
    done = json.loads(out_path.read_text())["queries"] if out_path.exists() else {}
    root = Path("data/gallery_v2")
    for n, (q, v) in enumerate(zip(queries, vectors, strict=True), 1):
        if q["image_id"] in done:
            continue
        base = locator.locate_vector(v)
        refs = [gallery.refs[i] for i in base["ranked_indices"]]
        with Image.open(DEV_ROOT / q["path"]) as im:
            query_image = im.copy()
        started = time.perf_counter()
        res = verifier.verify(query_image, [(r["sha256"], root / r["path"]) for r in refs])
        done[q["image_id"]] = [
            {
                "index": i,
                "image_id": r["image_id"],
                "lat": r["lat"],
                "lng": r["lng"],
                "similarity": s,
                "inliers": x["inliers"],
                "matches": x["matches"],
            }
            for i, r, s, x in zip(
                base["ranked_indices"], refs, base["ranked_similarities"], res, strict=True
            )
        ]
        verify_seconds = time.perf_counter() - started
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "k": k,
                    "index_sha256": gallery.index_sha256,
                    "verifier": verifier.identity,
                    "queries": done,
                }
            )
        )
        print(
            json.dumps(
                {
                    "query": n,
                    "of": len(queries),
                    "seconds": round(verify_seconds, 1),
                    "top_inliers": max(x["inliers"] for x in res),
                }
            ),
            flush=True,
        )
    return out_path


def apply(cands: list[dict[str, Any]], weight: float, min_inliers: int) -> dict[str, Any]:
    f = rr.finalize(cands, weight, min_inliers)  # Same function the API uses.
    return {
        "ranked": f["ranked"],
        "lat": f["predicted_lat"],
        "lng": f["predicted_lng"],
        "cluster_size": f["cluster_size"],
        "confidence": f["confidence"],
    }


def score_rows(
    queries: list[dict[str, Any]],
    data: dict[str, list[dict[str, Any]]],
    gallery: Gallery,
    weight: float,
    min_inliers: int,
) -> list[dict[str, Any]]:
    results = []
    for q in queries:
        a = apply(data[q["image_id"]], weight, min_inliers)
        results.append(
            {
                "predicted_lat": a["lat"],
                "predicted_lng": a["lng"],
                "ranked_indices": [c["index"] for c in a["ranked"]],
                "confidence": float(np.clip(a["ranked"][0]["score"], 0, 10)),
                "candidates": [
                    {
                        "image_id": c["image_id"],
                        "lat": c["lat"],
                        "lng": c["lng"],
                        "similarity": c["similarity"],
                    }
                    for c in a["ranked"][:5]
                ],
                "_top_inliers": a["ranked"][0]["inliers"],
                "_confidence_state": a["confidence"]["state"],
                "_cluster_size": a["cluster_size"],
            }
        )
    rows = dev_eval.evaluate_rows(queries, gallery.refs, results)
    for row, r in zip(rows, results, strict=True):
        row.update(
            {
                "top_inliers": r["_top_inliers"],
                "confidence_state": r["_confidence_state"],
                "cluster_size": r["_cluster_size"],
            }
        )
    return rows


def reliability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accuracy within 500 m by top-1 inlier band (the confidence signal)."""
    bands = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 10_000)]
    out = []
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["top_inliers"] < hi]
        if sel:
            hits = sum(r["error_m"] <= 500 for r in sel)
            out.append(
                {
                    "top1_inliers": f"{lo}-{hi - 1}" if hi < 10_000 else f">={lo}",
                    "photos": len(sel),
                    "within_500m": geo_eval.rate(hits, len(sel)),
                    "within_100m": sum(r["error_m"] <= 100 for r in sel),
                }
            )
    return out


def tune(index: Path, k: int, device: str) -> Path:
    data = json.loads((WORK / f"dev_pairs_k{k}.json").read_text())["queries"]
    queries, gallery, *_ = load(index, device)
    grid = []
    for w in WEIGHT_GRID:
        rows = score_rows(queries, data, gallery, w, 0)
        grid.append(
            {
                "weight": w,
                "within_100m": sum(r["error_m"] <= 100 for r in rows),
                "within_500m": sum(r["error_m"] <= 500 for r in rows),
                "top1_within_100m": sum(r["true_in_top_k"]["100m"]["1"] for r in rows),
            }
        )
    best = max(grid, key=lambda g: (g["within_100m"], g["within_500m"], -g["weight"]))
    weight = best["weight"]
    rows0 = score_rows(queries, data, gallery, weight, 0)
    thresholds = []
    for t in THRESHOLD_GRID:
        conf = [r for r in rows0 if r["top_inliers"] >= t]
        acc = sum(r["error_m"] <= 500 for r in conf) / len(conf) if conf else 0.0
        thresholds.append(
            {"min_inliers": t, "confident_photos": len(conf), "confident_within_500m": acc}
        )
    ok = [
        t
        for t in thresholds
        if t["confident_photos"] and t["confident_within_500m"] >= TARGET_CONFIDENT_ACCURACY
    ]
    min_inliers = ok[0]["min_inliers"] if ok else THRESHOLD_GRID[-1]
    rows = score_rows(queries, data, gallery, weight, min_inliers)
    plain = {
        q["image_id"]: q
        for q in json.loads(Path("eval/results/dev_historic.json").read_text())["queries"]
    }
    ranking_fail = [i for i, q in plain.items() if q["failure_type"] == "ranking"]
    by_id = {r["image_id"]: r for r in rows}
    fixed = [i for i in ranking_fail if by_id[i]["failure_type"] == "success"]
    lost = [
        i
        for i, q in plain.items()
        if q["failure_type"] == "success" and by_id[i]["failure_type"] != "success"
    ]
    lost500 = [i for i, q in plain.items() if q["error_m"] <= 500 and by_id[i]["error_m"] > 500]
    gained500 = [i for i, q in plain.items() if q["error_m"] > 500 and by_id[i]["error_m"] <= 500]
    report = dev_eval.report(
        queries,
        gallery.refs,
        rows,
        f"dev_historic_rerank_k{k}",
        {
            "index_sha256": gallery.index_sha256,
            "pairs": str(WORK / f"dev_pairs_k{k}.json"),
            "pairs_sha256": file_hash(WORK / f"dev_pairs_k{k}.json"),
        },
    )
    conf = [r for r in rows if r["confidence_state"] == "ok"]
    report["definitions"]["prediction"] = (
        "Score-weighted mean of the top-5 re-ranked candidates within 100 m of the top one"
    )
    report["rerank"] = {
        "k": k,
        "score": "cosine + w * log(1 + RANSAC inliers)",
        "weight": weight,
        "weight_grid": grid,
        "selection_rule": "max dev top-1-region within 100 m, then "
        "within 500 m, then smaller w (fixed before tuning)",
        "min_inliers": min_inliers,
        "threshold_grid": thresholds,
        "threshold_rule": f"smallest threshold with confident subset >= "
        f"{TARGET_CONFIDENT_ACCURACY:.0%} within 500 m on dev",
        "disagree_m": rr.DISAGREE_M,
        "cluster_radius_m": rr.CLUSTER_RADIUS_M,
        "reliability_by_top1_inliers": reliability(rows),
        "confidence_states": {
            "ok": {
                "photos": len(conf),
                "within_500m": geo_eval.rate(sum(r["error_m"] <= 500 for r in conf), len(conf))
                if conf
                else None,
            },
            "low": {
                "photos": len(rows) - len(conf),
                "within_500m": geo_eval.rate(
                    sum(r["error_m"] <= 500 for r in rows if r["confidence_state"] == "low"),
                    len(rows) - len(conf),
                )
                if len(rows) > len(conf)
                else None,
            },
        },
        "versus_plain_historic": {
            "ranking_failures_before": len(ranking_fail),
            "ranking_failures_fixed": len(fixed),
            "fixed_ids": fixed,
            "previous_successes_lost": len(lost),
            "lost_ids": lost,
            "within_500m_gained": len(gained500),
            "within_500m_lost": len(lost500),
            "lost_500m_ids": lost500,
        },
        "tuned_on": "66 dev photos only; test set untouched",
        "created_at": datetime.now(UTC).isoformat(),
    }
    out = Path("eval/results/dev_historic_rerank.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "weight": weight,
                "min_inliers": min_inliers,
                **{k2: v["count"] for k2, v in report["metrics"]["within"].items()},
                **report["rerank"]["versus_plain_historic"],
            },
            indent=1,
        )
    )
    return out


SWEEP = ((50, 1024), (20, 1024), (10, 1024), (5, 1024), (20, 512), (10, 512))


def split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for state in ("ok", "low"):
        sel = [r for r in rows if r["confidence_state"] == state]
        hits = sum(r["error_m"] <= 500 for r in sel)
        out[state] = {
            "n": len(sel),
            "within_500m": hits,
            "accuracy": hits / len(sel) if sel else None,
        }
    return out


def timed_queries(
    gallery: Gallery,
    embedder: Any,
    k: int,
    keypoints: int,
    queries_n: int,
    warm: bool,
    device: str = "mps",
) -> dict[str, float]:
    """Median per-query seconds; ``warm`` = reference features already stored on disk.

    The gallery (~3 GB of descriptors) and encoder are loaded once by the caller.
    """
    from PIL import Image

    queries = json.loads((DEV_ROOT / "manifest.json").read_text())["queries"][:queries_n]
    locator = GeoLocator(embedder, gallery, top_k=5, depth=k)
    feature_dir = WORK / (
        "features" if keypoints == rr.MAX_KEYPOINTS else f"features_kp{keypoints}"
    )
    verifier = rr.GeometricVerifier(
        device, cache_dir=feature_dir if warm else None, max_keypoints=keypoints
    )
    root = Path("data/gallery_v2")
    totals, verify = [], []
    for q in queries:
        data = (DEV_ROOT / q["path"]).read_bytes()
        t0 = time.perf_counter()
        base = locator.locate_bytes(data)
        t1 = time.perf_counter()
        refs = [gallery.refs[i] for i in base["ranked_indices"]]
        with Image.open(DEV_ROOT / q["path"]) as im:
            verifier.verify(
                im.copy(), [(r["sha256"] if warm else None, root / r["path"]) for r in refs]
            )
        t2 = time.perf_counter()
        totals.append(t2 - t0)
        verify.append(t2 - t1)
    return {
        "total_s": round(statistics.median(totals[1:] or totals), 2),
        "verification_s": round(statistics.median(verify[1:] or verify), 2),
    }


def sweep(index: Path, device: str, queries_n: int) -> Path:
    """Accuracy/confidence per (K, keypoints) with the fixed w and threshold, plus latency."""
    queries, gallery, embedder, _ = load(index, device)
    out = WORK / "sweep.json"
    table = json.loads(out.read_text())["rows"] if out.exists() else []
    for k, kp in SWEEP:
        if any(r["k"] == k and r["keypoints"] == kp for r in table):
            continue  # Resume: setting already measured.
        # K=10/20 are prefixes of the larger runs (MegaLoc order is unchanged).
        source = pairs_path(50, kp) if kp == rr.MAX_KEYPOINTS else pairs_path(20, kp)
        data = {qid: c[:k] for qid, c in json.loads(source.read_text())["queries"].items()}
        rows = score_rows(queries, data, gallery, rr.DEFAULT_WEIGHT, rr.DEFAULT_MIN_INLIERS)
        thresholds = []
        for t in THRESHOLD_GRID:
            conf = [r for r in rows if r["top_inliers"] >= t]
            acc = sum(r["error_m"] <= 500 for r in conf) / len(conf) if conf else 0.0
            thresholds.append((t, len(conf), acc))
        ok = [t for t, n, acc in thresholds if n and acc >= TARGET_CONFIDENT_ACCURACY]
        row = {
            "k": k,
            "keypoints": kp,
            "within_100m": sum(r["error_m"] <= 100 for r in rows),
            "within_500m": sum(r["error_m"] <= 500 for r in rows),
            "confidence_split_t25": split(rows),
            "rule_threshold": ok[0] if ok else THRESHOLD_GRID[-1],
        }
        row["latency_mps_cold"] = timed_queries(gallery, embedder, k, kp, queries_n, warm=False)
        row["latency_mps_warm"] = timed_queries(gallery, embedder, k, kp, queries_n, warm=True)
        print(json.dumps(row), flush=True)
        table.append(row)
        out.write_text(
            json.dumps(
                {
                    "weight": rr.DEFAULT_WEIGHT,
                    "min_inliers": rr.DEFAULT_MIN_INLIERS,
                    "latency_queries": queries_n,
                    "rows": table,
                },
                indent=1,
            )
        )
    out.write_text(
        json.dumps(
            {
                "weight": rr.DEFAULT_WEIGHT,
                "min_inliers": rr.DEFAULT_MIN_INLIERS,
                "latency_queries": queries_n,
                "rows": table,
            },
            indent=1,
        )
    )
    return out


def latency(index: Path, k: int, queries_n: int) -> dict[str, Any]:
    """Cold per-query latency (no feature cache): MegaLoc + search + K verifications."""
    from PIL import Image

    queries = json.loads((DEV_ROOT / "manifest.json").read_text())["queries"][:queries_n]
    gallery = Gallery(index)
    root = Path("data/gallery_v2")
    out = {}
    for name, enc, ext, match in (("mps", "mps", "mps", "cpu"), ("cpu", "cpu", "cpu", "cpu")):
        embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), enc)
        locator = GeoLocator(embedder, gallery, top_k=5, depth=k)
        verifier = rr.GeometricVerifier(ext, cache_dir=None, match_device=match)
        times = {"megaloc_and_search_s": [], "verification_s": []}
        for q in queries:
            data = (DEV_ROOT / q["path"]).read_bytes()
            t0 = time.perf_counter()
            base = locator.locate_bytes(data)
            t1 = time.perf_counter()
            refs = [gallery.refs[i] for i in base["ranked_indices"]]
            with Image.open(DEV_ROOT / q["path"]) as im:
                verifier.verify(im.copy(), [(None, root / r["path"]) for r in refs])
            t2 = time.perf_counter()
            times["megaloc_and_search_s"].append(t1 - t0)
            times["verification_s"].append(t2 - t1)
        out[name] = {k2: round(statistics.median(v[1:] or v), 2) for k2, v in times.items()}
        out[name]["total_s"] = round(sum(out[name].values()), 2)
        out[name]["devices"] = {"megaloc": enc, "aliked": ext, "lightglue": match}
        print(json.dumps({name: out[name]}), flush=True)
    result = {
        "k": k,
        "queries_timed": queries_n,
        "median_per_query": out,
        "note": "cold: reference features extracted per query (no cache)",
    }
    (WORK / f"latency_k{k}.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["pairs", "tune", "latency", "sweep"])
    parser.add_argument("--keypoints", type=int, default=rr.MAX_KEYPOINTS)
    parser.add_argument("--index", type=Path, default=Path(get_settings().gallery_index))
    parser.add_argument("--k", type=int, default=rr.DEFAULT_K)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--latency-queries", type=int, default=6)
    args = parser.parse_args()
    if args.action == "pairs":
        print(pairs(args.index, args.k, args.device, args.keypoints))
    elif args.action == "sweep":
        print(sweep(args.index, args.device, args.latency_queries))
    elif args.action == "tune":
        print(tune(args.index, args.k, args.device))
    else:
        latency(args.index, args.k, args.latency_queries)


if __name__ == "__main__":
    main()
