"""Experiment 15: approximate gallery search vs the exact scan, on the 66 dev photos.

    PYTORCH_ENABLE_MPS_FALLBACK=1 TORCH_HOME=data/torch python -m eval.ann_search select
    PYTORCH_ENABLE_MPS_FALLBACK=1 TORCH_HOME=data/torch python -m eval.ann_search latency

``select`` applies the rule fixed in docs/experiments/15_approximate_search.md: for each
(nprobe, refine) in the grid, the merged retrieval top 5 must equal the exact search for
>= 63 of 66 dev photos, for the whole photo and for the default pipeline with automatic
crops, and dev within 500 m must be unchanged; among passing settings, the lowest median
search time (full pipeline) wins. ``latency`` times the whole pipeline end to end on the
Mac with the exact and the chosen search. Dev set only; stops (keeping results) on low
disk or high swap, like experiment 14.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval.auto_regions import resource_problem
from eval.geo_eval import haversine_m
from eval.gsv_eval import load_sets

OUT = Path("data/auto_regions/ann")
# Grids fixed before each family was measured (docs/experiments/15_approximate_search.md).
GRIDS = {
    "ivfpq": [(n, r) for n in (8, 16, 32, 64, 128) for r in (4, 10)],
    "ivf": [(n, 1) for n in (16, 32, 64, 128, 256)],  # refine is unused by "ivf".
}
MIN_IDENTICAL = 63  # >= 95% of 66.
CORRECT_M = 500.0
LATENCY_N = 20


def guard() -> None:
    problem = resource_problem()
    if problem:
        raise SystemExit(f"Stopped: {problem}")


def models(family: str = "ivfpq") -> tuple[Any, Any, Any, Any, Any]:
    from swespy.config import get_settings
    from swespy.model import rerank as rr
    from swespy.model.locator import Gallery
    from swespy.model.megaloc import MegaLocEmbedder
    from swespy.model.variants import PersonSegmenter

    s = get_settings()
    exact = Gallery(Path(s.gallery_index), s.area_polygon)
    started = time.perf_counter()
    approx = Gallery(Path(s.gallery_index), s.area_polygon, search=family)
    print(f"approximate index ready in {time.perf_counter() - started:.0f} s", flush=True)
    return (exact, approx, MegaLocEmbedder(Path(s.megaloc_root), s.megaloc_device),
            PersonSegmenter(s.segment_device), rr.GeometricVerifier(s.rerank_device))


def locator(gallery: Any, embedder: Any, segmenter: Any, verifier: Any, mode: str) -> Any:
    from swespy.config import get_settings
    from swespy.model import rerank as rr
    from swespy.model.locator import GeoLocator

    s = get_settings()
    return GeoLocator(embedder, gallery, s.top_k_results, s.retrieval_depth, verifier=verifier,
                      weight=rr.DEFAULT_WEIGHT, min_inliers=rr.DEFAULT_MIN_INLIERS,
                      rerank_k=s.rerank_k, segmenter=segmenter, variants_mode=mode)


def top5(gallery: Any, vectors: np.ndarray, priorities: list[int], depth: int) -> tuple[
        list[int], float]:
    """Merged retrieval top 5 exactly as GeoLocator computes it, and the search time."""
    from swespy.model import variants as va

    started = time.perf_counter()
    indices, sims = gallery.search(vectors, depth)
    seconds = time.perf_counter() - started
    merged = va.merge([(i.tolist(), s.tolist()) for i, s in zip(indices, sims, strict=True)],
                      priorities)
    return [i for i, _, _ in merged[:5]], seconds


def select(family: str) -> dict[str, Any]:
    from swespy.config import get_settings
    from swespy.model import variants as va
    from swespy.model.imageio import decode_upload

    OUT.mkdir(parents=True, exist_ok=True)
    depth = get_settings().retrieval_depth
    exact, approx, embedder, segmenter, verifier = models(family)
    queries = load_sets()["dev"]
    # Query vectors exactly as the pipeline makes them: the whole photo alone, and all
    # variants of the full pipeline in one batch.
    photos = []
    for q in queries:
        guard()
        image = decode_upload(Path(q["path"]).read_bytes())
        variants = va.make_variants(image, segmenter.mask(image), None, "full")
        photos.append({
            "q": q,
            "whole": (embedder.batch_images([image]), [0]),
            "full": (embedder.batch_images([v.image for v in variants]),
                     [v.priority for v in variants]),
        })
    print("query vectors ready", flush=True)
    base: dict[str, list[list[int]]] = {"whole": [], "full": []}
    base_s: dict[str, list[float]] = {"whole": [], "full": []}
    for p in photos:
        guard()
        for mode in ("whole", "full"):
            ids, sec = top5(exact, *p[mode], depth)
            base[mode].append(ids)
            base_s[mode].append(sec)
    print("exact searches done", flush=True)
    grid = []
    for nprobe, refine in GRIDS[family]:
        approx.nprobe, approx.refine = nprobe, refine
        entry: dict[str, Any] = {"nprobe": nprobe, "refine": refine}
        for mode in ("whole", "full"):
            same, secs, differ = 0, [], []
            for n, p in enumerate(photos):
                ids, sec = top5(approx, *p[mode], depth)
                secs.append(sec)
                if ids == base[mode][n]:
                    same += 1
                else:
                    differ.append(n)
            entry[mode] = {"identical": same, "differ": differ,
                           "search_ms_median": round(1000 * statistics.median(secs), 1)}
        entry["top5_pass"] = all(entry[m]["identical"] >= MIN_IDENTICAL
                                 for m in ("whole", "full"))
        print(json.dumps({k: v for k, v in entry.items() if k != "differ"}), flush=True)
        grid.append(entry)
    # Criterion 2 for settings that pass criterion 1: re-run the photos whose top 5
    # differs, with the exact and the approximate search, through the whole pipeline.
    for entry in grid:
        if not entry["top5_pass"]:
            entry["pass"] = False
            continue
        approx.nprobe, approx.refine = entry["nprobe"], entry["refine"]
        entry["within_500m_delta"] = {}
        for mode in ("whole", "full"):
            delta = 0
            for n in entry[mode]["differ"]:
                guard()
                q = photos[n]["q"]
                raw = Path(q["path"]).read_bytes()
                for gallery, sign in ((exact, -1), (approx, 1)):
                    r = locator(gallery, embedder, segmenter, verifier, mode).locate_bytes(raw)
                    err = float(haversine_m(q["lat"], q["lng"], np.array([r["predicted_lat"]]),
                                            np.array([r["predicted_lng"]]))[0])
                    delta += sign * (err <= CORRECT_M)
            entry["within_500m_delta"][mode] = delta
        entry["pass"] = all(d == 0 for d in entry["within_500m_delta"].values())
    passing = [e for e in grid if e["pass"]]
    chosen = min(passing, key=lambda e: (e["full"]["search_ms_median"], e["nprobe"],
                                          e["refine"])) if passing else None
    report = {
        "family": family,
        "exact_search_ms_median": {m: round(1000 * statistics.median(v), 1)
                                   for m, v in base_s.items()},
        "grid": grid,
        "chosen": None if chosen is None else {"nprobe": chosen["nprobe"],
                                               "refine": chosen["refine"]},
    }
    (OUT / f"select_{family}.json").write_text(json.dumps(report, indent=1))
    return report


def latency(family: str, nprobe: int, refine: int) -> dict[str, Any]:
    exact, approx, embedder, segmenter, verifier = models(family)
    approx.nprobe, approx.refine = nprobe, refine
    queries = load_sets()["dev"][:LATENCY_N]
    out: dict[str, Any] = {"photos": LATENCY_N, "family": family, "nprobe": nprobe,
                           "refine": refine}
    searches = [("exact", exact)] + ([(family, approx)] if nprobe else [])
    for name, gallery in searches:
        for mode in ("whole", "full"):
            loc = locator(gallery, embedder, segmenter, verifier, mode)
            loc.locate_bytes(Path(queries[0]["path"]).read_bytes())  # Warm-up, not timed.
            totals, stages = [], {}
            for q in queries:
                guard()
                started = time.perf_counter()
                r = loc.locate_bytes(Path(q["path"]).read_bytes())
                totals.append(time.perf_counter() - started)
                for k, v in r["timings_ms"].items():
                    stages.setdefault(k, []).append(v)
            out[f"{name}/{mode}"] = {
                "total_s_median": round(statistics.median(totals), 2),
                "total_s_p90": round(float(np.percentile(totals, 90)), 2),
                **{k: round(statistics.median(v)) for k, v in stages.items()},
            }
            print(name, mode, out[f"{name}/{mode}"], flush=True)
    (OUT / "latency.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "latency"))
    parser.add_argument("--family", choices=tuple(GRIDS), default="ivfpq")
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select(args.family)["chosen"]))
    else:
        chosen = json.loads((OUT / f"select_{args.family}.json").read_text())["chosen"]
        if chosen is None:
            print("No setting passed; latency of the exact search only.")
            chosen = {"nprobe": 0, "refine": 0}
        print(json.dumps(latency(args.family, chosen["nprobe"], chosen["refine"]), indent=1))


if __name__ == "__main__":
    main()
