"""Evaluate the development photo set with CIs, per-photo diagnostics and baselines."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval import geo_eval
from eval.megaloc_compare import MegaLocEmbedder, file_hash
from eval.public_photos import INDEX, ROOT

DEPTH = geo_eval.RETRIEVAL_DEPTH


def load_index(index_manifest: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    import faiss

    saved = json.loads(index_manifest.read_text())
    path = index_manifest.parent / "gallery.faiss"
    if file_hash(path) != saved["index_sha256"]:
        raise ValueError("Exported search index changed")
    index = faiss.read_index(str(path))
    refs = saved["references_in_index_order"]
    if index.ntotal != len(refs):
        raise ValueError("Index and reference mapping differ")
    return index, refs, saved


def evaluate_rows(
    queries: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    ranked: list[list[int]],
    similarities: list[list[float]],
) -> list[dict[str, Any]]:
    """Score top-1 retrieval for each query; ``ranked`` holds reference indices."""
    lats = np.array([r["lat"] for r in refs])
    lngs = np.array([r["lng"] for r in refs])
    rows = []
    for q, order, sims in zip(queries, ranked, similarities, strict=True):
        nearest = float(geo_eval.haversine_m(q["lat"], q["lng"], lats, lngs).min())
        coords = [(refs[i]["lat"], refs[i]["lng"]) for i in order]
        row = geo_eval.diagnose(q["lat"], q["lng"], coords[0], coords, nearest)
        rows.append({
            "image_id": q["image_id"], "creator": q["creator"],
            "capture_date": q["capture_date"], "lat": q["lat"], "lng": q["lng"],
            **row,
            "top1_similarity": sims[0],
            "top5": [{"image_id": refs[i]["image_id"], "lat": refs[i]["lat"],
                      "lng": refs[i]["lng"], "similarity": s}
                     for i, s in zip(order[:5], sims[:5], strict=True)],
        })
    return rows


def report(
    queries: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    label: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    majority = geo_eval.majority_location(refs)
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
            "majority_location": "Mean position of the most populated 1 km UTM gallery cell "
                                 "(no evaluation labels used)",
            "random_reference": "Exact expectation of predicting a uniformly random reference",
        },
        "reference_count": len(refs),
        "metrics": geo_eval.summarize(rows),
        "baselines": {
            "majority_location": {**geo_eval.constant_baseline(
                queries, majority["lat"], majority["lng"]), "location": majority},
            "random_reference": geo_eval.random_reference_baseline(queries, refs),
        },
        "dev_set_distribution": geo_eval.distribution(queries),
        "provenance": provenance,
        "queries": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--label", default="baseline_v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.dev_root / "manifest.json"
    queries = json.loads(manifest_path.read_text())["queries"]
    index, refs, saved = load_index(args.index)
    embedder = MegaLocEmbedder(Path("data/megaloc"), args.device)
    if any(saved["model_identity"].get(k) != v for k, v in embedder.identity.items()):
        raise ValueError("Encoder identity differs from the indexed gallery")
    vectors = embedder.embed([args.dev_root / q["path"] for q in queries],
                             [q["sha256"] for q in queries], batch_size=4)
    import faiss

    faiss.omp_set_num_threads(1)
    distances, indices = index.search(vectors, min(DEPTH, len(refs)))
    rows = evaluate_rows(queries, refs, indices.tolist(), (1 - distances / 2).tolist())
    result = report(queries, refs, rows, args.label, {
        "dev_manifest_sha256": file_hash(manifest_path),
        "index_sha256": saved["index_sha256"],
        "model_identity": embedder.identity,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    m = result["metrics"]
    for radius, value in m["within"].items():
        print(f"within {radius}: {value['count']}/{value['n']} "
              f"[{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]")
    print("failure types:", {k: v["count"] for k, v in m["failure_type"].items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
