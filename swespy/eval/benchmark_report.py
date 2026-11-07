"""Render measured paper-reproduction and dense-gallery results without new inference."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval.common import render_map, thumbnail, write_new
from eval.metrics import summarize
from eval.paper_benchmark import recall_metrics
from swespy.model.megaloc import file_hash


def xy(paths: list[Path]) -> np.ndarray:
    """Read the official coordinate-bearing filename format."""
    return np.array([[float(p.name.split("@")[1]), float(p.name.split("@")[2])]
                     for p in paths], dtype=np.float64)


def positives_for(database_xy: np.ndarray, query_xy: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(np.linalg.norm(database_xy - q, axis=1) <= 25) for q in query_xy]


def thinning(
    db: np.ndarray, queries: np.ndarray, db_xy: np.ndarray, query_xy: np.ndarray,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use prespecified uniform gallery subsets, keeping all queries in the denominator."""
    import faiss

    faiss.omp_set_num_threads(1)
    rows = []
    for fraction in protocol["gallery_fractions"]:
        count = max(1, math.ceil(len(db) * fraction))
        for seed in protocol["seeds"]:
            selected = np.sort(np.random.default_rng(seed).permutation(len(db))[:count])
            index = faiss.IndexFlatL2(db.shape[1])
            index.add(db[selected])
            _, pred = index.search(queries, 20)
            measured = recall_metrics(pred, positives_for(db_xy[selected], query_xy))
            rows.append({"fraction": fraction, "reference_count": count,
                         "seed": seed, **measured})
    return rows


def density_plot(path: Path, samples: list[dict[str, Any]]) -> None:
    """Plot measured means and seed ranges, not statistical confidence intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = sorted({r["fraction"] for r in samples})
    groups = [[r for r in samples if r["fraction"] == f] for f in fractions]
    counts = [g[0]["reference_count"] for g in groups]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for key, label, color in ((None, "25m reference coverage", "#637381"),
                              ("1", "Recall@1 (all queries)", "#1664a5"),
                              ("10", "Recall@10 (all queries)", "#16834a")):
        values = [[r["coverage_pct"] if key is None else r["all_query_recall_pct"][key]
                   for r in g] for g in groups]
        ax.plot(counts, [np.mean(v) for v in values], "o-", color=color, label=label)
        ax.fill_between(counts, [min(v) for v in values], [max(v) for v in values],
                        color=color, alpha=0.12)
    ax.set(xlabel="Reference photos in the same area", ylabel="Percent of fixed queries",
           ylim=(0, 100), title="MegaLoc: effect of reference-gallery density")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.text(0.5, 0.01,
             "Bands show min–max across five fixed reference subsets; not confidence intervals.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def dense_map(run: Path, dataset: Path, manifest_path: Path) -> dict[str, Any]:
    protocol = json.loads((run / "protocol.json").read_text())
    result = json.loads((run / "results.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    db_paths = sorted((dataset / "images/test/database").glob("*.jpg"))
    query_paths = sorted((dataset / "images/test/queries").glob("*.jpg"))
    by_name = {Path(r["path"]).name: r for r in manifest["references"] + manifest["queries"]}
    refs = [by_name[p.name] for p in db_paths]
    queries = [by_name[p.name] for p in query_paths]
    for path in db_paths + query_paths:
        if file_hash(path) != by_name[path.name]["sha256"]:
            raise ValueError("Frozen dense image bytes changed")
    arrays = next(run.glob("*/database_descriptors.npy")).parent
    db = np.load(arrays / "database_descriptors.npy", allow_pickle=False)
    query_vectors = np.load(arrays / "queries_descriptors.npy", allow_pickle=False)
    predictions = np.load(run / "predictions.npy", allow_pickle=False)
    if len(db) != len(refs) or len(query_vectors) != len(queries):
        raise ValueError("Saved descriptors do not match the dense split")
    db_coords, query_coords = xy(db_paths), xy(query_paths)
    rows = []
    matching_heading = 0
    for i, (query, path, pred) in enumerate(zip(queries, query_paths, predictions, strict=True)):
        distances = np.linalg.norm(db_coords - query_coords[i], axis=1)
        nearby = np.flatnonzero(distances <= 25)
        if query["heading"] is not None:
            if any(refs[j]["heading"] is not None and
                   abs((refs[j]["heading"] - query["heading"] + 180) % 360 - 180) <= 45
                   for j in nearby):
                matching_heading += 1
        candidates = pred[:10]
        scores = query_vectors[i] @ db[candidates].T
        matches = [{"lat": refs[j]["lat"], "lng": refs[j]["lng"],
                    "similarity_score": float(score), "image_id": refs[j]["image_id"]}
                   for j, score in zip(candidates, scores, strict=True)]
        top = refs[candidates[0]]
        rows.append({
            **query, "filename": path.name, "viewpoint_type": "street",
            "error_m": float(distances[candidates[0]]),
            "top_1_error_m": float(distances[candidates[0]]),
            "best_candidate_error_m": float(distances[candidates].min()),
            "nearest_reference_m": float(distances.min()),
            "top_k_hit": bool((distances[candidates] <= 25).any()),
            "candidate_count": len(candidates), "confidence": float(np.clip(scores[0], 0, 1)),
            "prediction": {"predicted_lat": top["lat"], "predicted_lng": top["lng"],
                           "top_matches": matches}, "preview": thumbnail(path),
        })
    map_report = {
        "model_label": manifest.get("model_label",
                                    "MegaLoc · dense Linköping · top-ranked reference"),
        "prediction_label": "Top-ranked reference", "query_sequence_count":
            manifest["query_sequence_count"], "top_k": 10, "hit_radius_m": 25,
        "metrics": summarize(rows), "queries": rows,
        "metric_definition": "Euclidean XY labels, same 25m criterion as official evaluation",
        "confidence_status": "uncalibrated top-reference similarity",
        "manifest_sha256": file_hash(manifest_path),
    }
    expected = result["metrics"]["all_query_recall_pct"]["10"]
    if abs(map_report["metrics"]["overall"]["top_k_hit_pct"] - expected) > 1e-8:
        raise ValueError("Map metrics differ from official evaluation")
    write_new(run / "map-results.json", map_report)
    render_map(run / "map.html", map_report)
    density_protocol = json.loads((dataset / "density_protocol.json").read_text())
    subsets = thinning(db, query_vectors, db_coords, query_coords, density_protocol)
    for full in (r for r in subsets if r["fraction"] == 1):
        for k, value in result["metrics"]["all_query_recall_pct"].items():
            if abs(full["all_query_recall_pct"][k] - value) > 1e-8:
                raise ValueError("Full-gallery density check differs from official recall")
    density = {"protocol": density_protocol, "protocol_sha256":
               file_hash(dataset / "density_protocol.json"), "samples": subsets,
               "queries_with_25m_and_45deg_reference": matching_heading}
    write_new(run / "density-results.json", density)
    density_plot(run / "density.png", subsets)
    return {"manifest": manifest, "density": density, "protocol": protocol}


def render(run: Path, dataset: Path | None = None, manifest: Path | None = None) -> Path:
    result = json.loads((run / "results.json").read_text())
    metrics = result["metrics"]
    dense = dense_map(run, dataset, manifest) if dataset and manifest else None
    rows = "".join(
        f"<tr><td>Recall@{k}</td><td>{metrics['all_query_recall_pct'][k]:.2f}%</td>"
        f"<td>{metrics['covered_query_recall_pct'][k]:.2f}%</td></tr>"
        for k in ("1", "5", "10", "20")
        if metrics["covered_query_recall_pct"][k] is not None
    )
    extra = ""
    if dense:
        d = dense["density"]
        lines = []
        for fraction in d["protocol"]["gallery_fractions"]:
            sample = [r for r in d["samples"] if r["fraction"] == fraction]
            coverage = float(np.mean([s["coverage_pct"] for s in sample]))
            recalls = {k: float(np.mean([s["all_query_recall_pct"][k] for s in sample]))
                       for k in ("1", "5", "10")}
            lines.append(f"<tr><td>{sample[0]['reference_count']}</td>"
                         f"<td>{coverage:.1f}%</td><td>{recalls['1']:.1f}%</td>"
                         f"<td>{recalls['5']:.1f}%</td><td>{recalls['10']:.1f}%</td></tr>")
        extra = "<p><a href='map.html'>Open attributed query photos and candidate map</a></p>"
        extra += f"<p>{dense['manifest']['reference_count']} gallery photos; "
        extra += f"{dense['manifest']['query_sequence_count']} held-out query sequences. "
        extra += f"{d['queries_with_25m_and_45deg_reference']} queries have a reference "
        extra += "within both 25m and 45° heading difference.</p>"
        extra += "<h2>Reference-density comparison</h2><p>Same area, model and queries. "
        extra += "Means across five fixed random reference subsets; uncovered queries count "
        extra += "as misses. No retraining or query selection based on predictions.</p>"
        extra += "<table><tr><th>References</th><th>25m coverage</th><th>R@1</th>"
        extra += "<th>R@5</th><th>R@10</th></tr>" + "".join(lines) + "</table>"
        extra += "<p><img src='density.png' "
        extra += "alt='Reference density versus measured recall and coverage' "
        extra += "style='width:100%;max-width:1000px'></p>"
        extra += "<p>This is a small same-provider development test. GPS and headings may "
        extra += "contain errors. It does not establish accuracy for arbitrary photos.</p>"
    elif result["dataset"] == "baidu":
        status = "within the declared tolerance" if result["paper_reproduced_within_tolerance"] \
            else "outside the declared tolerance"
        extra = f"<p>Published targets: R@1 87.7%, R@10 98.0%. This run is {status} "
        extra += "of 0.15 percentage points. Full author-provided 5,207/3,208 split, "
        extra += "25m XY positives. This reproduces pretrained evaluation, not training.</p>"
    page = f"""<!doctype html><html lang='en'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>MegaLoc benchmark · {html.escape(result['dataset'])}</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;
color:#17253c;background:#f8f9fb}}p{{line-height:1.6}}table{{width:100%;background:white;
border-collapse:collapse}}th,td{{padding:12px;text-align:right;border-bottom:1px solid #ddd}}
td:first-child,th:first-child{{text-align:left}}a{{color:#165ca3}}</style>
<h1>MegaLoc · {html.escape(result['dataset'])}</h1>
<p>{metrics['query_count']} fixed queries. {metrics['covered_query_count']} have any gallery image
within 25m: <strong>{metrics['coverage_pct']:.2f}% coverage</strong>.</p>
<table><tr><th>Measure</th><th>All queries</th><th>Covered queries only</th></tr>{rows}</table>
{extra}<p>The official label, exact FAISS search and recall code are preserved. Runtime adapters
change batching, pinned loading and caching. No coordinate averaging is used for Recall@K.</p>
<p><a href='results.json'>Raw results</a> · <a href='protocol.json'>Protocol and provenance</a></p>
</html>"""
    path = run / "report.html"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(page)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print("REPORT", render(args.run, args.dataset, args.manifest))


if __name__ == "__main__":
    main()
