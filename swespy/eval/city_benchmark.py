"""Format the frozen city gallery and old queries; export a searchable MegaLoc index."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from eval.benchmark_report import positives_for, render, xy
from eval.common import write_new
from eval.dense_linkoping import OLD_MANIFEST, formatted_link
from eval.expand_linkoping import CENTRAL, DEFAULT_ROOT
from eval.paper_benchmark import dataset_identity, recall_metrics
from swespy.model.megaloc import DIM, checked_vectors, file_hash

DATASET = Path("data/linkoping_dense/city_eval_v1")


def prepare(root: Path = DATASET) -> None:
    """Hard-link original bytes; never modify source datasets or acquire new queries."""
    import utm

    manifests = [DEFAULT_ROOT/"manifest.json", CENTRAL/"manifest.json", OLD_MANIFEST]
    sources = {str(p): file_hash(p) for p in manifests}
    root.mkdir(parents=True, exist_ok=True)
    frozen = root/"manifest.json"
    if frozen.exists():
        saved = json.loads(frozen.read_text())
        if saved["source_manifests"] != sources:
            raise ValueError("Source manifests changed")
        for row in saved["references"] + saved["queries"]:
            if file_hash(root/row["path"]) != row["sha256"]:
                raise ValueError("Frozen city evaluation image changed")
        print("Validated prepared city evaluation dataset", flush=True)
        return
    city, central, old = [json.loads(p.read_text()) for p in manifests]
    (root/"originals").mkdir(exist_ok=True)

    def link(row: dict[str, Any], source: Path, role: str) -> dict[str, Any]:
        if file_hash(source) != row["sha256"]:
            raise ValueError("Source image changed")
        if "easting" not in row:
            easting, northing, zone, letter = utm.from_latlon(row["lat"], row["lng"])
            row = {**row, "easting": easting, "northing": northing,
                   "utm_zone": zone, "utm_letter": letter, "heading": None}
        original = root/"originals"/f"{row['image_id']}.jpg"
        if not original.exists():
            os.link(source, original)
        if file_hash(original) != row["sha256"]:
            raise ValueError("Image ID collision in evaluation dataset")
        return {**row, "path": formatted_link(root, row, role)}

    refs = [link(r, DEFAULT_ROOT/r["path"], "database") for r in city["references"]]
    queries = [link({**q, "cohort": "central18"}, CENTRAL/q["path"], "queries")
               for q in central["queries"]]
    queries += [link({**q, "cohort": "legacy30"}, OLD_MANIFEST.parent/"photos"/q["filename"],
                     "queries") for q in old["queries"]]
    if len({q["image_id"] for q in queries}) != len(queries):
        raise ValueError("Query sets overlap; do not double-count images")
    if {r["sequence"] for r in refs} & {q["sequence"] for q in queries}:
        raise ValueError("Reference and query capture sequences overlap")
    if {r["sha256"] for r in refs} & {q["sha256"] for q in queries}:
        raise ValueError("Query image bytes occur in gallery")
    write_new(frozen, {
        "kind": "frozen_city_gallery_evaluation", "source_manifests": sources,
        "reference_count": len(refs), "query_count": len(queries),
        "query_sequence_count": len({q["sequence"] for q in queries}),
        "references": refs, "queries": queries,
        "model_label": "MegaLoc · city-wide Linköping · top-ranked reference",
        "cohorts": {"central18": "Original dense central held-out queries",
                    "legacy30": "Original sparse-city held-out queries"},
        "protocol": "All frozen queries included; no selection based on predictions; "
                    "25m official Recall@K; additionally report 100m/500m and cohort results",
        "limitations": "Same-provider development sets; not independent ordinary-photo accuracy",
    })
    write_new(root/"density_protocol.json", {
        "gallery_fractions": [0.1, 0.25, 0.5, 1.0], "seeds": [0, 1, 2, 3, 42],
        "selection": "Uniform fixed subsets; full frozen queries; no training",
    })
    print(f"Prepared {len(refs)} references and {len(queries)} frozen queries", flush=True)


def score_radii(
    predictions: np.ndarray, db_xy: np.ndarray, query_xy: np.ndarray,
) -> dict[str, Any]:
    return {str(radius): recall_metrics(predictions, [
        np.flatnonzero(np.linalg.norm(db_xy-q, axis=1) <= radius) for q in query_xy
    ]) for radius in (25, 100, 500)}


def finalize(run: Path, dataset: Path = DATASET) -> None:
    """Verify arrays, publish FAISS/row mapping, and compare identical central queries."""
    import faiss

    faiss.omp_set_num_threads(1)
    manifest = json.loads((dataset/"manifest.json").read_text())
    protocol = json.loads((run/"protocol.json").read_text())
    result = json.loads((run/"results.json").read_text())
    db_paths = sorted((dataset/"images/test/database").glob("*.jpg"))
    query_paths = sorted((dataset/"images/test/queries").glob("*.jpg"))
    if dataset_identity([str(p) for p in db_paths+query_paths], dataset) != \
            protocol["dataset_sha256"]:
        raise ValueError("Evaluation dataset differs from the encoded dataset")
    by_name = {Path(r["path"]).name: r for r in manifest["references"]+manifest["queries"]}
    refs = [by_name[p.name] for p in db_paths]
    queries = [by_name[p.name] for p in query_paths]
    arrays = next(run.glob("*/database_descriptors.npy")).parent
    db = checked_vectors(np.load(arrays/"database_descriptors.npy", allow_pickle=False), len(refs))
    qv = checked_vectors(np.load(arrays/"queries_descriptors.npy", allow_pickle=False),
                         len(queries))
    pred = np.load(run/"predictions.npy", allow_pickle=False)
    coords, qcoords = xy(db_paths), xy(query_paths)
    if recall_metrics(pred, positives_for(coords, qcoords)) != result["metrics"]:
        raise ValueError("Saved predictions do not match official metrics")
    index = faiss.IndexFlatL2(DIM)
    index.add(db)
    _, checked = index.search(qv, 20)
    if not np.array_equal(pred, checked):
        raise ValueError("Rebuilt index differs from saved predictions")
    index_path = run/"gallery.faiss"
    if index_path.exists():
        restored = faiss.read_index(str(index_path))
        _, restored_pred = restored.search(qv, 20)
        if restored.ntotal != len(refs) or not np.array_equal(restored_pred, pred):
            raise ValueError("Existing exported index differs")
    else:
        temporary = run/"gallery.faiss.tmp"
        faiss.write_index(index, str(temporary))
        temporary.replace(index_path)
    if not (run/"gallery-index.json").exists():
        write_new(run/"gallery-index.json", {
            "dimension": DIM, "distance": "squared L2 over normalized descriptors",
            "similarity_conversion": "cosine = 1 - squared_L2 / 2",
            "index_sha256": file_hash(index_path),
            "dataset": str(dataset.resolve()), "dataset_sha256": protocol["dataset_sha256"],
            "manifest_sha256": file_hash(dataset/"manifest.json"),
            "model_identity": protocol["model_identity"], "references_in_index_order": refs,
        })
    cohorts = {name: np.array([i for i, q in enumerate(queries) if q["cohort"] == name])
               for name in manifest["cohorts"]}
    measures = {"all48": score_radii(pred, coords, qcoords)}
    measures.update({name: score_radii(pred[indices], coords, qcoords[indices])
                     for name, indices in cohorts.items()})
    central = json.loads((CENTRAL/"manifest.json").read_text())
    central_ids = {r["image_id"] for r in central["references"]}
    subset = np.array([i for i, r in enumerate(refs) if r["image_id"] in central_ids])
    if len(subset) != len(central_ids):
        raise ValueError("Original central references are missing")
    central_index = faiss.IndexFlatL2(DIM)
    central_index.add(db[subset])
    _, baseline = central_index.search(qv[cohorts["central18"]], 20)
    before = score_radii(baseline, coords[subset], qcoords[cohorts["central18"]])
    original_run = Path("data/benchmarks/runs/linkoping_dense-20261106T115241.727823Z")
    original = json.loads((original_run/"results.json").read_text())
    if before["25"] != original["metrics"]:
        raise ValueError("Same-query central baseline no longer matches the recorded run")
    summary = {"metrics_by_cohort_and_radius": measures,
               "central369_same_query_baseline": before,
               "reference_count": len(refs), "query_count": len(queries),
               "manifest_sha256": file_hash(dataset/"manifest.json"),
               "index_sha256": file_hash(index_path),
               "limitations": "Small frozen Mapillary development sets; "
                              "not ordinary-photo accuracy"}
    if not (run/"city-results.json").exists():
        write_new(run/"city-results.json", summary)
    if not (run/"report.html").exists():
        render(run, dataset, dataset/"manifest.json")
    rows = []
    for cohort, metrics in measures.items():
        for radius, m in metrics.items():
            r = m["all_query_recall_pct"]
            rows.append(f"<tr><td>{html.escape(cohort)}</td><td>{m['query_count']}</td>"
                        f"<td>{radius}m</td><td>{m['coverage_pct']:.1f}%</td>"
                        f"<td>{r['1']:.1f}%</td><td>{r['5']:.1f}%</td>"
                        f"<td>{r['10']:.1f}%</td></tr>")
    old_r = before["25"]["all_query_recall_pct"]
    new_r = measures["central18"]["25"]["all_query_recall_pct"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MegaLoc city-wide Linköping results</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;
color:#17253c;background:#f8f9fb}}p{{line-height:1.6}}table{{width:100%;border-collapse:collapse;
background:white}}td,th{{padding:12px;text-align:right;border-bottom:1px solid #ddd}}
td:first-child,th:first-child{{text-align:left}}a{{color:#165ca3}}</style>
<h1>MegaLoc · city-wide Linköping</h1>
<p>{len(refs)} reference images, {len(queries)} frozen queries. Pretrained inference on Mac;
no retraining, API model switch, or query selection based on predictions.</p>
<p>Same 18 central queries, 369 → {len(refs)} references: 25m Recall@1
{old_r['1']:.1f}% → {new_r['1']:.1f}%; Recall@10 {old_r['10']:.1f}% → {new_r['10']:.1f}%.
The original central baseline was independently reproduced from the saved descriptors.</p>
<table><tr><th>Query set</th><th>Queries</th><th>Radius</th><th>Coverage</th>
<th>Recall@1</th><th>Recall@5</th><th>Recall@10</th></tr>{''.join(rows)}</table>
<p>Uncovered queries remain misses. These same-provider development sets do not establish
accuracy on ordinary user photos. Geographic labels are provider GPS.</p>
<p><a href="map.html">Photo and candidate map</a> · <a href="report.html">Official 25m metrics</a>
 · <a href="city-results.json">Full results</a>
 · <a href="gallery-index.json">Index manifest</a></p>
</html>"""
    if not (run/"city-summary.html").exists():
        (run/"city-summary.html").write_text(page)
    print(json.dumps(summary, indent=2), flush=True)
    print("CITY_RESULTS", run/"city-summary.html", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Finalize an already encoded official run")
    args = parser.parse_args()
    if args.run:
        finalize(args.run)
    else:
        prepare()


if __name__ == "__main__":
    main()
