"""Controlled geographic-head training on cached MegaLoc references; public photos held out."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.megaloc_compare import checked_vectors, file_hash
from eval.metrics import distance_m
from eval.public_photos import INDEX, ROOT
from eval.run_eval import write_new
from swespy.model.geo_head import GeoHead, fit_head, grid_label, sequence_split

OUT = Path("data/classifier/linkoping_linear_v1")
PUBLIC_REPORT = ROOT / "runs/20261106T135549400456Z/results.json"


def metric_rows(
    points: list[list[dict[str, Any]]], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    errors = [
        [distance_m(q["lat"], q["lng"], p["lat"], p["lng"]) for p in candidates[:5]]
        for q, candidates in zip(queries, points, strict=True)
    ]
    return {
        "query_count": len(queries),
        "median_error_m": float(np.median([r[0] for r in errors])),
        "radii": {
            str(radius): {
                "top1_hits": sum(row[0] <= radius for row in errors),
                "top5_hits": sum(min(row) <= radius for row in errors),
            }
            for radius in (25, 100, 500, 1000, 2000)
        },
    }


def reference_points(indices: np.ndarray, refs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        [{k: refs[int(i)][k] for k in ("image_id", "lat", "lng")} for i in row[:5]]
        for row in indices
    ]


def run(out: Path = OUT) -> None:
    import faiss
    import utm

    faiss.omp_set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        print("Completed experiment exists:", out / "results.json")
        return
    saved = json.loads(INDEX.read_text())
    refs = saved["references_in_index_order"]
    arrays = next(INDEX.parent.glob("*/database_descriptors.npy"))
    features = checked_vectors(np.load(arrays, allow_pickle=False), len(refs))
    labels_by_row = [grid_label(r["easting"], r["northing"]) for r in refs]
    labels = sorted(set(labels_by_row))
    mapping = {label: i for i, label in enumerate(labels)}
    targets = np.array([mapping[label] for label in labels_by_row], dtype=np.int64)
    folds = sequence_split(labels_by_row, [r["sequence"] for r in refs])
    if {(r["utm_zone"], r["utm_letter"]) for r in refs} != {(33, "V")}:
        raise ValueError("This experiment is limited to the Linkoping UTM33V gallery")
    centers = [
        list(
            utm.to_latlon(
                (int(label.split(":")[0]) + 0.5) * 1000,
                (int(label.split(":")[1]) + 0.5) * 1000,
                33,
                "V",
            )
        )
        for label in labels
    ]
    public_manifest = ROOT / "manifest.json"
    heldout = json.loads(public_manifest.read_text())
    if {r["sha256"] for r in refs} & {q["sha256"] for q in heldout["queries"]}:
        raise ValueError("Public test image occurs in training gallery")
    protocol = {
        "kind": "frozen_megaloc_linear_geographic_head",
        "grid_size_m": 1000,
        "encoder_identity": saved["model_identity"],
        "encoder_updates": False,
        "descriptor_sha256": file_hash(arrays),
        "index_manifest_sha256": file_hash(INDEX),
        "heldout_manifest_sha256": file_hash(public_manifest),
        "seed": 42,
        "split": "70/15/15 capture groups; move whole groups to training only when "
        "needed to cover every output class; no frame or sequence overlap",
        "fold_ids": [[refs[int(i)]["image_id"] for i in fold] for fold in folds],
        "class_labels": labels,
        "fusion_candidates": 50,
        "fusion_weights": [0.0, 0.02, 0.05, 0.1, 0.2],
        "fusion_selection": "Mapillary validation: max first-choice hits500m; "
        "then min median error; then smaller weight. Train-only retrieval gallery.",
        "head_selection": "Mapillary validation cell accuracy then cross entropy; "
        "AdamW lr0.03 decay0.01 inverse-sqrt class weights;batch256;epochs<=60;patience8",
        "public_evaluation": "Once after training/validation selection; all66 retained; "
        "classifier only, full-gallery MegaLoc, combined reranking",
        "limitations": "Mapillary training and validation differ from Commons photos. "
        "Public set previously inspected for development; not a new blind benchmark.",
    }
    if (out / "protocol.json").exists():
        if json.loads((out / "protocol.json").read_text()) != protocol:
            raise ValueError("Training protocol changed; create a new experiment directory")
    else:
        write_new(out / "protocol.json", protocol)
    print("Fixed protocol:", len(labels), "classes; folds", [len(f) for f in folds], flush=True)
    head, training = fit_head(features, targets, folds, labels, centers, saved["model_identity"])
    faiss.omp_set_num_threads(1)
    train, val, test = folds
    train_index = faiss.IndexFlatL2(features.shape[1])
    train_index.add(features[train])
    distances, local_indices = train_index.search(features[val], min(50, len(train)))
    candidates = train[local_indices]
    val_queries = [refs[int(i)] for i in val]
    fusion_trials = []
    for weight in protocol["fusion_weights"]:
        ranked = head.rerank(features[val], candidates, 1 - distances / 2, refs, weight)
        metrics = metric_rows(reference_points(ranked, refs), val_queries)
        fusion_trials.append({"weight": weight, "metrics": metrics})
    selected = max(
        fusion_trials,
        key=lambda t: (
            t["metrics"]["radii"]["500"]["top1_hits"],
            -t["metrics"]["median_error_m"],
            -t["weight"],
        ),
    )
    head.checkpoint["fusion_weight"] = selected["weight"]
    checkpoint = out / "head.pt"
    if checkpoint.exists():
        raise ValueError("Refusing to overwrite an existing checkpoint")
    torch.save(head.checkpoint, checkpoint)
    loaded = GeoHead.load(checkpoint, saved["model_identity"])
    np.testing.assert_allclose(
        loaded.probabilities(features[val]), head.probabilities(features[val]), atol=0, rtol=0
    )
    d_test, pred_test = train_index.search(features[test], min(50, len(train)))
    global_test = train[pred_test]
    internal_test = {
        "classifier": metric_rows(loaded.locations(features[test]), [refs[int(i)] for i in test]),
        "retrieval": metric_rows(reference_points(global_test, refs), [refs[int(i)] for i in test]),
        "combined": metric_rows(
            reference_points(
                loaded.rerank(features[test], global_test, 1 - d_test / 2, refs), refs
            ),
            [refs[int(i)] for i in test],
        ),
    }
    # Read public query descriptors only AFTER checkpoint/weight selection has finished.
    old_report = json.loads(PUBLIC_REPORT.read_text())
    identity = old_report["model_identity"]
    if any(saved["model_identity"].get(k) != v for k, v in identity.items()):
        raise ValueError("Held-out query descriptors use a different encoder")
    cache = (
        Path("data/megaloc/descriptors")
        / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    )
    queries = heldout["queries"]
    query_features = checked_vectors(
        np.vstack([np.load(cache / f"{q['sha256']}.npy", allow_pickle=False) for q in queries]),
        len(queries),
    )
    full = faiss.read_index(str(INDEX.parent / "gallery.faiss"))
    distances, candidates = full.search(query_features, 50)
    pure = reference_points(candidates, refs)
    combined = reference_points(
        loaded.rerank(query_features, candidates, 1 - distances / 2, refs), refs
    )
    classification = loaded.locations(query_features)
    results = {
        "reference_count": len(refs),
        "class_count": len(labels),
        "training": training,
        "selected_fusion_weight": selected["weight"],
        "validation_trials": fusion_trials,
        "internal_test": internal_test,
        "public": {
            "classifier": metric_rows(classification, queries),
            "retrieval": metric_rows(pure, queries),
            "combined": metric_rows(combined, queries),
        },
        "queries": [
            {"image_id": q["image_id"], "classifier": c, "retrieval": r, "combined": f}
            for q, c, r, f in zip(queries, classification, pure, combined, strict=True)
        ],
        "checkpoint_sha256": file_hash(checkpoint),
        "protocol_sha256": file_hash(out / "protocol.json"),
        "checkpoint_roundtrip_exact": True,
    }
    for radius, metrics in old_report["radius_metrics"].items():
        for key in ("top1_hits", "top5_hits"):
            if results["public"]["retrieval"]["radii"][radius][key] != metrics[key]:
                raise ValueError("Retrieval baseline changed")
    if (
        file_hash(arrays) != protocol["descriptor_sha256"]
        or file_hash(public_manifest) != protocol["heldout_manifest_sha256"]
    ):
        raise ValueError("Frozen inputs changed during training")
    write_new(out / "results.json", results)
    print(
        json.dumps(
            {
                "training": {k: v for k, v in training.items() if k != "history"},
                "fusion_weight": selected["weight"],
                "public": results["public"],
            },
            indent=2,
        )
    )
    print("TRAINED_HEAD_RESULTS", out / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out)
