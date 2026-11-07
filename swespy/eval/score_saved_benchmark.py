"""Recover official scoring from fully saved arrays without repeating model inference."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval.common import write_new
from eval.paper_benchmark import RECALLS, UPSTREAM, dataset_identity, recall_metrics
from swespy.model.megaloc import DIM, checked_vectors, file_hash


def main() -> None:
    import faiss
    from loguru import logger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    protocol = json.loads((run / "protocol.json").read_text())
    sys.path.insert(0, str(UPSTREAM.resolve()))
    dataset_class = importlib.import_module("test_dataset").TestDataset
    dataset = dataset_class(
        str(args.dataset / "images/test/database"), str(args.dataset / "images/test/queries"),
        positive_dist_threshold=25, image_size=[322, 322], use_labels=True,
    )
    if dataset_identity(dataset.images_paths, args.dataset) != protocol["dataset_sha256"]:
        raise ValueError("Saved arrays belong to a different dataset")
    arrays = next(run.glob("*/database_descriptors.npy")).parent
    db = checked_vectors(np.load(arrays / "database_descriptors.npy", allow_pickle=False),
                         dataset.num_database)
    queries = checked_vectors(np.load(arrays / "queries_descriptors.npy", allow_pickle=False),
                              dataset.num_queries)
    source = (UPSTREAM / "main.py").read_text()
    tail = source[source.index("    # Use a kNN"):source.index("    # Save visualizations")]
    code = textwrap.dedent(tail)
    (run / "scoring.execution.py").write_text(code)
    faiss.omp_set_num_threads(1)
    namespace: dict[str, Any] = {
        "faiss": faiss, "np": np, "logger": logger,
        "database_descriptors": db, "queries_descriptors": queries,
        "all_descriptors": None, "test_ds": dataset,
        "args": argparse.Namespace(recall_values=RECALLS, use_labels=True,
                                   descriptors_dimension=DIM),
    }
    start = time.perf_counter()
    exec(compile(code, str(run / "scoring.execution.py"), "exec"), namespace)
    predictions = namespace["predictions"]
    metrics = recall_metrics(predictions, dataset.get_positives())
    displayed = {str(k): float(value) for k, value in
                 zip(RECALLS, namespace["recalls"], strict=True)}
    for k in RECALLS:
        if abs(displayed[str(k)] - metrics["all_query_recall_pct"][str(k)]) > 1e-8:
            raise RuntimeError("Independent recall differs from official metric code")
    if dataset_identity(dataset.images_paths, args.dataset) != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed during recovered scoring")
    deltas = {str(k): metrics["all_query_recall_pct"][str(k)] - target
              for k, target in ((1, 87.7), (10, 98.0))} if protocol["dataset"] == "baidu" else None
    result = {
        "dataset": protocol["dataset"], "metrics": metrics,
        "official_displayed_recall": displayed,
        "paper_deltas_percentage_points": deltas,
        "paper_reproduced_within_tolerance":
            all(abs(v) <= 0.15 for v in deltas.values()) if deltas else None,
        "elapsed_seconds": time.perf_counter() - start,
        "timing_scope": "Recovered scoring only; image inference ran in the original worker",
        "descriptor_count": dataset.num_database + dataset.num_queries,
        "dataset_sha256": protocol["dataset_sha256"],
        "protocol_sha256": file_hash(run / "protocol.json"),
        "recovery_driver_sha256": file_hash(Path(__file__)),
        "official_scoring_sha256": file_hash(run / "scoring.execution.py"),
        "recovery_reason": "Original combined MPS/FAISS process stalled in an OpenMP wait; "
                           "saved descriptors scored in a fresh single-thread FAISS process",
        "package_versions": {d.metadata["Name"]: d.version
                             for d in importlib.metadata.distributions()},
    }
    np.save(run / "predictions.npy", predictions, allow_pickle=False)
    write_new(run / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
