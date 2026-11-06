"""Run the authors' fixed MegaLoc VPR benchmark with resumable descriptors on MPS.

Uses unmodified official labels, FAISS search and Recall@K. The execution copy
changes only query batching; a factory injects the pinned, compatibility-adapted
MegaLoc model instead of importing unrelated model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import re
import sys
import time
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.megaloc_compare import DIM, MegaLocEmbedder, checked_vectors, file_hash
from eval.run_eval import write_new

UPSTREAM_REVISION = "9c8dfd496b49a49549323b94a6af204cbee91604"
UPSTREAM = Path("data/benchmarks/upstream") / UPSTREAM_REVISION
RECALLS = [1, 5, 10, 20]


def recall_metrics(predictions: np.ndarray, positives: list[np.ndarray]) -> dict[str, Any]:
    """Measure all-query recall and separately expose queries with no possible match."""
    if len(predictions) != len(positives) or not len(positives):
        raise ValueError("Prediction and positive-label counts must match and be nonempty")
    covered = sum(bool(len(p)) for p in positives)
    hits = {
        k: sum(bool(np.isin(pred[:k], good).any())
               for pred, good in zip(predictions, positives, strict=True))
        for k in RECALLS
    }
    return {
        "query_count": len(positives), "covered_query_count": covered,
        "coverage_pct": 100 * covered / len(positives),
        "all_query_recall_pct": {str(k): 100 * n / len(positives) for k, n in hits.items()},
        "covered_query_recall_pct":
            {str(k): 100 * n / covered if covered else None for k, n in hits.items()},
    }


class DescriptorCache(torch.nn.Module):
    """Cache normalized-input descriptors without changing the encoder's outputs."""

    def __init__(self, encoder: torch.nn.Module, root: Path, identity: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = encoder
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.root = root / digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.computed = 0
        self.last_progress = time.monotonic()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cpu = images.detach().cpu().contiguous().numpy()
        keys = [hashlib.sha256(sample.tobytes()).hexdigest() for sample in cpu]
        values = np.zeros((len(images), DIM), dtype=np.float32)
        pending = []
        for i, key in enumerate(keys):
            path = self.root / f"{key}.npy"
            if path.exists():
                values[i] = checked_vectors(np.load(path, allow_pickle=False)[None, :], 1)[0]
                self.hits += 1
            else:
                pending.append(i)
        if pending:
            descriptors = checked_vectors(
                self.encoder(images[pending]).detach().cpu().numpy(), len(pending)
            )
            for i, vector in zip(pending, descriptors, strict=True):
                values[i] = vector
                path = self.root / f"{keys[i]}.npy"
                temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("xb") as stream:
                        np.save(stream, vector, allow_pickle=False)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            self.computed += len(pending)
        if time.monotonic() - self.last_progress >= 20:
            print(f"Descriptor progress: {self.computed} encoded, {self.hits} cached", flush=True)
            self.last_progress = time.monotonic()
        return torch.from_numpy(values).to(images.device)


def dataset_identity(paths: list[str], root: Path) -> str:
    """Fingerprint the exact downloaded split and its encoded coordinate labels."""
    digest = hashlib.sha256()
    for name in paths:
        path = Path(name)
        row = [str(path.resolve().relative_to(root.resolve())), file_hash(path)]
        digest.update(json.dumps(row).encode() + b"\n")
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    """Execute official evaluation and cross-check its displayed recalls independently."""
    import faiss

    sys.path.insert(0, str(UPSTREAM.resolve()))
    test_module = importlib.import_module("test_dataset")
    dataset = test_module.TestDataset(
        str(args.dataset / "images/test/database"), str(args.dataset / "images/test/queries"),
        positive_dist_threshold=25, image_size=[322, 322], use_labels=True,
    )
    if args.name == "baidu" and (dataset.num_database, dataset.num_queries) != (5207, 3208):
        raise ValueError("This is not the authors' exact 5207/3208 Baidu split")
    before = dataset_identity(dataset.images_paths, args.dataset)
    positives = dataset.get_positives()
    print(f"Dataset verified: {dataset.num_database} references, {dataset.num_queries} queries; "
          f"{sum(bool(len(p)) for p in positives)} have positives within 25m", flush=True)
    embedder = MegaLocEmbedder(Path("data/megaloc"), args.device)
    probe = embedder.probe([Path(dataset.database_paths[0]), Path(dataset.queries_paths[0])])
    print("CPU/device probe:", json.dumps(probe), flush=True)
    # All layers normalize/aggregate per image. Check batch-size invariance explicitly.
    probe_paths = [Path(p) for p in dataset.database_paths[:args.batch_size]]
    together = embedder.batch(probe_paths)
    alone = np.vstack([embedder.batch([p]) for p in probe_paths[:4]])
    batch_delta = float(np.max(np.abs(together[:len(alone)] - alone)))
    if batch_delta > 0.001:
        raise RuntimeError("Batching changes descriptors materially")
    identity = {**embedder.identity, "official_batch_size": args.batch_size,
                "cache_format": "normalized-input-bytes-v1"}
    cached_model = DescriptorCache(
        embedder.model, Path("data/benchmarks/descriptor_cache"), identity
    )
    factory = types.ModuleType("vpr_models")

    def get_model(method, backbone=None, descriptors_dimension=None):
        if method != "megaloc" or descriptors_dimension != DIM:
            raise ValueError("Only the pinned MegaLoc model is permitted")
        return cached_model

    factory.get_model = get_model  # type: ignore[attr-defined]
    sys.modules["vpr_models"] = factory
    original = (UPSTREAM / "main.py").read_text()
    old = "queries_dataloader = DataLoader(dataset=queries_subset_ds, "
    old += "num_workers=args.num_workers, "
    old += "batch_size=1)"
    if original.count(old) != 1:
        raise ValueError("Official query batching code changed; inspect before adapting")
    portable = original.replace(old, old.replace("batch_size=1)", "batch_size=args.batch_size)"))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = Path("data/benchmarks/runs") / f"{args.name}-{timestamp}"
    folder.mkdir(parents=True)
    (folder / "main.execution.py").write_text(portable)
    write_new(folder / "protocol.json", {
        "dataset": args.name, "dataset_sha256": before,
        "reference_count": dataset.num_database, "query_count": dataset.num_queries,
        "official_repository": "https://github.com/gmberton/VPR-methods-evaluation",
        "official_revision": UPSTREAM_REVISION,
        "official_source_sha256": {p.name: file_hash(p) for p in UPSTREAM.glob("*.py")},
        "execution_main_sha256": file_hash(folder / "main.execution.py"),
        "execution_changes": ["Query batching only; labels/search/recall unchanged",
                              "Pinned MegaLoc loader with documented MPS adapters",
                              "Value-preserving descriptor cache for interruption recovery"],
        "model_identity": identity, "cpu_device_probe": probe,
        "batch_size_max_abs_difference": batch_delta,
        "metric": "Recall@1/5/10/20; Euclidean XY positive radius 25m; all queries included",
        "paper_target": {"R@1": 87.7, "R@10": 98.0} if args.name == "baidu" else None,
        "reproduction_tolerance_percentage_points": 0.15,
    })
    official_args = argparse.Namespace(
        method="megaloc", backbone="Dinov2", descriptors_dimension=DIM,
        database_folder=str((args.dataset / "images/test/database").resolve()),
        queries_folder=str((args.dataset / "images/test/queries").resolve()),
        positive_dist_threshold=25, image_size=[322, 322], use_labels=True,
        device=args.device, batch_size=args.batch_size, num_workers=0,
        log_dir=str(folder.resolve()), recall_values=RECALLS,
        save_descriptors=True, num_preds_to_save=0, save_only_wrong_preds=False,
    )
    namespace: dict[str, Any] = {
        "__name__": "official_megaloc_evaluation", "__file__": str(UPSTREAM / "main.py"),
    }
    exec(compile(portable, str(folder / "main.execution.py"), "exec"), namespace)
    start = time.perf_counter()
    # Separate Torch/FAISS OpenMP runtimes can deadlock after MPS inference on macOS.
    faiss.omp_set_num_threads(1)
    namespace["main"](official_args)
    elapsed = time.perf_counter() - start
    log_folder = next(folder.glob("*/info.log")).parent
    db = np.load(log_folder / "database_descriptors.npy", allow_pickle=False)
    queries = np.load(log_folder / "queries_descriptors.npy", allow_pickle=False)
    checked_vectors(db, dataset.num_database)
    checked_vectors(queries, dataset.num_queries)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatL2(DIM)
    index.add(db)
    _, predictions = index.search(queries, max(RECALLS))
    metrics = recall_metrics(predictions, positives)
    displayed = dict(re.findall(r"R@(\d+): ([0-9.]+)", (log_folder / "info.log").read_text()))
    for k in RECALLS:
        if abs(float(displayed[str(k)]) - metrics["all_query_recall_pct"][str(k)]) > 0.051:
            raise RuntimeError("Independent metric check differs from official output")
    if dataset_identity(dataset.images_paths, args.dataset) != before:
        raise RuntimeError("Dataset changed during benchmark")
    deltas = None
    if args.name == "baidu":
        deltas = {"1": metrics["all_query_recall_pct"]["1"] - 87.7,
                  "10": metrics["all_query_recall_pct"]["10"] - 98.0}
    report = {
        "dataset": args.name, "metrics": metrics, "official_displayed_recall": displayed,
        "paper_deltas_percentage_points": deltas,
        "paper_reproduced_within_tolerance": all(abs(v) <= 0.15 for v in deltas.values())
            if deltas else None,
        "elapsed_seconds": elapsed, "newly_encoded": cached_model.computed,
        "cache_hits": cached_model.hits, "dataset_sha256": before,
        "protocol_sha256": file_hash(folder / "protocol.json"),
        "driver_sha256": file_hash(Path(__file__)),
        "package_versions": {d.metadata["Name"]: d.version
                             for d in importlib.metadata.distributions()},
    }
    np.save(folder / "predictions.npy", predictions, allow_pickle=False)
    write_new(folder / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("BENCHMARK_REPORT", folder / "results.json", flush=True)
    return folder / "results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmarks/baidu"))
    parser.add_argument("--name", default="baidu")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error("Batch size must be 1–32")
    run(args)


if __name__ == "__main__":
    main()
