"""Compare pinned pretrained MegaLoc with a frozen CLIP street-check report.

Run natively for Apple MPS: .venv/bin/python -u -m eval.megaloc_compare
No training or API configuration changes. All experiment files are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
import platform
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from eval.dataset import Photo
from eval.mapillary_check import reference_points, render_map, verify_manifest
from eval.metrics import distance_m, observation, print_table, summarize
from eval.run_eval import index_fingerprint, write_new
from swespy.config import get_settings
from swespy.vector_store.qdrant_client import VectorStore

SOURCE_REVISION = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
SOURCE_SHA256 = "e3a66ae823abbbbc5e7975c649419a33036400eaa9fa5a955dd97eaa37f46c75"
WEIGHTS_REVISION = "a0f34722c4297ff787e022433799250180860af7"
DIM = 8448
PREPROCESSING = {
    "rgb": True, "to_tensor": True,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "resize_after_normalization": [322, 322], "antialias": True,
    "resize_interpolation": "bilinear", "dtype": "float32",
    "augmentation": "none", "whitening_or_pca": "none",
}
DEFAULT_RUN = Path("data/mapillary_eval/20261106T095121.923253Z")


def file_hash(path: Path) -> str:
    """Hash large weights/files without loading their complete contents into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_comparison(
    manifest_path: Path, clip_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refuse comparisons against a different query set, labels or source index."""
    manifest, clip = json.loads(manifest_path.read_text()), json.loads(clip_path.read_text())
    verify_manifest(manifest, manifest_path.parent)
    if clip["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("CLIP report belongs to a different manifest")
    if clip["index_sha256"] != manifest["index_sha256"]:
        raise ValueError("CLIP reference index differs from the frozen manifest")
    if clip["top_k"] != 5 or clip["hit_radius_m"] != 100:
        raise ValueError("Comparison requires the frozen top-five/100m CLIP protocol")
    expected = [(q["image_id"], q["sha256"], q["lat"], q["lng"])
                for q in manifest["queries"]]
    actual = [(q["image_id"], q["sha256"], q["lat"], q["lng"])
              for q in clip["queries"]]
    if actual != expected:
        raise ValueError("CLIP query IDs, bytes, labels or order differ")
    return manifest, clip


def checked_vectors(vectors: np.ndarray, count: int) -> np.ndarray:
    """Reject malformed descriptors before caching them or writing a collection."""
    if (
        vectors.shape != (count, DIM)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("MegaLoc descriptors have invalid shape, values or normalization")
    return vectors.astype(np.float32, copy=False)


def contiguous_conv_inputs(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Preserve tensor values while avoiding the torch 2.5 MPS Conv2d stride bug."""
    return tuple(value.contiguous() for value in inputs)


def load_assets(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Download immutable upstream assets, verify reviewed source, and retain licenses."""
    from huggingface_hub import hf_hub_download

    source = root / "assets" / SOURCE_REVISION
    source.mkdir(parents=True, exist_ok=True)
    for filename in ("megaloc_model.py", "LICENSE", "README.md"):
        target = source / filename
        if not target.exists():
            response = httpx.get(
                f"https://raw.githubusercontent.com/gmberton/MegaLoc/{SOURCE_REVISION}/{filename}",
                timeout=30,
            )
            response.raise_for_status()
            with target.open("xb") as stream:
                stream.write(response.content)
    code = source / "megaloc_model.py"
    if file_hash(code) != SOURCE_SHA256:
        raise ValueError("MegaLoc source checksum differs from the reviewed pinned implementation")
    downloaded = {}
    for filename in ("LICENSE", "README.md", "model.safetensors"):
        downloaded[filename] = Path(hf_hub_download(
            repo_id="gberton/MegaLoc", filename=filename, revision=WEIGHTS_REVISION,
            cache_dir="data/huggingface/hub", token=False,
        ))
    weights = downloaded["model.safetensors"]
    return code, weights, {
        "source_repository": "https://github.com/gmberton/MegaLoc",
        "source_revision": SOURCE_REVISION, "source_sha256": file_hash(code),
        "weights_repository": "https://huggingface.co/gberton/MegaLoc",
        "weights_revision": WEIGHTS_REVISION, "weights_sha256": file_hash(weights),
        "source_license_sha256": file_hash(source / "LICENSE"),
        "weights_license_sha256": file_hash(downloaded["LICENSE"]),
        "preprocessing": PREPROCESSING,
    }


class MegaLocEmbedder:
    """Experiment-only loader; production CLIP startup and routes remain unchanged."""

    def __init__(self, root: Path, device: str) -> None:
        import torch
        import torchvision
        from safetensors.torch import load_file
        from torchvision import transforms as tfm

        torch.set_num_threads(4)
        code, weights, identity = load_assets(root)
        spec = importlib.util.spec_from_file_location("pinned_megaloc", code)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load pinned MegaLoc source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.MegaLoc().eval()
        state = load_file(weights, device="cpu")
        self.model.load_state_dict(state, strict=True)
        del state
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU is unavailable in this interpreter; use native macOS")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device == "mps":
            for layer in self.model.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    layer.register_forward_pre_hook(contiguous_conv_inputs)
        self.device = device
        self.transform = tfm.Compose([
            tfm.ToTensor(), tfm.Normalize(mean=PREPROCESSING["mean"], std=PREPROCESSING["std"]),
            tfm.Resize(size=[322, 322], antialias=True),
        ])
        self.identity = {
            **identity, "device": device, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "python": platform.python_version(),
            "platform": platform.platform(),
            "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            "conv_input_layout": "contiguous" if device == "mps" else "upstream",
        }
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.cache = root / "descriptors" / key
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model.to(device)

    def batch(self, paths: list[Path]) -> np.ndarray:
        """Apply the official single-image preprocessing without crops or test augmentation."""
        import torch
        from PIL import Image

        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(self.transform(image.convert("RGB")))
        with torch.inference_mode():
            result = self.model(torch.stack(tensors).to(self.device)).cpu().numpy()
        return checked_vectors(result, len(paths))

    def probe(self, paths: list[Path]) -> dict[str, Any]:
        """Check native acceleration against CPU outputs before the complete experiment."""
        selected = self.device
        self.device = "cpu"
        self.model.to("cpu")
        start = time.perf_counter()
        cpu = self.batch(paths)
        cpu_seconds = time.perf_counter() - start
        self.device = selected
        self.model.to(selected)
        start = time.perf_counter()
        accelerated = self.batch(paths)
        seconds = time.perf_counter() - start
        cosine = np.sum(cpu * accelerated, axis=1)
        difference = float(np.max(np.abs(cpu - accelerated)))
        if float(cosine.min()) < 0.999 or difference > 0.001:
            raise RuntimeError("Accelerated MegaLoc outputs differ materially from CPU")
        return {
            "image_count": len(paths), "device": selected,
            "cpu_seconds": cpu_seconds, "accelerated_first_batch_seconds": seconds,
            "minimum_cpu_device_cosine": float(cosine.min()), "maximum_abs_difference": difference,
        }

    def embed(self, paths: list[Path], hashes: list[str], batch_size: int) -> np.ndarray:
        """Cache each validated descriptor so interrupted jobs can resume safely."""
        result = np.zeros((len(paths), DIM), dtype=np.float32)
        pending = []
        for i, (path, digest) in enumerate(zip(paths, hashes, strict=True)):
            if file_hash(path) != digest:
                raise ValueError("Image bytes changed before embedding")
            cached = self.cache / f"{digest}.npy"
            if cached.exists():
                vector = np.load(cached, allow_pickle=False)
                result[i] = checked_vectors(vector[None, :], 1)[0]
            else:
                pending.append(i)
        print(f"Descriptors cached: {len(paths) - len(pending)}/{len(paths)}", flush=True)
        for start in range(0, len(pending), batch_size):
            indices = pending[start:start + batch_size]
            vectors = self.batch([paths[i] for i in indices])
            for i, vector in zip(indices, vectors, strict=True):
                result[i] = vector
                destination = self.cache / f"{hashes[i]}.npy"
                temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("xb") as stream:
                        np.save(stream, vector, allow_pickle=False)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            print(f"Embedded {len(paths) - len(pending) + start + len(indices)}/{len(paths)}",
                  flush=True)
        return checked_vectors(result, len(paths))


def quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize identical geographic criteria for each model."""
    return {
        **summarize(rows)["overall"],
        "median_top_1_error_m": statistics.median(q["top_1_error_m"] for q in rows),
        "median_best_of_5_error_m": statistics.median(q["best_candidate_error_m"] for q in rows),
        "pct_top_1_within_500m": 100 * sum(q["top_1_error_m"] <= 500 for q in rows) / len(rows),
        "pct_top_5_within_500m":
            100 * sum(q["best_candidate_error_m"] <= 500 for q in rows) / len(rows),
    }


def comparison_html(path: Path, report: dict[str, Any], clip_path: Path) -> None:
    """Write an offline-readable comparison linking both interactive maps."""
    clip, mega = report["comparison"]["clip"], report["comparison"]["megaloc"]
    measures = {
        "median_error_m": "Median averaged-pin error (m)",
        "pct_within_100m": "Averaged pin within 100m (%)",
        "pct_within_500m": "Averaged pin within 500m (%)",
        "pct_within_2000m": "Averaged pin within 2km (%)",
        "top_k_hit_pct": "Any top-five within 100m (%)",
        "pct_top_5_within_500m": "Any top-five within 500m (%)",
        "median_top_1_error_m": "Median top-ranked candidate error (m)",
        "median_best_of_5_error_m": "Median best-of-five error (m)",
    }
    table = "".join(
        f"<tr><td>{label}</td><td>{clip[key]:.2f}</td><td>{mega[key]:.2f}</td></tr>"
        for key, label in measures.items()
    )
    queries = "".join(
        f"<tr><td>{i}</td><td>{q['clip_error_m']:.0f}</td><td>{q['error_m']:.0f}</td>"
        f"<td>{q['error_m'] - q['clip_error_m']:+.0f}</td></tr>"
        for i, q in enumerate(report["queries"], start=1)
    )
    clip_link = html.escape(
        os.path.relpath(clip_path.with_suffix(".html"), path.parent), quote=True
    )
    paired = report["comparison"]["paired"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MegaLoc vs CLIP · Linköping</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;
color:#17253c;background:#f8f9fb}}table{{width:100%;border-collapse:collapse;background:white}}
td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{
text-align:left}}a{{color:#165ca3}}p{{line-height:1.6}}h1{{font-size:30px}}</style>
<h1>MegaLoc vs CLIP · Linköping</h1>
<p>Same {mega['count']} frozen photos, same {report['reference_count']} references.
No training, new imagery or query-driven tuning. {report['query_sequence_count']} query sequences.
These same-provider street results do not establish general photo-geolocation accuracy.</p>
<p><a href="map.html">Inspect MegaLoc predictions on the map</a> ·
<a href="{clip_link}">Inspect original CLIP predictions</a></p>
<table><thead><tr><th>Measure</th><th>CLIP</th><th>MegaLoc</th></tr></thead>
<tbody>{table}</tbody></table>
<p>MegaLoc pin error improved for {paired['improved']} queries, worsened for
{paired['worsened']}, and tied for {paired['tied']}. Query coordinates come from Mapillary GPS;
they were not manually verified. Only 5 of 30 queries have a reference within 100m.
Similarity scores remain uncalibrated and are not comparable probabilities.</p>
<p>Compute: MegaLoc on {html.escape(report['model_identity']['device'])};
original CLIP on Linux CPU. Quality comparisons use identical geographic criteria.
Runtime measurements use different execution paths/devices
and are not a controlled speed benchmark.</p>
<h2>Per-query averaged-pin error</h2><table><thead><tr><th>Query</th><th>CLIP (m)</th>
<th>MegaLoc (m)</th><th>Change (m)</th></tr></thead><tbody>{queries}</tbody></table>
<p>Original reference collection and CLIP report were preserved. Exact revisions, source/weight
checksums, preprocessing, image hashes, collection fingerprints and predictions are in results.json.
Photo previews and attribution are available in the linked maps.</p></html>"""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(page)


def run(args: argparse.Namespace) -> Path:
    """Run a separate MegaLoc retrieval experiment without changing the production API."""
    manifest, clip = validate_comparison(args.manifest, args.clip_report)
    settings = get_settings()
    if settings.testing:
        raise ValueError("TESTING is forbidden")
    original = VectorStore(settings)
    experiment = None
    try:
        before = index_fingerprint(original)
        if before != manifest["index_sha256"]:
            raise ValueError("Original CLIP index changed since the frozen test")
        references = reference_points(original)
        if len(references) != manifest["reference_count"]:
            raise ValueError("Reference count differs")
        reference_paths = [Path("data/raw") / f"{r['image_id']}.jpg" for r in references]
        reference_hashes = [file_hash(p) for p in reference_paths]
        queries = manifest["queries"]
        query_paths = [args.manifest.parent / "photos" / q["filename"] for q in queries]
        source_hash = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path("swespy").rglob("*.py")))
        ).hexdigest()
        if source_hash != clip["inference_source_sha256"]:
            raise ValueError("Production inference source differs from CLIP's frozen report")
        print(f"Frozen inputs verified: {len(references)} references, {len(queries)} queries",
              flush=True)
        embedder = MegaLocEmbedder(args.root, args.device)
        probe = embedder.probe([reference_paths[0], query_paths[0]])
        print("Device/CPU parity probe:", json.dumps(probe), flush=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
        folder = args.root / "experiments" / timestamp
        if args.probe_only:
            write_new(folder / "probe.json", {"identity": embedder.identity, "probe": probe})
            return folder / "probe.json"
        paths = reference_paths + query_paths
        hashes = reference_hashes + [q["sha256"] for q in queries]
        start = time.perf_counter()
        vectors = embedder.embed(paths, hashes, args.batch_size)
        embedding_seconds = time.perf_counter() - start
        if index_fingerprint(original) != before:
            raise RuntimeError("CLIP index changed during embedding; experiment not indexed")
        if any(file_hash(p) != digest for p, digest in zip(paths, hashes, strict=True)):
            raise ValueError("Input image bytes changed during embedding")
        name = "swespy_megaloc_" + timestamp.lower()
        if name == original.collection_name:
            raise ValueError("Experiment collection must differ from production")
        experiment = VectorStore(settings.model_copy(update={
            "qdrant_collection_name": name, "embedding_dim": DIM,
            "vector_upsert_batch_size": 32,
        }))
        if experiment.client.collection_exists(name):
            raise ValueError("Experiment collection already exists; refusing to overwrite")
        write_new(folder / "experiment.json", {
            "collection": name, "original_collection": original.collection_name,
            "original_index_sha256": before, "model_identity": embedder.identity,
            "manifest_sha256": file_hash(args.manifest),
            "clip_report_sha256": file_hash(args.clip_report),
            "references": [{**r, "sha256": h}
                           for r, h in zip(references, reference_hashes, strict=True)],
        })
        experiment.create_collection()
        experiment.upsert_embeddings(vectors[:len(references)], references)
        if experiment.client.count(name, exact=True).count != len(references):
            raise RuntimeError("Experiment index is incomplete")
        replay = experiment.query(vectors[0], 1)
        if (not replay or replay[0]["image_id"] != references[0]["image_id"]
                or replay[0]["similarity_score"] < 0.99):
            raise RuntimeError("MegaLoc reference replay failed; refusing to report query metrics")
        experiment_before = index_fingerprint(experiment)
        measured_rows = []
        rows = []
        for i, (q, vector, old) in enumerate(zip(
            queries, vectors[len(references):], clip["queries"], strict=True
        ), start=1):
            start = time.perf_counter()
            matches = experiment.query(vector, 5)
            aggregate = experiment.aggregate_prediction(matches)
            elapsed = (time.perf_counter() - start) * 1000
            prediction = {
                "predicted_lat": aggregate["lat"], "predicted_lng": aggregate["lng"],
                "confidence": max(0.0, min(1.0, aggregate["confidence"])),
                "top_matches": matches, "processing_time_ms": elapsed,
            }
            photo = Photo(q["filename"], query_paths[i - 1], q["lat"], q["lng"],
                          "street", q["sha256"])
            measured = observation(photo, prediction, 100)
            distances = [distance_m(q["lat"], q["lng"], m["lat"], m["lng"]) for m in matches]
            if len(distances) != 5:
                raise RuntimeError("Experiment did not return five candidates")
            measured_rows.append(measured)
            rows.append({
                **q, **measured, "prediction": prediction,
                "best_candidate_error_m": min(distances), "top_1_error_m": distances[0],
                "nearest_reference_m": old["nearest_reference_m"], "preview": old["preview"],
                "clip_error_m": old["error_m"],
            })
            print(f"Query {i}/{len(queries)}: CLIP {old['error_m']:.0f}m, "
                  f"MegaLoc {measured['error_m']:.0f}m", flush=True)
        if index_fingerprint(original) != before:
            raise RuntimeError("Original index changed during comparison")
        if index_fingerprint(experiment) != experiment_before:
            raise RuntimeError("Experiment index changed during comparison")
        report = {
            **{k: v for k, v in manifest.items() if k != "queries"},
            "kind": "megaloc_vs_frozen_clip_street_comparison",
            "model": "gberton/MegaLoc", "model_label": "MegaLoc · Linköping street check",
            "model_identity": embedder.identity, "device_probe": probe,
            "embedding_seconds_including_cache_reads": embedding_seconds,
            "descriptor_count": len(paths), "collection": name,
            "experiment_index_sha256": experiment_before,
            "original_index_sha256": before, "original_index_unchanged": True,
            "manifest_sha256": file_hash(args.manifest),
            "clip_report_sha256": file_hash(args.clip_report),
            "evaluator_sha256": file_hash(Path(__file__)),
            "lock_sha256": file_hash(Path("uv.lock")),
            "production_inference_source_sha256": source_hash,
            "top_k": 5, "hit_radius_m": 100, "coverage": clip["coverage"],
            "metrics": summarize(measured_rows), "queries": rows,
            "confidence_status": "uncalibrated mean similarity; not comparable across models",
            "timing_scope": "Prediction times exclude precomputed embedding; different devices "
                            "and execution paths prevent a controlled latency comparison",
            "comparison": {
                "clip": quality(clip["queries"]), "megaloc": quality(rows),
                "paired": {
                    "improved": sum(q["error_m"] < q["clip_error_m"] for q in rows),
                    "worsened": sum(q["error_m"] > q["clip_error_m"] for q in rows),
                    "tied": sum(q["error_m"] == q["clip_error_m"] for q in rows),
                },
            },
        }
        output = folder / "results.json"
        write_new(output, report)
        render_map(folder / "map.html", report)
        comparison_html(folder / "comparison.html", report, args.clip_report)
        print_table(report["metrics"])
        print("Comparison:", json.dumps(report["comparison"], indent=2), flush=True)
        print("Saved:", output, flush=True)
        return output
    finally:
        original.client.close()
        if experiment is not None:
            experiment.client.close()


def main() -> None:
    """Run the pretrained comparison with explicit immutable inputs and small batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RUN / "manifest.json")
    parser.add_argument("--clip-report", type=Path,
                        default=DEFAULT_RUN / "report-20261106T095433.665610Z.json")
    parser.add_argument("--root", type=Path, default=Path("data/megaloc"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error("Batch size must be 1–32")
    print("Saved artifact:", run(args))


if __name__ == "__main__":
    main()
