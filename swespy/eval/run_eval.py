"""Evaluate the real API in process: python -m eval.run_eval (no dummy mode)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dataset import DatasetError, Photo, load_dataset
from eval.metrics import observation, print_table, summarize

ROOT = Path(__file__).resolve().parents[1]


def write_new(path: Path, data: dict[str, Any]) -> None:
    """Write a new JSON artifact without replacing an existing baseline or snapshot."""
    serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def index_fingerprint(store: Any) -> str:
    """Hash all reference point IDs, payloads and vectors without retaining their contents."""
    digest = hashlib.sha256()
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            record = {"id": point.id, "payload": point.payload, "vector": point.vector}
            digest.update(json.dumps(record, sort_keys=True, allow_nan=False).encode())
            digest.update(b"\n")
        if offset is None:
            break
    return digest.hexdigest()


async def infer(photos: list[Photo]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run actual startup and POST /predict against CLIP/Qdrant, rejecting test mode.

    Args:
        photos: Validated user-owned queries.

    Returns:
        Predictions and environment/model/index identity.

    Raises:
        RuntimeError: Real inference inputs/services are unavailable or change during the run.
    """
    from httpx import ASGITransport, AsyncClient
    from loguru import logger

    from swespy.config import get_settings
    from swespy.vector_store.qdrant_client import VectorStore

    get_settings.cache_clear()
    settings = get_settings()
    if settings.testing:
        raise RuntimeError("TESTING=true is forbidden for real evaluation and snapshots.")
    # Check before API startup: never create a missing/empty collection as a baseline.
    store = VectorStore(settings)
    try:
        info = store.client.get_collection(store.collection_name)
        if not info.points_count:
            raise RuntimeError("The reference collection is empty; supply the existing real index.")
        before = index_fingerprint(store)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Cannot inspect the reference collection. Start Qdrant and supply its existing "
            "collection; evaluation will not download data or build an index."
        ) from exc
    finally:
        store.client.close()

    from swespy.api.main import create_app

    app = create_app()
    # The unchanged route logs image hashes/pins; suppress its logs for private evaluation.
    logger.disable("swespy.api.routes.predict")
    predictions: list[dict[str, Any]] = []
    try:
        async with app.router.lifespan_context(app):
            revision = getattr(app.state.embedder.model.config, "_commit_hash", None)
            if not revision:
                raise RuntimeError(
                    "Cannot identify the loaded CLIP revision; baseline is not reproducible."
                )
            model_device = str(app.state.embedder.device)
            processor_digest = hashlib.sha256(
                json.dumps(
                    app.state.embedder.processor.image_processor.to_dict(), sort_keys=True
                ).encode()
            ).hexdigest()
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://eval"
            ) as client:
                for i, photo in enumerate(photos, start=1):
                    # Rate limiting is deliberately unchanged; wait between real requests.
                    if i > 1:
                        await asyncio.sleep(60 / settings.api_rate_limit_per_minute + 0.05)
                    content_type = (
                        "image/png"
                        if photo.path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
                        else "image/jpeg"
                    )
                    response = await client.post(
                        "/predict", files={"file": ("query", photo.path.read_bytes(), content_type)}
                    )
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Real prediction {i} failed with HTTP {response.status_code}."
                        )
                    predictions.append(response.json())
            after = await asyncio.to_thread(index_fingerprint, app.state.vector_store)
            if before != after:
                raise RuntimeError(
                    "Reference index changed during evaluation; no result was saved."
                )
    finally:
        if hasattr(app.state, "vector_store"):
            app.state.vector_store.client.close()
        logger.enable("swespy.api.routes.predict")
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    revision_command = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    dataset_digest = hashlib.sha256(
        json.dumps(
            [[p.filename, p.lat, p.lng, p.viewpoint_type, p.sha256] for p in photos], sort_keys=True
        ).encode()
    ).hexdigest()
    return predictions, {
        "git_revision": revision_command.stdout.strip(),
        "inference_source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted((ROOT / "swespy").rglob("*.py")))
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "model": settings.clip_model_name,
        "model_revision": revision,
        "processor_sha256": processor_digest,
        "device": model_device,
        "collection": settings.qdrant_collection_name,
        "index_sha256": before,
        "dataset_sha256": dataset_digest,
        "top_k": settings.top_k_results,
        "s2_level": settings.s2_cell_level,
        "embedding_dim": settings.embedding_dim,
    }


def compare_snapshot(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Require identical fixtures/index/model and predictions within 1m / 1e-5 score."""
    from eval.metrics import distance_m

    for key in (
        "dataset_sha256",
        "index_sha256",
        "model",
        "model_revision",
        "processor_sha256",
        "top_k",
        "s2_level",
        "embedding_dim",
    ):
        if expected["identity"][key] != actual["identity"][key]:
            raise ValueError(f"Snapshot identity changed: {key}; do not replace the snapshot.")
    old, new = expected["predictions"], actual["predictions"]
    if len(old) != len(new):
        raise ValueError("Snapshot prediction count changed")
    for i, (a, b) in enumerate(zip(old, new, strict=True), start=1):
        if (
            distance_m(
                a["predicted_lat"], a["predicted_lng"], b["predicted_lat"], b["predicted_lng"]
            )
            > 1
        ):
            raise ValueError(f"Snapshot pin changed beyond 1m at query {i}")
        if abs(a["confidence"] - b["confidence"]) > 1e-5:
            raise ValueError(f"Snapshot confidence changed at query {i}")
        if len(a["top_matches"]) != len(b["top_matches"]):
            raise ValueError(f"Snapshot alternatives count changed at query {i}")
        for x, y in zip(a["top_matches"], b["top_matches"], strict=True):
            if (
                x["s2_cell"] != y["s2_cell"]
                or abs(x["similarity_score"] - y["similarity_score"]) > 1e-5
                or distance_m(x["lat"], x["lng"], y["lat"], y["lng"]) > 1
            ):
                raise ValueError(f"Snapshot ranked alternatives changed at query {i}")


def compare_baseline(baseline: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Compare every viewpoint with a frozen baseline using declared refactor tolerances."""
    for key in ("dataset_sha256", "index_sha256", "model_revision", "processor_sha256", "top_k"):
        if baseline["identity"][key] != report["identity"][key]:
            raise ValueError(f"Baseline comparison inputs changed: {key}")
    if baseline["hit_radius_m"] != report["hit_radius_m"]:
        raise ValueError("Baseline geographic hit radius changed")
    deltas: dict[str, Any] = {}
    before = {"overall": baseline["metrics"]["overall"], **baseline["metrics"]["by_viewpoint"]}
    after = {"overall": report["metrics"]["overall"], **report["metrics"]["by_viewpoint"]}
    regressions: list[str] = []
    for viewpoint, old in before.items():
        new = after[viewpoint]
        if old["count"] != new["count"]:
            raise ValueError(f"Baseline group count changed: {viewpoint}")
        if not old["count"]:
            deltas[viewpoint] = None
            continue
        changes = {key: new[key] - value for key, value in old.items() if key != "count"}
        deltas[viewpoint] = changes
        if changes["median_error_m"] > 1 or any(
            value < 0 for key, value in changes.items() if key != "median_error_m"
        ):
            regressions.append(viewpoint)
    return {
        "deltas": deltas,
        "regressions": regressions,
        "tolerance": "median +1m, zero percentage-point regression in every group",
    }


def main() -> int:
    """Validate inputs, run real inference and save timestamped metrics with provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "eval/labels.csv")
    parser.add_argument("--photos", type=Path, default=ROOT / "eval/photos")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--hit-radius-m", type=float, default=100.0)
    parser.add_argument(
        "--baseline", action="store_true", help="Create baseline.json once; never overwrite"
    )
    parser.add_argument(
        "--capture-snapshot", action="store_true", help="Create private predictions.json once"
    )
    parser.add_argument("--check-snapshot", action="store_true")
    parser.add_argument("--compare-baseline", type=Path)
    args = parser.parse_args()
    try:
        if args.hit_radius_m <= 0 or not math.isfinite(args.hit_radius_m):
            raise ValueError("Hit radius must be finite and positive")
        if args.capture_snapshot and args.check_snapshot:
            raise ValueError("Capture and check snapshot cannot be combined")
        photos = load_dataset(args.labels, args.photos, ROOT / "data/raw")
        if args.validate_only:
            print(f"Validated {len(photos)} private photos; no inference performed.")
            return 0
        snapshot_path = ROOT / "eval/snapshots/predictions.json"
        baseline_path = ROOT / "eval/results/baseline.json"
        if args.capture_snapshot and len(photos) < 10:
            raise ValueError("Snapshot capture requires at least 10 fixed private photos")
        if args.baseline and {p.viewpoint_type for p in photos} != {
            "street",
            "window",
            "elevated",
            "indoor_view",
        }:
            raise ValueError("Baseline capture requires photos from all four viewpoint types")
        if args.capture_snapshot and snapshot_path.exists():
            raise ValueError("Snapshot already exists; refusing to replace it")
        if args.baseline and baseline_path.exists():
            raise ValueError("Baseline already exists; refusing to replace it")
        expected = json.loads(snapshot_path.read_text()) if args.check_snapshot else None
        # Respect the project's relative .env convention, independent of caller directory.
        os.chdir(ROOT)
        predictions, identity = asyncio.run(infer(photos))
        snapshot = {"identity": identity, "predictions": predictions}
        if expected:
            compare_snapshot(expected, snapshot)
            print("Real-pipeline snapshot matched (1m coordinates, 1e-5 scores).")
        rows = [
            observation(p, prediction, args.hit_radius_m)
            for p, prediction in zip(photos, predictions, strict=True)
        ]
        now = datetime.now(UTC)
        report = {
            "schema_version": 1,
            "created_at": now.isoformat(),
            "identity": identity,
            "top_k_hit_definition": (
                "any returned reference coordinate within hit_radius_m of camera label"
            ),
            "hit_radius_m": args.hit_radius_m,
            "metrics": summarize(rows),
            "confidence_status": "uncalibrated mean similarity; calibration deferred to Part 4",
            "per_photo": rows,
        }
        comparison_path = args.compare_baseline
        if not comparison_path and baseline_path.exists() and not args.baseline:
            comparison_path = baseline_path
        if comparison_path:
            report["baseline_comparison"] = compare_baseline(
                json.loads(comparison_path.read_text()), report
            )
        # Baseline contains aggregate metrics and provenance, not private photo-level predictions.
        if args.baseline:
            write_new(baseline_path, {k: v for k, v in report.items() if k != "per_photo"})
        if args.capture_snapshot:
            write_new(snapshot_path, snapshot)
        output = ROOT / "eval/results" / now.strftime("%Y%m%dT%H%M%S.%fZ.json")
        write_new(output, report)
        print_table(report["metrics"])
        print(f"Saved evaluation to {output}")
        if report.get("baseline_comparison", {}).get("regressions"):
            print("Baseline regression detected; inspect the saved deltas. Baseline was preserved.")
            return 1
        return 0
    except (DatasetError, ValueError, OSError, RuntimeError) as exc:
        print(f"Evaluation unavailable: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
