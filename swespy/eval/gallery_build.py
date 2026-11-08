"""Pipelined, staged, resumable gallery v2 build: download+filter and encode concurrently.

    python -m eval.gallery_build --stage recent      # Mapillary 2018+, all panoramas,
                                                     # Commons, KartaView, approved walks
    python -m eval.gallery_build --stage historic    # adds pre-2018 Mapillary images

Both stages use the frozen ``plan.json`` (thinning and plan-time leakage guards
already applied). The plan is processed in fixed chunks, in plan order:

- producer (asyncio): download -> panorama crops -> quality rules -> near-duplicate
  check against everything kept so far -> pixel-hash leakage guards (dev visible,
  test sealed) -> chunk checkpoint ``chunks/<stage>-<n>.json``;
- consumer (GPU thread): MegaLoc encodes each checkpointed chunk while the producer
  continues; descriptors go to the hash-keyed cache and a ``.encoded`` marker is
  written per chunk.

Restarting skips finished chunks and re-queues filtered-but-unencoded ones. When all
chunks of a stage are encoded, ``gallery-index-<stage>.json`` is written; the
historic index contains recent + historic references.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import queue
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval import densify, leakage, panorama, provenance, quality
from eval.common import signature
from swespy.config import get_settings

STAGES = ("recent", "historic")
CHUNK = 1000
# Defaults from eval/encode_benchmark.py on the M1 (docs/experiments/05_build_speed.md):
# fp32 batch 16 = 8.9 img/s end to end with 2.1 GB GPU memory; batch 32 is 0.6% faster
# but needs 3.3 GB on a machine that was already swapping; fp16 (autocast) is slower on
# MPS and changed 2/66 dev top-1 and 8/66 top-5 sets, so it is not used.
DEFAULT_BATCH = 16
DEFAULT_DTYPE = "float32"


def stage_of(ref: dict[str, Any]) -> str:
    if ref["source"] == "mapillary" and not ref.get("is_pano") and ref["capture_year"] < 2018:
        return "historic"
    return "recent"


def chunks_for(root: Path, stage: str) -> list[tuple[str, list[dict[str, Any]]]]:
    plan = densify.read_json(root / "plan.json")["references"]
    refs = [r for r in plan if stage_of(r) == stage]
    return [(f"{stage}-{i // CHUNK:04d}", refs[i : i + CHUNK]) for i in range(0, len(refs), CHUNK)]


def state_path(root: Path, chunk_id: str) -> Path:
    return root / "chunks" / f"{chunk_id}.json"


def encoded_marker(root: Path, chunk_id: str) -> Path:
    return root / "chunks" / f"{chunk_id}.encoded"


# ---------------------------------------------------------------- producer side


def expand_panoramas(root: Path, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace each downloaded panorama by its perspective crops; set paths."""
    from PIL import Image

    out = []
    for ref in refs:
        src = densify.image_path(root, ref)
        if ref["source"] != "self_collected":
            ref = {**ref, "path": str(src.relative_to(root))}
        if not ref.get("is_pano"):
            out.append(ref)
            continue
        if not src.exists():
            continue
        folder = root / "images" / "mapillary_pano_crops"
        folder.mkdir(parents=True, exist_ok=True)
        targets = [folder / f"{ref['source_id']}_{k}.jpg" for k in range(panorama.CROP_COUNT)]
        if not all(t.exists() for t in targets):
            with Image.open(src) as pano:
                for target, (crop, _, _) in zip(
                    targets, panorama.split(pano, ref["heading"]), strict=True
                ):
                    crop.save(target, quality=densify.JPEG_QUALITY)
        for k, (target, (offset, heading)) in enumerate(
            zip(targets, panorama.crop_headings(ref["heading"]), strict=True)
        ):
            out.append(
                {
                    **ref,
                    "ref_id": f"{ref['ref_id']}-crop{k}",
                    "is_pano": False,
                    "pano_crop": {
                        "index": k,
                        "yaw_offset_deg": offset,
                        "fov_deg": panorama.FOV_DEG,
                    },
                    "heading": heading,
                    "heading_source": "derived_from_panorama",
                    "path": str(target.relative_to(root)),
                }
            )
    return out


class Deduper:
    """Near-duplicate check against every reference kept so far (all chunks)."""

    def __init__(self) -> None:
        self.cells: dict[tuple[int, int], list[dict[str, Any]]] = {}

    @staticmethod
    def key(r: dict[str, Any]) -> tuple[int, int]:
        size = quality.DUPLICATE_RADIUS_M
        return (
            int(r["lat"] * 111_320 // size),
            int(r["lng"] * 111_320 * np.cos(np.radians(r["lat"])) // size),
        )

    def is_duplicate(self, r: dict[str, Any]) -> bool:
        cx, cy = self.key(r)
        near = [
            k
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for k in self.cells.get((cx + dx, cy + dy), [])
        ]
        return quality.is_duplicate(r, near)

    def add(self, r: dict[str, Any]) -> None:
        self.cells.setdefault(self.key(r), []).append(
            {k: r[k] for k in ("sha256", "phash", "dhash", "source", "lat", "lng")}
        )


def assess(path: Path) -> dict[str, Any] | None:
    """Hashes + quality measures for one stored image (runs in a worker process)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            return {**signature(path), **quality.measures(im)}
    except OSError:
        return None


def filter_chunk(
    root: Path, refs: list[dict[str, Any]], dedup: Deduper, pool: Any = None
) -> dict[str, Any]:
    dropped: Counter[str] = Counter()
    kept = []
    ordered = sorted(refs, key=lambda r: (-r["captured_at"], r["ref_id"]))
    present = [r for r in ordered if (root / r["path"]).exists()]
    dropped["not_downloaded"] = len(ordered) - len(present)
    paths = [root / r["path"] for r in present]
    measured = list(pool.map(assess, paths, chunksize=16) if pool else map(assess, paths))
    for r, q in zip(present, measured, strict=True):  # Sequential: dedup order is fixed.
        if q is None:
            dropped["undecodable"] += 1
            continue
        rule = quality.image_rule(q)
        row = {**r, **q, "image_id": r["ref_id"]}
        if rule is None and dedup.is_duplicate(row):
            rule = "near_duplicate"
        if rule:
            dropped[rule] += 1
            continue
        dedup.add(row)
        kept.append(row)
    dropped = Counter({k: v for k, v in dropped.items() if v})
    kept, dev = leakage.guard_dev(kept)
    kept, sealed = leakage.guard_sealed(kept)
    bad = [r for r in kept if provenance.problems(r)]
    if bad:
        raise ValueError(f"{len(bad)} references violate the provenance contract")
    return {
        "input": len(refs),
        "kept": kept,
        "dropped_by_rule": dict(dropped),
        "leakage_dev": {k: v for k, v in dev.items() if k != "exclusions"},
        "leakage_dev_exclusions": dev["exclusions"],
        "leakage_test_sealed_counts_only": sealed,
    }


async def produce(
    root: Path,
    stage: str,
    work: queue.Queue,
    stop: threading.Event,
    dedup: Deduper,
    pool: Any = None,
) -> None:
    for chunk_id, refs in chunks_for(root, stage):
        if stop.is_set():
            return
        if state_path(root, chunk_id).exists():
            continue
        if densify.free_bytes(root) < densify.MIN_FREE_BYTES:
            densify.log("build", state="stopped_low_disk", chunk=chunk_id)
            break
        started = time.monotonic()
        failures = await densify.download_refs(root, refs)
        expanded = expand_panoramas(root, refs)
        state = filter_chunk(root, expanded, dedup, pool)
        state.update(
            {
                "chunk": chunk_id,
                "stage": stage,
                "download_failures": failures,
                "seconds": time.monotonic() - started,
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        densify.write_json(state_path(root, chunk_id), state)
        densify.log(
            "build",
            chunk=chunk_id,
            kept=len(state["kept"]),
            dropped=state["dropped_by_rule"],
            seconds=round(state["seconds"]),
            queue=work.qsize(),
        )
        if not put(work, chunk_id, stop):
            return
    put(work, None, stop)


# ---------------------------------------------------------------- consumer side


def put(work: queue.Queue, item: str | None, stop: threading.Event) -> bool:
    """Blocking put that gives up if the encoder has stopped (avoids a deadlock)."""
    while not stop.is_set():
        try:
            work.put(item, timeout=5)
            return True
        except queue.Full:
            continue
    return False


def encode_worker(
    root: Path,
    device: str,
    dtype: str,
    batch: int,
    work: queue.Queue,
    stop: threading.Event,
    errors: list[BaseException],
    make_embedder: Any = None,
) -> None:
    from PIL import Image

    from swespy.model.megaloc import MegaLocEmbedder

    try:
        factory = make_embedder or (
            lambda: MegaLocEmbedder(Path(get_settings().megaloc_root), device, dtype)
        )
        embedder = factory()
        densify.write_json(root / "encoder-identity.json", embedder.identity)

        def load(r: dict[str, Any]) -> Any:
            with Image.open(root / r["path"]) as im:
                return embedder.transform(im.convert("RGB"))

        with ThreadPoolExecutor(max_workers=4) as pool:
            while True:
                chunk_id = work.get()
                if chunk_id is None or stop.is_set():
                    return
                refs = densify.read_json(state_path(root, chunk_id))["kept"]
                todo = [r for r in refs if not (embedder.cache / f"{r['sha256']}.npy").exists()]
                started = time.monotonic()
                window = batch * 4  # Bounded prefetch: decode ahead of the GPU, not all at once.
                upcoming = pool.map(load, todo[:window])
                for start in range(0, len(todo), window):
                    current = list(upcoming)
                    nxt = todo[start + window : start + 2 * window]
                    upcoming = pool.map(load, nxt) if nxt else iter(())
                    for i in range(0, len(current), batch):
                        vectors = embedder.batch_tensors(current[i : i + batch])
                        for r, v in zip(todo[start + i : start + i + batch], vectors, strict=True):
                            target = embedder.cache / f"{r['sha256']}.npy"
                            tmp = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
                            with tmp.open("xb") as stream:
                                np.save(stream, v, allow_pickle=False)
                            tmp.replace(target)
                encoded_marker(root, chunk_id).write_text(datetime.now(UTC).isoformat())
                seconds = time.monotonic() - started
                densify.log(
                    "encode",
                    chunk=chunk_id,
                    encoded=len(todo),
                    img_per_s=round(len(todo) / seconds, 1) if todo else None,
                )
    except BaseException as exc:  # noqa: BLE001 - surfaced by the main thread.
        errors.append(exc)
        stop.set()


# ---------------------------------------------------------------- index


def write_index(root: Path, stage: str) -> Path:
    stages = STAGES[: STAGES.index(stage) + 1]
    chunk_ids = [cid for s in stages for cid, _ in chunks_for(root, s)]
    missing = [c for c in chunk_ids if not encoded_marker(root, c).exists()]
    if missing:
        raise ValueError(f"{len(missing)} chunks not encoded yet (e.g. {missing[0]})")
    states = [densify.read_json(state_path(root, c)) for c in chunk_ids]
    refs = [r for s in states for r in s["kept"]]
    identity = densify.read_json(root / "encoder-identity.json")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(get_settings().megaloc_root) / "descriptors" / key
    absent = [r for r in refs if not (cache / f"{r['sha256']}.npy").exists()]
    if absent:
        raise ValueError(f"{len(absent)} references lack descriptors")
    totals: Counter[str] = Counter()
    sealed: Counter[str] = Counter()
    for s in states:
        totals.update(s["dropped_by_rule"])
        sealed.update(s["leakage_test_sealed_counts_only"])
    out = root / f"gallery-index-{stage}.json"
    densify.write_json(
        out,
        {
            "format": "descriptor_cache_v1",
            "dimension": 8448,
            "stage": stage,
            "stages_included": list(stages),
            "descriptor_dir": str(cache),
            "references_in_index_order": refs,
            "index_sha256": hashlib.sha256(
                "\n".join(r["sha256"] for r in refs).encode()
            ).hexdigest(),
            "model_identity": identity,
            "created_at": datetime.now(UTC).isoformat(),
            "build_parameters": densify.build_parameters(),
            "plan_report_sha256": hashlib.sha256(
                (root / "plan-report.json").read_bytes()
            ).hexdigest(),
            "filter_totals": {
                "input": sum(s["input"] for s in states),
                "kept": len(refs),
                "dropped_by_rule": dict(totals),
            },
            "post_download_leakage_test_sealed_counts_only": dict(sealed),
        },
    )
    densify.log("index", build_stage=stage, references=len(refs))
    return out


def build(
    root: Path, stage: str, device: str, dtype: str, batch: int, make_embedder: Any = None
) -> Path:
    if stage == "historic":
        recent = [c for c, _ in chunks_for(root, "recent")]
        if any(not encoded_marker(root, c).exists() for c in recent):
            raise ValueError("Finish --stage recent before --stage historic")
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    densify._LEGACY = densify.legacy_images()
    # Rebuild duplicate state from every chunk already kept (all stages, plan order).
    dedup = Deduper()
    for s in STAGES:
        for cid, _ in chunks_for(root, s):
            if state_path(root, cid).exists():
                for r in densify.read_json(state_path(root, cid))["kept"]:
                    dedup.add(r)
    work: queue.Queue = queue.Queue(maxsize=3)  # Bounded: limits disk ahead of encoding.
    stop, errors = threading.Event(), []
    for cid, _ in chunks_for(root, stage):  # Resume: filtered but not encoded first.
        if state_path(root, cid).exists() and not encoded_marker(root, cid).exists():
            work.put(cid)
    worker = threading.Thread(
        target=encode_worker, args=(root, device, dtype, batch, work, stop, errors, make_embedder)
    )
    worker.start()
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            asyncio.run(produce(root, stage, work, stop, dedup, pool))
    finally:
        if stop.is_set() or errors:
            try:
                work.put_nowait(None)
            except queue.Full:
                pass
        worker.join()
    if errors:
        raise errors[0]
    return write_index(root, stage)


def verify(root: Path, stage: str, device: str, sample: int, seed: int = 0) -> dict[str, Any]:
    """Re-encode a random sample locally and compare with the index's descriptors.

    Use after copying a gallery encoded on another machine (e.g. CUDA in the cloud):
    the descriptors must match the local encoder closely enough to share one index.
    """
    import random

    from PIL import Image

    from swespy.model.megaloc import MegaLocEmbedder

    saved = densify.read_json(root / f"gallery-index-{stage}.json")
    refs = random.Random(seed).sample(
        saved["references_in_index_order"], min(sample, len(saved["references_in_index_order"]))
    )
    folder = Path(saved["descriptor_dir"])
    stored = np.stack([np.load(folder / f"{r['sha256']}.npy") for r in refs])
    embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), device)
    local = []
    for i in range(0, len(refs), 16):
        images = []
        for r in refs[i : i + 16]:
            with Image.open(root / r["path"]) as im:
                images.append(im.convert("RGB"))
        local.append(embedder.batch_images(images))
    cos = np.sum(stored * np.concatenate(local), axis=1)
    result = {
        "sample": len(refs),
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "ok": bool(cos.min() >= 0.999),
        "index_device": saved["model_identity"].get("device"),
        "local_device": embedder.identity["device"],
    }
    densify.log("verify", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument(
        "--verify",
        type=int,
        metavar="N",
        help="Only re-encode N random references and compare descriptors",
    )
    parser.add_argument("--root", type=Path, default=densify.ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16"], default=DEFAULT_DTYPE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.root, args.stage, args.device, args.verify)))
        return
    print(build(args.root, args.stage, args.device, args.dtype, args.batch_size))


if __name__ == "__main__":
    main()
