"""Benchmark MegaLoc encoding throughput and check fp16 against fp32 retrieval.

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-benchmark/bin/python -m eval.encode_benchmark \
        --out data/gallery_v2/encode-benchmark.json

1. Throughput on 500 gallery images stored as gallery v2 stores them (640 px JPEG):
   model-only (pre-decoded tensors) and end-to-end (JPEG decode + preprocessing +
   model), per batch size and dtype.
2. fp16 safety: encode the polygon gallery (v1, 2,489 refs) and the 66 dev photos in
   fp16, then compare with fp32 (cosine per descriptor, top-1/top-5 retrieval).
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval.quality_review import stored_version
from swespy.config import get_settings, in_area
from swespy.model.megaloc import MegaLocEmbedder, cache_pos_encoding

GALLERY = Path("data/linkoping_dense/city_v1")
DEV = Path("data/public_eval/recent_linkoping_streets_v1")
BATCHES = (1, 8, 16, 32, 64)


def decode(data: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")


def run(embedder: MegaLocEmbedder, tensors: list[Any], batch: int, dtype: Any) -> np.ndarray:
    import torch

    out = []
    half = dtype == torch.float16  # fp16 via autocast; weights stay fp32 (see megaloc.py).
    with torch.inference_mode(), torch.autocast(embedder.device, torch.float16, enabled=half):
        for i in range(0, len(tensors), batch):
            x = torch.stack(tensors[i : i + batch]).to(embedder.device)
            y = embedder.model(x).float()
            y = torch.nn.functional.normalize(y, dim=1)  # fp16 output re-normalised in fp32.
            out.append(y.cpu().numpy())
    if embedder.device == "mps":
        torch.mps.synchronize()
    return np.concatenate(out)


def set_pos_cache(embedder: MegaLocEmbedder, enabled: bool) -> None:
    for module in embedder.model.modules():
        if "interpolate_pos_encoding" in vars(module):
            del module.interpolate_pos_encoding  # Back to the original class method.
    if enabled:
        cache_pos_encoding(embedder.model)


def pos_cache_identity(embedder: MegaLocEmbedder, blobs: list[bytes]) -> dict[str, Any]:
    """Cached vs uncached positional encoding must give bit-identical descriptors."""
    import torch

    tensors = [embedder.transform(decode(b)) for b in blobs[:32]]
    set_pos_cache(embedder, False)
    plain = run(embedder, tensors, 16, torch.float32)
    set_pos_cache(embedder, True)
    cached = run(embedder, tensors, 16, torch.float32)
    return {"images": len(tensors), "max_abs_diff": float(np.abs(plain - cached).max())}


def timed(
    embedder: MegaLocEmbedder,
    blobs: list[bytes],
    tensors: list[Any],
    batch: int,
    dtype: Any,
    label: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    run(embedder, tensors, batch, dtype)
    model_s = time.perf_counter() - start
    start = time.perf_counter()
    fresh = [embedder.transform(decode(b)) for b in blobs]
    run(embedder, fresh, batch, dtype)
    e2e_s = time.perf_counter() - start
    row = {
        "variant": label,
        "dtype": str(dtype).removeprefix("torch."),
        "batch": batch,
        "model_only_img_per_s": len(blobs) / model_s,
        "end_to_end_img_per_s": len(blobs) / e2e_s,
    }
    print(json.dumps(row), flush=True)
    return row


def throughput(embedder: MegaLocEmbedder, blobs: list[bytes]) -> list[dict[str, Any]]:
    import torch

    tensors = [embedder.transform(decode(b)) for b in blobs]
    rows = []
    set_pos_cache(embedder, False)  # "Before": original model, CPU fallback every batch.
    run(embedder, tensors[:32], 16, torch.float32)
    rows.append(timed(embedder, blobs, tensors, 16, torch.float32, "uncached_pos_encoding"))
    set_pos_cache(embedder, True)
    for dtype in (torch.float32, torch.float16):
        run(embedder, tensors[:64], 32, dtype)  # Warm-up (kernel compilation).
        for batch in BATCHES:
            rows.append(timed(embedder, blobs, tensors, batch, dtype, "cached_pos_encoding"))
    return rows


def fp16_check(embedder: MegaLocEmbedder, batch: int) -> dict[str, Any]:
    import torch

    refs = [
        r
        for r in json.loads((GALLERY / "manifest.json").read_text())["references"]
        if in_area(r["lng"], r["lat"])
    ]
    queries = json.loads((DEV / "manifest.json").read_text())["queries"]

    def encode(paths: list[Path], dtype: Any) -> np.ndarray:
        out = []
        for i in range(0, len(paths), 256):
            tensors = [embedder.transform(decode(p.read_bytes())) for p in paths[i : i + 256]]
            out.append(run(embedder, tensors, batch, dtype))
        return np.concatenate(out)

    gallery_paths = [GALLERY / r["path"] for r in refs]
    query_paths = [DEV / q["path"] for q in queries]
    g32, q32 = encode(gallery_paths, torch.float32), encode(query_paths, torch.float32)
    g16, q16 = encode(gallery_paths, torch.float16), encode(query_paths, torch.float16)

    def top(q: np.ndarray, g: np.ndarray) -> np.ndarray:
        return np.argsort(-(q @ g.T), axis=1)[:, :5]

    t32 = top(q32, g32)
    t16 = top(q16, g16)  # Both gallery and queries in fp16 (worst case).
    t_mixed = top(q32, g16)  # fp16 gallery, fp32 query (as served by a CPU API).
    cos = np.concatenate([np.sum(g32 * g16, 1), np.sum(q32 * q16, 1)])
    return {
        "gallery_refs": len(refs),
        "dev_queries": len(queries),
        "descriptor_cosine_fp16_vs_fp32": {"min": float(cos.min()), "mean": float(cos.mean())},
        "fp16_both": {
            "top1_same": int((t16[:, 0] == t32[:, 0]).sum()),
            "top5_identical_order": int((t16 == t32).all(1).sum()),
            "top5_same_set": int(sum(set(a) == set(b) for a, b in zip(t16, t32, strict=True))),
        },
        "fp16_gallery_fp32_query": {
            "top1_same": int((t_mixed[:, 0] == t32[:, 0]).sum()),
            "top5_same_set": int(sum(set(a) == set(b) for a, b in zip(t_mixed, t32, strict=True))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--one",
        metavar="DTYPE:BATCH",
        help="Time a single setting in this process (isolates MPS memory)",
    )
    parser.add_argument(
        "--fp16-check",
        type=int,
        metavar="BATCH",
        help="Only run the fp16-vs-fp32 retrieval check at this batch size",
    )
    args = parser.parse_args()
    if args.one or args.fp16_check:
        import torch

        embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), args.device)
        if args.fp16_check:
            result: dict[str, Any] = {"fp16_check": fp16_check(embedder, args.fp16_check)}
        else:
            name, batch_text = args.one.split(":")
            dtype, batch = getattr(torch, name), int(batch_text)
            refs = json.loads((GALLERY / "manifest.json").read_text())["references"]
            rng = np.random.default_rng(0)
            chosen = [refs[i] for i in rng.choice(len(refs), args.images, replace=False)]
            blobs = [stored_version(GALLERY / r["path"]) for r in chosen]
            tensors = [embedder.transform(decode(b)) for b in blobs]
            run(embedder, tensors[: 2 * batch], batch, dtype)  # Warm-up at this batch size.
            result = timed(embedder, blobs, tensors, batch, dtype, "cached_pos_encoding")
            if embedder.device == "mps":
                result["mps_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 2**30
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        return
    embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), args.device)
    refs = json.loads((GALLERY / "manifest.json").read_text())["references"]
    rng = np.random.default_rng(0)
    chosen = [refs[i] for i in rng.choice(len(refs), args.images, replace=False)]
    blobs = [stored_version(GALLERY / r["path"]) for r in chosen]
    identical = pos_cache_identity(embedder, blobs)
    print(json.dumps({"pos_cache_identity": identical}), flush=True)
    rows = throughput(embedder, blobs)
    best32 = max(
        (r for r in rows if r["dtype"] == "float32" and r["variant"] == "cached_pos_encoding"),
        key=lambda r: r["end_to_end_img_per_s"],
    )
    check = fp16_check(embedder, best32["batch"])
    result = {
        "device": args.device,
        "images": args.images,
        "throughput": rows,
        "pos_cache_identity": identical,
        "fp16_check": check,
        "identity": embedder.identity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
