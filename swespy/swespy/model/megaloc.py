"""Pinned pretrained MegaLoc global descriptor: the only retrieval encoder in SweSpy.

Shared by the API and every evaluation script, so both encode images identically.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np

SOURCE_REVISION = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"

SOURCE_SHA256 = "e3a66ae823abbbbc5e7975c649419a33036400eaa9fa5a955dd97eaa37f46c75"

WEIGHTS_REVISION = "a0f34722c4297ff787e022433799250180860af7"

DIM = 8448

PREPROCESSING = {
    "rgb": True,
    "to_tensor": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "resize_after_normalization": [322, 322],
    "antialias": True,
    "resize_interpolation": "bilinear",
    "dtype": "float32",
    "augmentation": "none",
    "whitening_or_pca": "none",
}


# Identity fields that define the descriptor space; device/software details may differ
# between a gallery built on Apple MPS and a CPU/CUDA server without changing it.
MODEL_IDENTITY_KEYS = (
    "source_repository",
    "source_revision",
    "source_sha256",
    "weights_repository",
    "weights_revision",
    "weights_sha256",
    "preprocessing",
)


def same_descriptor_space(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a.get(k) == b.get(k) for k in MODEL_IDENTITY_KEYS)


def file_hash(path: Path) -> str:
    """Hash large weights/files without loading their complete contents into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def cache_pos_encoding(model: Any) -> int:
    """Memoise DINOv2 positional-embedding interpolation per (size, dtype, device).

    The interpolation depends only on the input size, which is fixed (322x322). Its
    bicubic operator has no MPS kernel, so without the cache every forward pass falls
    back to the CPU and synchronises with the GPU. The cached tensor is the exact
    result of the original method; descriptors are unchanged (verified bit-identical
    in eval/encode_benchmark.py). Returns the number of patched modules.
    """
    patched = 0
    for module in model.modules():
        original = getattr(module, "interpolate_pos_encoding", None)
        if original is None or getattr(original, "_cached", False):
            continue
        memo: dict[tuple[Any, ...], Any] = {}

        def cached(
            x: Any,
            w: int,
            h: int,
            _original: Any = original,
            _memo: dict[tuple[Any, ...], Any] = memo,
        ) -> Any:
            key = (x.shape[1], x.shape[-1], w, h, x.dtype, x.device)
            if key not in _memo:
                _memo[key] = _original(x, w, h).detach()
            return _memo[key]

        cached._cached = True  # type: ignore[attr-defined]
        module.interpolate_pos_encoding = cached
        patched += 1
    return patched


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
        downloaded[filename] = Path(
            hf_hub_download(
                repo_id="gberton/MegaLoc",
                filename=filename,
                revision=WEIGHTS_REVISION,
                cache_dir="data/huggingface/hub",
                token=False,
            )
        )
    weights = downloaded["model.safetensors"]
    return (
        code,
        weights,
        {
            "source_repository": "https://github.com/gmberton/MegaLoc",
            "source_revision": SOURCE_REVISION,
            "source_sha256": file_hash(code),
            "weights_repository": "https://huggingface.co/gberton/MegaLoc",
            "weights_revision": WEIGHTS_REVISION,
            "weights_sha256": file_hash(weights),
            "source_license_sha256": file_hash(source / "LICENSE"),
            "weights_license_sha256": file_hash(downloaded["LICENSE"]),
            "preprocessing": PREPROCESSING,
        },
    )


class MegaLocEmbedder:
    """Load pinned MegaLoc once; encode paths (with a resumable cache) or in-memory images."""

    def __init__(
        self, root: Path, device: str, dtype: str = "float32", cache_pos: bool = True
    ) -> None:
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
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
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
        self.transform = tfm.Compose(
            [
                tfm.ToTensor(),
                tfm.Normalize(mean=PREPROCESSING["mean"], std=PREPROCESSING["std"]),
                tfm.Resize(size=[322, 322], antialias=True),
            ]
        )
        self.identity = {
            **identity,
            "device": device,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            "conv_input_layout": "contiguous" if device == "mps" else "upstream",
        }
        if dtype not in ("float32", "float16"):
            raise ValueError("dtype must be float32 or float16")
        if dtype != "float32":
            # Only non-default precision enters the identity, so existing fp32 caches keep
            # their keys. fp16 output is re-normalised in fp32 (see batch_images).
            self.identity["dtype"] = dtype
        self.dtype = getattr(torch, dtype)
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.cache = root / "descriptors" / key
        self.cache.mkdir(parents=True, exist_ok=True)
        # Weights stay fp32: the pinned model mixes in fp32 constants, so a plain .half()
        # fails; fp16 runs through autocast instead (see batch_tensors).
        self.model.to(device)
        self.pos_cache_modules = cache_pos_encoding(self.model) if cache_pos else 0

    def batch(self, paths: list[Path]) -> np.ndarray:
        """Apply the official single-image preprocessing without crops or test augmentation."""
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return self.batch_images(images)

    def batch_images(self, images: list[Any]) -> np.ndarray:
        """Encode RGB PIL images held in memory."""
        return self.batch_tensors([self.transform(image.convert("RGB")) for image in images])

    def batch_tensors(self, tensors: list[Any]) -> np.ndarray:
        """Encode already-preprocessed CHW tensors (lets callers decode in parallel)."""
        import torch

        half = self.dtype == torch.float16
        with torch.inference_mode(), torch.autocast(self.device, torch.float16, enabled=half):
            x = torch.stack(tensors).to(self.device)
            y = self.model(x).float()
            if self.dtype != torch.float32:
                y = torch.nn.functional.normalize(y, dim=1)
            result = y.cpu().numpy()
        return checked_vectors(result, len(tensors))

    def embed_bytes(self, contents: bytes) -> np.ndarray:
        """Encode one uploaded image without writing it anywhere; returns shape (DIM,)."""
        from PIL import Image

        with Image.open(BytesIO(contents)) as image:
            return self.batch_images([image.convert("RGB")])[0]

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
            "image_count": len(paths),
            "device": selected,
            "cpu_seconds": cpu_seconds,
            "accelerated_first_batch_seconds": seconds,
            "minimum_cpu_device_cosine": float(cosine.min()),
            "maximum_abs_difference": difference,
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
            indices = pending[start : start + batch_size]
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
            print(
                f"Embedded {len(paths) - len(pending) + start + len(indices)}/{len(paths)}",
                flush=True,
            )
        return checked_vectors(result, len(paths))
