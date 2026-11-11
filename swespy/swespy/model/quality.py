"""Image quality rules shared by the gallery build (eval/quality.py) and the API's
automatic search variants. Each rule has one documented threshold; see eval/quality.py
for how the thresholds were calibrated.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

BLUR_EFFECT_THRESHOLD = 0.47
EDGE_SHARE_THRESHOLD = 0.02
EDGE_MAGNITUDE = 20.0


def blur_effect(a: np.ndarray, size: int = 9) -> float:
    """Crete et al. 2007 blur metric on a greyscale array; max over both directions."""
    values = []
    for axis in (0, 1):
        f = np.moveaxis(a, axis, 0)
        c = np.cumsum(np.pad(f, ((size // 2 + 1, size // 2), (0, 0)), mode="edge"), axis=0)
        b = (c[size:] - c[:-size]) / size  # Box filter along ``axis``.
        df = np.abs(np.diff(f, axis=0))[size // 2:-size // 2]
        db = np.abs(np.diff(b, axis=0))[size // 2:-size // 2]
        s_f = df.sum()
        if s_f > 0:  # A direction with no variation at all carries no blur evidence.
            values.append(float((s_f - np.maximum(0, df - db).sum()) / s_f))
    return max(values) if values else 1.0


def measures(image: Image.Image) -> dict[str, float]:
    gray = ImageOps.exif_transpose(image).convert("L")
    gray.thumbnail((512, 512))
    a = np.asarray(gray, dtype=np.float32)
    lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    gx = a[1:-1, 2:] - a[1:-1, :-2]
    gy = a[2:, 1:-1] - a[:-2, 1:-1]
    edges = np.hypot(gx, gy) > EDGE_MAGNITUDE
    return {"laplacian_var": float(lap.var()), "edge_share": float(edges.mean()),
            "blur_effect": blur_effect(a.astype(np.float64))}


def image_rule(m: dict[str, float]) -> str | None:
    # Featureless first: a frame with no gradients also scores as maximally "blurred".
    if m["edge_share"] < EDGE_SHARE_THRESHOLD:
        return "sky_or_ground_only"
    if m["blur_effect"] >= BLUR_EFFECT_THRESHOLD:
        return "blurry"
    return None
