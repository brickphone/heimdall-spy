"""Image quality rules for gallery references. Each rule has one documented threshold.

- ``blurry``: motion/defocus blur via the no-reference blur metric of Crete et al.
  (2007): re-blur with a 9-pixel box filter per direction and measure how much
  neighbouring-pixel variation is lost (0 = sharp, 1 = blurred); the worse of the
  horizontal/vertical values is used, so directional (motion) blur is caught. It
  depends on how sharp the existing edges are, not on how many edges there are, so
  low-detail but sharp scenes (rural roads, open sky) are not rejected. Threshold
  calibrated on 150 gallery images: originals <= 0.43 (99th pct), the same images
  with Gaussian (r=2.5) or 11 px motion blur >= 0.50 (1st pct).
- ``sky_or_ground_only``: share of pixels with a meaningful gradient, measured on the
  whole image; frames of only sky, tarmac or grass have almost none.
- ``indoor``: keyword rule on source titles/descriptions (Commons). Mapillary and
  KartaView are street captures and are not keyword-screened; no image classifier is
  used, so indoor frames from street sources can remain (reported as a limitation).
- ``near_duplicate``: the same image stored twice, not a similar-looking view:
  identical SHA-256 within ``DUPLICATE_RADIUS_M`` (30 m), or pHash <= 6 / dHash <= 5
  within ``SAME_SPOT_M`` (5 m), or within 30 m when the two come from different
  sources (e.g. a Mapillary photo re-uploaded to Commons). Spatial redundancy along
  a drive is handled by sequence thinning, not here: on the 3,388 existing images a
  hash-only rule flagged 401 "duplicates", 389 of them consecutive frames of the same
  drive a median 20 m apart on low-detail roads.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageOps

from eval.geo_eval import haversine_m

BLUR_EFFECT_THRESHOLD = 0.47
EDGE_SHARE_THRESHOLD = 0.02
EDGE_MAGNITUDE = 20.0
DUPLICATE_RADIUS_M = 30.0
SAME_SPOT_M = 5.0
INDOOR_WORDS = (
    "interior", "interiör", "inside", "inomhus", "indoor", "foyer", "corridor",
    "korridor", "aula", "classroom", "lecture hall", "museum exhibit", "utställning",
    "kyrkans inre", "altar", "organ", "orgel", "painting", "målning", "portrait",
)


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


def indoor_text(*texts: str) -> bool:
    words = " ".join(texts).casefold()
    return any(w in words for w in INDOOR_WORDS)


def is_duplicate(sig: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    """``others`` are kept references within DUPLICATE_RADIUS_M (pre-filtered by grid)."""
    for o in others:
        if sig["sha256"] == o["sha256"]:
            return True
        similar = ((sig["phash"] ^ o["phash"]).bit_count() <= 6
                   or (sig["dhash"] ^ o["dhash"]).bit_count() <= 5)
        if not similar:
            continue
        if sig.get("source") != o.get("source"):
            return True
        d = float(haversine_m(sig["lat"], sig["lng"], np.array([o["lat"]]),
                              np.array([o["lng"]]))[0])
        if d <= SAME_SPOT_M:
            return True
    return False
