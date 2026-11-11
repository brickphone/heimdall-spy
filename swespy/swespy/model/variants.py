"""Automatic search variants: find the part of a photo that shows the place.

For every query the locator searches several versions of the photo and keeps, per
reference, the version that matches it best. All rules and numbers below were fixed
before any evaluation (docs/experiments/14_auto_regions.md):

(a) the whole photo, or the user's manual region when one is given;
(b) the whole photo with people masked (filled with the image's mean colour), only when
    people cover at least ``MASK_MIN_SHARE`` of it, since otherwise it equals (a);
(c) up to ``MAX_CROPS`` automatic crops of the whole photo: the largest rectangle with
    less than ``CLEAN_MAX_PERSON`` person area, left / centre / right vertical strips
    (``STRIP_WIDTH`` wide) and the central horizontal band (``BAND_HEIGHT`` high).
    A crop is skipped if it is smaller than ``CROP_MIN_AREA`` of the image or the
    existing quality rule says it is sky or ground only.

People are found with torchvision's DeepLabV3-MobileNetV3-Large (COCO with VOC labels,
class "person"); see docs/LICENSES.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from swespy.model.quality import image_rule, measures

Region = dict[str, float]

MASK_MIN_SHARE = 0.02
CLEAN_MAX_PERSON = 0.10
CROP_MIN_AREA = 0.25
STRIP_WIDTH = 0.40
BAND_HEIGHT = 0.50
MAX_CROPS = 6
GRID = 16
SUGGEST_MIN_SHARE = 0.30  # The UI pre-draws the suggested region above this share.
SEG_SIDE = 520  # Segmentation input: shorter side in pixels.
PERSON_CLASS = 15  # VOC label index of "person".
SEG_WEIGHTS = "deeplabv3_mobilenet_v3_large-fc3c493d.pth"
SEG_SHA256 = "fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70"
FULL: Region = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}

# Fixed automatic crops (normalised to the whole photo).
FIXED_CROPS: tuple[tuple[str, Region], ...] = (
    ("left", {"x": 0.0, "y": 0.0, "width": STRIP_WIDTH, "height": 1.0}),
    ("center", {"x": (1 - STRIP_WIDTH) / 2, "y": 0.0, "width": STRIP_WIDTH, "height": 1.0}),
    ("right", {"x": 1 - STRIP_WIDTH, "y": 0.0, "width": STRIP_WIDTH, "height": 1.0}),
    ("band", {"x": 0.0, "y": (1 - BAND_HEIGHT) / 2, "width": 1.0, "height": BAND_HEIGHT}),
)


@dataclass
class Variant:
    id: str  # "whole", "manual", "masked" or "crop:<name>".
    region: Region  # Normalised to the whole photo.
    image: Image.Image
    priority: int  # Lower wins ties: the manual region or whole photo first.


class PersonSegmenter:
    """Person mask for a photo (boolean array at the photo's size)."""

    def __init__(self, device: str = "cpu", weights_dir: Path | None = None) -> None:
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        if weights_dir is None:
            weights_dir = Path(torch.hub.get_dir()) / "checkpoints"
        path = weights_dir / SEG_WEIGHTS
        if path.exists():
            with path.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != SEG_SHA256:
                    raise RuntimeError("Person segmentation weights do not match the pinned hash")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.identity = {"model": "torchvision DeepLabV3-MobileNetV3-Large",
                         "weights": "COCO_WITH_VOC_LABELS_V1", "sha256": SEG_SHA256[:16]}

    def mask(self, image: Image.Image) -> np.ndarray:
        import torch

        scale = SEG_SIDE / min(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        small = image.resize(size, Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - self.mean) / self.std).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            labels = self.model(x)["out"].argmax(1)[0].cpu().numpy()
        person = Image.fromarray((labels == PERSON_CLASS).astype(np.uint8) * 255)
        return np.asarray(person.resize(image.size, Image.Resampling.NEAREST)) > 0


def person_share(mask: np.ndarray | None) -> float:
    return float(mask.mean()) if mask is not None and mask.size else 0.0


def region_area(r: Region) -> float:
    return r["width"] * r["height"]


def crop(image: Image.Image, r: Region) -> Image.Image:
    w, h = image.size
    box = (round(r["x"] * w), round(r["y"] * h), round((r["x"] + r["width"]) * w),
           round((r["y"] + r["height"]) * h))
    return image.crop(box)


def person_fraction(mask: np.ndarray, r: Region) -> float:
    h, w = mask.shape
    sub = mask[round(r["y"] * h):round((r["y"] + r["height"]) * h),
               round(r["x"] * w):round((r["x"] + r["width"]) * w)]
    return float(sub.mean()) if sub.size else 1.0


def largest_clean_rect(mask: np.ndarray, max_person: float = CLEAN_MAX_PERSON,
                       min_area: float = CROP_MIN_AREA, grid: int = GRID) -> Region | None:
    """Largest grid-aligned rectangle whose person share is below ``max_person``.

    Exhaustive over a ``grid`` x ``grid`` raster (about 18,000 rectangles). Ties go to
    the rectangle whose centre is closest to the photo's centre. None if no rectangle
    of at least ``min_area`` qualifies.
    """
    h, w = mask.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    cell = np.array([[mask[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() if ys[i + 1] > ys[i]
                      and xs[j + 1] > xs[j] else 0.0 for j in range(grid)] for i in range(grid)])
    integral = np.zeros((grid + 1, grid + 1))
    integral[1:, 1:] = cell.cumsum(0).cumsum(1)
    best: tuple[float, float, Region] | None = None
    for top in range(grid):
        for bottom in range(top + 1, grid + 1):
            for left in range(grid):
                for right in range(left + 1, grid + 1):
                    cells = (bottom - top) * (right - left)
                    area = cells / grid**2
                    if area < min_area or (best and area < best[0]):
                        continue
                    s = (integral[bottom, right] - integral[top, right] - integral[bottom, left]
                         + integral[top, left]) / cells
                    if s >= max_person:
                        continue
                    centre = (abs((top + bottom) / 2 / grid - 0.5)
                              + abs((left + right) / 2 / grid - 0.5))
                    if best is None or area > best[0] or (area == best[0] and centre < best[1]):
                        best = (area, centre, {"x": left / grid, "y": top / grid,
                                               "width": (right - left) / grid,
                                               "height": (bottom - top) / grid})
    return best[2] if best else None


def masked(image: Image.Image, mask: np.ndarray) -> Image.Image:
    a = np.asarray(image).copy()
    a[mask] = a.reshape(-1, 3).mean(0).round().astype(np.uint8)
    return Image.fromarray(a)


def make_variants(image: Image.Image, mask: np.ndarray | None, manual: Region | None = None,
                  mode: str = "full") -> list[Variant]:
    """The search variants for one photo. ``mode``: "whole", "masked" or "full"."""
    from swespy.model.imageio import crop_region

    first = Variant("manual", manual, crop_region(image, manual), 0) if manual else \
        Variant("whole", dict(FULL), image, 0)
    out = [first]
    if mode == "whole" or mask is None:
        return out
    if person_share(mask) >= MASK_MIN_SHARE:
        out.append(Variant("masked", dict(FULL), masked(image, mask), 1))
    if mode != "full":
        return out
    candidates: list[tuple[str, Region]] = []
    clean = largest_clean_rect(mask)
    if clean and region_area(clean) < 0.999:  # The whole photo is already variant (a).
        candidates.append(("clean", clean))
    candidates.extend(FIXED_CROPS)
    for name, r in candidates:
        if len([v for v in out if v.id.startswith("crop:")]) >= MAX_CROPS:
            break
        if region_area(r) < CROP_MIN_AREA:
            continue
        part = crop(image, r)
        if image_rule(measures(part)) == "sky_or_ground_only":
            continue
        out.append(Variant(f"crop:{name}", r, part, 2))
    return out


def merge(variant_hits: list[tuple[list[int], list[float]]], priorities: list[int]
          ) -> list[tuple[int, float, int]]:
    """Merge per-variant retrieval lists: (gallery index, best similarity, variant index).

    Each reference keeps its best variant; equal similarities go to the variant with the
    lower priority number (the manual region or whole photo). Sorted by similarity, with
    the same tie rule.
    """
    best: dict[int, tuple[float, int]] = {}
    for v, (indices, sims) in enumerate(variant_hits):
        for i, s in zip(indices, sims, strict=True):
            old = best.get(i)
            if old is None or s > old[0] or (s == old[0] and priorities[v] < priorities[old[1]]):
                best[i] = (s, v)
    return sorted(((i, s, v) for i, (s, v) in best.items()),
                  key=lambda t: (-t[1], priorities[t[2]]))


def analyze(image: Image.Image, mask: np.ndarray) -> dict[str, Any]:
    """Person share, the suggested region (largest people-free rectangle) and the
    automatic variants that a search would run. No retrieval."""
    share = person_share(mask)
    suggested = largest_clean_rect(mask)
    return {
        "person_share": share,
        "suggested_region": suggested,
        "suggest": share > SUGGEST_MIN_SHARE and suggested is not None,
        "variants": [{"id": v.id, "region": v.region} for v in make_variants(image, mask)],
    }
