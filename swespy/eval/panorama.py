"""Split equirectangular panoramas into perspective crops with derived headings.

Assumption (Mapillary convention): the panorama's centre column faces the image's
``compass_angle``; columns to the right are clockwise. A crop at yaw offset ``d``
therefore faces ``(compass_angle + d) % 360``.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

CROP_COUNT = 8
FOV_DEG = 90.0
CROP_SIZE = 640


def crop_headings(pano_heading: float | None, count: int = CROP_COUNT
                  ) -> list[tuple[float, float | None]]:
    """Return (yaw offset from the panorama centre, absolute heading or None)."""
    if count < 1:
        raise ValueError("At least one crop is required")
    offsets = [i * 360.0 / count for i in range(count)]
    return [(o, None if pano_heading is None else (pano_heading + o) % 360.0) for o in offsets]


def perspective(pano: Image.Image, yaw_deg: float, fov_deg: float = FOV_DEG,
                size: int = CROP_SIZE, pitch_deg: float = 0.0) -> Image.Image:
    """Render a pinhole view from an equirectangular image (bilinear sampling)."""
    src = np.asarray(pano.convert("RGB"), dtype=np.float32)
    height, width = src.shape[:2]
    f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(size) - size / 2 + 0.5, np.arange(size) - size / 2 + 0.5)
    # Camera rays: x right, y down, z forward; rotate by pitch then yaw.
    x, y, z = u, v, np.full_like(u, f)
    p = math.radians(pitch_deg)
    y, z = y * math.cos(p) - z * math.sin(p), y * math.sin(p) + z * math.cos(p)
    lon = np.arctan2(x, z) + math.radians(yaw_deg)
    lat = np.arctan2(-y, np.hypot(x, z))
    # Centre column (lon = 0) is the panorama's forward direction.
    px = ((lon / (2 * math.pi) + 0.5) % 1.0) * width - 0.5
    py = (0.5 - lat / math.pi) * height - 0.5
    x0, y0 = np.floor(px).astype(int), np.clip(np.floor(py).astype(int), 0, height - 1)
    x1, y1 = (x0 + 1) % width, np.clip(y0 + 1, 0, height - 1)
    x0 %= width
    wx, wy = (px - np.floor(px))[..., None], np.clip(py - np.floor(py), 0, 1)[..., None]
    top = src[y0, x0] * (1 - wx) + src[y0, x1] * wx
    bottom = src[y1, x0] * (1 - wx) + src[y1, x1] * wx
    out = top * (1 - wy) + bottom * wy
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def split(pano: Image.Image, pano_heading: float | None, count: int = CROP_COUNT
          ) -> list[tuple[Image.Image, float, float | None]]:
    """Return (crop, yaw offset, derived heading) for each of ``count`` directions."""
    return [(perspective(pano, offset), offset, heading)
            for offset, heading in crop_headings(pano_heading, count)]
