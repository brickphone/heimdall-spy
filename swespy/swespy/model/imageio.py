"""Decode an uploaded photo to plain RGB pixels: the only input inference ever sees.

- EXIF orientation is applied (phones store rotated pixels plus an orientation tag), so
  the pixels match how the gallery images were stored.
- The result is a fresh RGB image with **no metadata**: EXIF (including GPS), XMP and
  ICC data are dropped here, so no location information can reach the model.
- HEIC/HEIF (iPhone) is decoded when the optional ``heic`` extra (pillow-heif) is
  installed. Its binary wheels are GPL-2.0 (bundled x265 encoder); see docs/LICENSES.md.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

HEIC_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis")
_heif_registered: bool | None = None


class UnsupportedImage(ValueError):
    """The upload is not a decodable image (or HEIC support is not installed)."""


def is_heif(contents: bytes) -> bool:
    return len(contents) >= 12 and contents[4:8] == b"ftyp" and contents[8:12] in HEIC_BRANDS


def heic_supported() -> bool:
    global _heif_registered
    if _heif_registered is None:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
            _heif_registered = True
        except ImportError:
            _heif_registered = False
    return _heif_registered


def decode_upload(contents: bytes) -> Image.Image:
    """Bytes -> upright RGB image without any metadata."""
    if is_heif(contents) and not heic_supported():
        raise UnsupportedImage("HEIC upload, but HEIC support (pillow-heif) is not installed")
    heic_supported()  # Registers the opener when available, so Image.open handles HEIC.
    try:
        with Image.open(BytesIO(contents)) as image:
            upright = ImageOps.exif_transpose(image)
            rgb = upright.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedImage("Unreadable image") from exc
    # Copy pixels only: nothing from image.info / EXIF / XMP survives.
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    return clean


MIN_REGION_FRACTION = 0.05  # Each side at least 5% of the photo.
MIN_REGION_PX = 64
PREVIEW_SIDE = 1600  # Longest side of the display preview returned by /analyze.


def preview_jpeg(image: Image.Image) -> str:
    """A JPEG data URL of the decoded (upright, metadata-free) image for display and region
    drawing, at most ``PREVIEW_SIDE`` px. Browsers that cannot show HEIC use this; region
    coordinates are relative to the same upright pixels the search uses."""
    import base64

    small = image.copy()
    small.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    out = BytesIO()
    small.save(out, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def crop_region(image: Image.Image, region: dict[str, float] | None) -> Image.Image:
    """Crop to a normalised region {x, y, width, height} of the upright photo (0..1).

    Used when the user selects the part of the photo that shows the place; retrieval and
    re-ranking then both see only that part. Raises UnsupportedImage on an invalid region.
    """
    if region is None:
        return image
    try:
        x, y = float(region["x"]), float(region["y"])
        w, h = float(region["width"]), float(region["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedImage("Region must have x, y, width and height") from exc
    if not (0 <= x < 1 and 0 <= y < 1 and w >= MIN_REGION_FRACTION
            and h >= MIN_REGION_FRACTION and x + w <= 1 + 1e-6 and y + h <= 1 + 1e-6):
        raise UnsupportedImage("Region must lie inside the photo and cover at least 5% per side")
    W, H = image.size
    box = (round(x * W), round(y * H), round(min(1.0, x + w) * W), round(min(1.0, y + h) * H))
    if box[2] - box[0] < MIN_REGION_PX or box[3] - box[1] < MIN_REGION_PX:
        raise UnsupportedImage(f"Region is too small (at least {MIN_REGION_PX} px per side)")
    return image.crop(box)
