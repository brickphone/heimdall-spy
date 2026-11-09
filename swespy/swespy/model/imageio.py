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
