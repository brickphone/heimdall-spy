"""Upload decoding: pixels only (no EXIF/GPS), orientation applied, HEIC accepted."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from swespy.model import imageio

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pattern.heic"


def pattern(w=60, h=40):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, : w // 2] = (220, 40, 40)  # Red left half.
    a[:, w // 2 :] = (30, 90, 160)
    return Image.fromarray(a)


def jpeg(image, exif=None):
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95, exif=exif if exif is not None else b"")
    return buf.getvalue()


def gps_exif(lat, lng):
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (float(int(lat)), (lat % 1) * 60, 0.0),
                3: "E", 4: (float(int(lng)), (lng % 1) * 60, 0.0)})
    return exif


def test_gps_exif_is_dropped_and_does_not_change_pixels():
    base = pattern()
    plain = imageio.decode_upload(jpeg(base))
    tagged_bytes = jpeg(base, gps_exif(59.33, 18.07))  # Stockholm, far from Linköping.
    with Image.open(BytesIO(tagged_bytes)) as raw:
        assert 0x8825 in raw.getexif()  # The upload really carries GPS.
    tagged = imageio.decode_upload(tagged_bytes)
    assert np.array_equal(np.asarray(plain), np.asarray(tagged))
    assert not tagged.getexif() and not tagged.info  # No metadata survives decoding.


def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6  # "Rotate 90 CW to display" (typical portrait iPhone photo).
    decoded = imageio.decode_upload(jpeg(pattern(60, 40), exif))
    assert decoded.size == (40, 60)  # Upright: width/height swapped.
    top = np.asarray(decoded)[:20].reshape(-1, 3).mean(0)
    assert top[0] > 150 and top[2] < 100  # Red half is now on top.


def test_heic_upload_is_decoded():
    pytest.importorskip("pillow_heif")
    contents = FIXTURE.read_bytes()
    assert imageio.is_heif(contents)
    decoded = imageio.decode_upload(contents)
    assert decoded.size == (64, 48) and decoded.mode == "RGB"
    left = np.asarray(decoded)[:, :20].reshape(-1, 3).mean(0)
    assert left[0] > 150  # The red block survived HEVC compression.


def test_heic_without_extra_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(imageio, "_heif_registered", False)
    with pytest.raises(imageio.UnsupportedImage, match="HEIC"):
        imageio.decode_upload(FIXTURE.read_bytes())


def test_garbage_is_rejected():
    with pytest.raises(imageio.UnsupportedImage):
        imageio.decode_upload(b"not an image")


def test_crop_region_normalised_and_validated():
    from swespy.model.imageio import UnsupportedImage, crop_region

    img = Image.new("RGB", (1000, 800))
    out = crop_region(img, {"x": 0.1, "y": 0.25, "width": 0.5, "height": 0.5})
    assert out.size == (500, 400)
    assert crop_region(img, None) is img
    for bad in ({"x": 0.9, "y": 0, "width": 0.5, "height": 0.5},
                {"x": 0, "y": 0, "width": 0.01, "height": 0.5},
                {"x": -0.1, "y": 0, "width": 0.5, "height": 0.5},
                {"y": 0, "width": 0.5, "height": 0.5}):
        with pytest.raises(UnsupportedImage):
            crop_region(img, bad)


def test_preview_is_upright_small_jpeg_without_metadata() -> None:
    import base64

    from swespy.model.imageio import PREVIEW_SIDE, preview_jpeg

    image = Image.new("RGB", (4000, 3000), (10, 120, 200))
    url = preview_jpeg(image)
    assert url.startswith("data:image/jpeg;base64,")
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as out:
        assert out.format == "JPEG" and max(out.size) == PREVIEW_SIDE
        assert out.size == (1600, 1200) and not out.getexif()
