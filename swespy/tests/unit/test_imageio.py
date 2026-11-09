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
