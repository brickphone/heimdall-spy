"""Readiness, versions, reference images and opt-in field-photo storage."""

import io
import json
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def app(tmp_path, monkeypatch):
    os.environ["TESTING"] = "true"
    monkeypatch.setenv("FIELD_PHOTOS_DIR", str(tmp_path / "field_photos"))
    from swespy.api.main import create_app
    from swespy.config import get_settings

    get_settings.cache_clear()
    application = create_app()
    application.state.field_photos_dir = str(tmp_path / "field_photos")
    return application


async def client_for(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_and_ready(app):
    async with await client_for(app) as c:
        assert (await c.get("/demo")).status_code == 404  # The demo is the separate web app.
        assert (await c.get("/health")).json() == {"status": "ok"}
        ready = await c.get("/ready")
        assert ready.status_code == 200 and ready.json()["ready"] is True
        del app.state.locator
        assert (await c.get("/ready")).status_code == 503


async def test_predict_is_versioned_and_stores_nothing(app, tmp_path):
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    async with await client_for(app) as c:
        r = await c.post("/predict", files={"file": ("x.png", png(), "image/png")})
    assert r.status_code == 200
    assert r.json()["versions"]["model"]["name"] == "dummy"
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert not (tmp_path / "field_photos").exists()


async def test_field_photo_saved_only_on_request_and_without_metadata(app, tmp_path):
    exif = Image.Exif()
    exif.get_ifd(0x8825).update({1: "N", 2: (59.0, 20.0, 0.0), 3: "E", 4: (18.0, 4.0, 0.0)})
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (200, 30, 30)).save(buf, format="JPEG", exif=exif)
    async with await client_for(app) as c:
        bad = await c.post("/field-photos", files={"file": ("p.jpg", buf.getvalue(), "image/jpeg")},
                           data={"prediction": "{}", "true_lat": "999", "true_lng": "15.6"})
        assert bad.status_code == 400
        ok = await c.post("/field-photos", files={"file": ("p.jpg", buf.getvalue(), "image/jpeg")},
                          data={"prediction": json.dumps({"predicted_lat": 58.4}),
                                "true_lat": "58.41", "true_lng": "15.62"})
    assert ok.status_code == 200
    folder = Path(app.state.field_photos_dir) / ok.json()["saved"]
    record = json.loads((folder / "record.json").read_text())
    assert record["true_location"]["lat"] == 58.41 and "never gallery" in record["use"]
    with Image.open(folder / "photo.jpg") as saved:
        assert 0x8825 not in saved.getexif()  # Stored without GPS.


async def test_unknown_reference_is_404(app):
    async with await client_for(app) as c:
        assert (await c.get("/references/../../etc/passwd/image")).status_code == 404
        assert (await c.get("/references/nope/image")).status_code == 404


async def test_predict_accepts_a_normalised_region(app):
    region = json.dumps({"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.6})
    async with await client_for(app) as c:
        ok = await c.post("/predict", files={"file": ("x.png", png(), "image/png")},
                          data={"region": region})
        assert ok.status_code == 200 and ok.json()["region"]["width"] == 0.5
        bad = await c.post("/predict", files={"file": ("x.png", png(), "image/png")},
                           data={"region": "not json"})
        assert bad.status_code == 400
