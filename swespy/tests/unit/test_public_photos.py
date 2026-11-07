"""Public-photo tests must enforce capture recency and fail honestly on blocked acquisition."""

import json
from datetime import date
from io import BytesIO

import httpx
import pytest
from PIL import Image

from eval.public_photos import SINCE, UNTIL, acquire, candidate, taken_date


def page(captured="2025-04-24 13:10:00"):
    values = {"DateTimeOriginal": captured, "DateTime": "2026-10-01",
              "LicenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/",
              "LicenseShortName": "CC BY-SA 4.0", "Artist": "<a>Photographer</a>",
              "GPSLatitude": "58.411793", "GPSLongitude": "15.619310"}
    return {"pageid": 123, "title": "File:Linköping street.jpg", "imageinfo": [{
        "mime": "image/jpeg", "width": 4032, "height": 3024,
        "url": "https://upload.wikimedia.org/example.jpg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:example.jpg",
        "extmetadata": {k: {"value": v} for k, v in values.items()},
    }]}


@pytest.mark.parametrize("captured", ["2022-01-01", "2023-11-05", "2026-11-07", "", "2025"])
def test_upload_date_does_not_qualify_old_unknown_or_future_capture(captured):
    assert candidate(page(captured), SINCE, UNTIL) is None


def test_exact_capture_window_endpoints_and_exif_date_format_are_accepted():
    assert candidate(page("2023-11-06"), SINCE, UNTIL)["capture_date"] == "2023-11-06"
    assert candidate(page("2026-11-06"), SINCE, UNTIL)["capture_date"] == "2026-11-06"
    assert taken_date("2024:02:29 12:00:00") == date(2024, 2, 29)
    assert taken_date("2025-02-29") is None


def test_camera_gps_and_license_are_required():
    row = page()
    del row["imageinfo"][0]["extmetadata"]["GPSLatitude"]
    row["coordinates"] = [{"lat": 58.41, "lon": 15.62}]
    assert candidate(row, SINCE, UNTIL) is None  # Object location is not a camera label.
    row = page()
    row["imageinfo"][0]["extmetadata"]["LicenseUrl"]["value"] = "https://example.org/copyright"
    assert candidate(row, SINCE, UNTIL) is None
    row = page()
    row["imageinfo"][0]["extmetadata"]["GPSLongitude"]["value"] = "18.0"
    assert candidate(row, SINCE, UNTIL) is None


@pytest.mark.asyncio
async def test_403_is_not_retried_or_published_as_an_empty_test_set(tmp_path, monkeypatch):
    import eval.public_photos as module

    gallery = tmp_path/"gallery"
    gallery.mkdir()
    (gallery/"manifest.json").write_text('{"references": []}')
    monkeypatch.setattr(module, "DEFAULT_ROOT", gallery)
    calls = []

    def blocked(request):
        assert "https://github.com/brickphone/heimdall-spy" in request.headers["User-Agent"]
        assert "Bot/" in request.headers["User-Agent"]
        calls.append(request)
        return httpx.Response(403, text="Please respect our robot policy")

    original = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(blocked), **kwargs))
    root = tmp_path/"public"
    with pytest.raises(httpx.HTTPStatusError):
        await acquire(root, SINCE, UNTIL, 75)
    assert len(calls) == 1
    saved = json.loads((root/"acquisition-status.json").read_text())
    assert saved["state"] == "blocked" and saved["http_status"] == 403
    assert not saved["photos_frozen"]
    assert not (root/"manifest.json").exists()
    assert not (root/"selection.json").exists()


@pytest.mark.asyncio
async def test_successful_acquisition_strips_image_metadata_before_review(tmp_path, monkeypatch):
    import eval.public_photos as module

    gallery = tmp_path/"gallery"
    gallery.mkdir()
    (gallery/"manifest.json").write_text('{"references": []}')
    monkeypatch.setattr(module, "DEFAULT_ROOT", gallery)
    image = Image.new("RGB", (40, 30), "red")
    exif = Image.Exif()
    exif[36867] = "2025:04:24 13:10:00"
    exif[270] = "Location answer key must not reach the encoder"
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    def respond(request):
        if request.url.host == "upload.wikimedia.org":
            return httpx.Response(200, content=buffer.getvalue())
        if request.url.params.get("list") == "geosearch":
            return httpx.Response(200, json={"query": {"geosearch": [{"pageid": 123}]}})
        return httpx.Response(200, json={"query": {"pages": {"123": page()}}})

    async def no_sleep(_):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    original = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    root = tmp_path/"public"
    await acquire(root, SINCE, UNTIL, 1)
    saved = json.loads((root/"selection.json").read_text())
    query = saved["queries"][0]
    assert query["capture_date"] == "2025-04-24"
    assert query["lat"] == 58.411793
    with Image.open(root/query["path"]) as pixels:
        assert not pixels.getexif()
    assert not (root/"manifest.json").exists()  # Review must precede freezing/inference.
    assert json.loads((root/"acquisition-status.json").read_text())["state"] == \
        "awaiting_image_review"
