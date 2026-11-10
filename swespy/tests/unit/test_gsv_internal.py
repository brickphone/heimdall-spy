"""Internal-backend Street View pilot: everything mocked, no network."""

import json
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from eval import gsv_internal_fetch as f

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif", ".bmp", ".tif", ".tiff"}
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8", b"BM")


# ---------------------------------------------------------------- rate limiter


def test_rate_limiter_spacing_with_fake_clock():
    now = [0.0]
    limiter = f.RateLimiter(spacing=1.0, clock=lambda: now[0],
                            sleep=lambda s: now.__setitem__(0, now[0] + s))
    for duration in (0.3, 0.0, 2.0, 0.1):
        with limiter:
            now[0] += duration
    gaps = [s - e for s, e in zip(limiter.starts[1:], limiter.ends, strict=False)]
    assert all(g >= 1.0 - 1e-9 for g in gaps)
    assert limiter.max_in_flight == 1


def test_rate_limiter_one_in_flight_across_threads():
    limiter = f.RateLimiter(spacing=0.05)

    def work():
        for _ in range(3):
            with limiter:
                time.sleep(0.01)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert limiter.max_in_flight == 1
    gaps = [s - e for s, e in zip(limiter.starts[1:], limiter.ends, strict=False)]
    assert len(gaps) == 11 and min(gaps) >= 0.05 - 1e-3


# ---------------------------------------------------------------- circuit breaker


@pytest.mark.parametrize("status, ctype, body, url, reason", [
    (403, "text/html", b"", "https://x/tile", "http_403"),
    (429, "application/json", b"{}", "https://x/tile", "http_429"),
    (200, "text/html; charset=UTF-8", b"<html>Before you continue to Google</html>",
     "https://x/tile", "consent_or_captcha"),
    (200, "text/html", b"<html>Our systems have detected unusual traffic</html>",
     "https://x/tile", "consent_or_captcha"),
    (200, "image/jpeg", b"\xff\xd8\xff", "https://consent.google.com/ml?continue=x",
     "consent_or_captcha"),
    (302, "text/html", b"", "https://www.google.com/sorry/index?continue=x",
     "consent_or_captcha"),
])
def test_breaker_stops_on_blocks(status, ctype, body, url, reason):
    with pytest.raises(f.PilotStopped) as stop:
        f.Breaker().check(status, ctype, body, url, "image")
    assert stop.value.reason == reason


def test_breaker_three_consecutive_errors_and_reset():
    b = f.Breaker()
    assert not b.check(500, "text/plain", b"", "https://x", "image")
    assert b.check(200, "image/jpeg", b"\xff\xd8", "https://x", "image")  # Resets.
    assert not b.check(500, "text/plain", b"", "https://x", "image")
    assert not b.check(200, "application/json", b"{}", "https://x", "image")  # Wrong type.
    with pytest.raises(f.PilotStopped) as stop:
        b.error("ConnectionError")
    assert stop.value.reason == "consecutive_errors"


# ---------------------------------------------------------------- photosphere filter


def pano(**kw):
    base = dict(id="AF1QipXYZ" * 0 + "a" * 22, lat=58.41, lng=15.62, heading_deg=0.0,
                date="2023-06", copyright="© 2023 Google", source="launch", third_party=False,
                historical=2, sizes=[(512, 256), (1024, 512), (2048, 1024), (4096, 2048)],
                tile=(512, 512))
    base.update(kw)
    return f.Pano(**base)


def test_official_capture_filter():
    assert f.is_official(pano())
    assert f.is_official(pano(source="scout"))
    assert not f.is_official(pano(third_party=True))
    assert not f.is_official(pano(copyright="© Anna Svensson"))
    assert not f.is_official(pano(source="innerspace"))
    assert not f.is_official(pano(source="photos:street_view_android"))


# ---------------------------------------------------------------- zoom, rows, rendering


def test_zoom_and_rows_for_level_views():
    assert f.choose_zoom([(512, 256), (1024, 512), (2048, 1024), (4096, 2048)]) == 3
    assert f.choose_zoom([(416, 208), (832, 416), (1664, 832), (3328, 1664)]) == 3
    assert f.choose_zoom([(512, 256), (1024, 512)]) is None
    assert f.needed_rows(2048, 512) == [1, 2]  # Rows 0 and 3 (sky/ground) are skipped.
    assert f.needed_rows(1664, 512) == [0, 1, 2]


def synthetic_pano(width=4096, height=2048, pano_heading=30.0):
    """Red stripe at compass 90°, green at 135°, blue horizon line."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    col = lambda compass: int((((compass - pano_heading) / 360 + 0.5) % 1.0) * width)  # noqa: E731
    img[:, col(90) - 3:col(90) + 3, 0] = 255
    img[:, col(135) - 3:col(135) + 3, 1] = 255
    img[height // 2 - 2:height // 2 + 2, :, 2] = 255
    return img


def test_render_view_heading_fov_and_horizon():
    full = synthetic_pano()
    rows = f.needed_rows(2048, 512)
    band = full[rows[0] * 512:(rows[-1] + 1) * 512]
    view = f.render_view(band, rows[0] * 512, 2048, 30.0, heading=90.0)
    assert view.shape == (640, 640, 3)
    red_cols = np.where(view[320, :, 0] > 128)[0]
    assert abs(red_cols.mean() - 319.5) < 3  # Compass 90° is the centre column.
    green_cols = np.where(view[320, :, 1] > 128)[0]
    assert green_cols.min() > 630  # 45° to the right = right edge (FOV 90).
    blue_rows = np.where(view[:, 100, 2] > 128)[0]
    assert abs(blue_rows.mean() - 319.5) < 3  # Pitch 0: horizon in the middle.
    left = f.render_view(band, rows[0] * 512, 2048, 30.0, heading=135.0)
    assert abs(np.where(left[320, :, 1] > 128)[0].mean() - 319.5) < 3
    assert np.where(left[320, :, 0] > 128)[0].max() < 10  # Red now at the left edge.


def test_view_headings_relative_to_way():
    assert f.view_headings(30.0) == [30.0, 120.0, 210.0, 300.0]
    assert f.view_headings(None) == [0.0, 90.0, 180.0, 270.0]


def test_real_library_url_builder_and_rows():
    """The library's own tile URL builder, filtered to the rows a level view needs."""
    pytest.importorskip("streetlevel", reason="pilot-only library (.venv-gsv-internal)")
    sv = f.import_streetlevel()
    from streetlevel.dataclasses import Size

    raw = sv.StreetViewPanorama(id="a" * 22, lat=58.41, lon=15.62, tile_size=Size(512, 512),
                                image_sizes=[Size(512, 256), Size(1024, 512), Size(2048, 1024),
                                             Size(4096, 2048)])
    lib = f.Library.__new__(f.Library)
    lib.sv = sv
    urls = lib.tile_urls(pano(raw=raw), 3)
    assert len(urls) == 32
    needed = {k: u for k, u in urls.items() if k[1] in f.needed_rows(2048, 512)}
    assert len(needed) == 16
    assert all("zoom=3" in u and "streetviewpixels-pa.googleapis.com" in u
               for u in needed.values())


# ---------------------------------------------------------------- pull log


def test_pull_log_schema(tmp_path):
    log = f.PullLog(tmp_path / "pull_log.jsonl", "runpod:test")
    log.append("search", 10, 9, (15.5, 58.3, 15.7, 58.5))
    row = log.append("tile", 32, 32, None, "completed")
    assert tuple(row) == f.PULL_LOG_FIELDS
    rows = [json.loads(x) for x in (tmp_path / "pull_log.jsonl").read_text().splitlines()]
    assert [r["cumulative_count"] for r in rows] == [10, 32]
    text = (tmp_path / "pull_log.jsonl").read_text()
    assert "http" not in text and "\\xff" not in text


# ---------------------------------------------------------------- mocked end-to-end


def jpeg_tile(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8).repeat(8, 0).repeat(8, 1)
    out = BytesIO()
    Image.fromarray(a).save(out, format="JPEG")
    return out.getvalue()


def response(status=200, ctype="image/jpeg", body=b"", url="https://t/tile"):
    r = requests.models.Response()
    r.status_code, r._content, r.url = status, body, url
    r.headers["content-type"] = ctype
    return r


class FakeTransport:
    """Stands in for requests.Session: searches answer JSON, tiles answer JPEG."""

    def __init__(self, fail_at: int | None = None, fail_status: int = 429):
        self.calls = 0
        self.fail_at = fail_at
        self.fail_status = fail_status

    def request(self, method, url, timeout=None, **kw):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            return response(self.fail_status, "text/html", b"", url)
        if method == "POST":
            return response(body=json.dumps({"q": url}).encode(), ctype="application/json")
        return response(body=jpeg_tile(self.calls), url=url)


class FakeLibrary:
    """Same interface as f.Library; searches go through the guarded session."""

    def __init__(self, session):
        self.session = session

    def search(self, lat, lng):
        self.session.kind = "search"
        self.session.post("https://search", data=b"{}")
        if lat > 58.45:
            return None
        if lat > 58.44:
            return pano(id="user" + "x" * 30, third_party=True)
        return pano(id=f"P{round(lat * 1e4)}", lat=lat, lng=lng, heading_deg=17.0)

    def tile_urls(self, p, zoom):
        w, h = p.sizes[zoom]
        return {(x, y): f"https://t/tile?x={x}&y={y}" for x in range(w // 512)
                for y in range(h // 512)}


def points(n=6):
    pts = [{"point_id": f"p1-{i:03d}", "lat": 58.40 + i * 0.001, "lng": 15.62,
            "priority": 1, "highway": "footway", "bearing": 10.0, "area": 0}
           for i in range(n)]
    pts.append({**pts[0], "point_id": "p2-900", "lat": 58.445, "priority": 2,
                "highway": "service"})  # Photosphere: filtered.
    pts.append({**pts[0], "point_id": "p2-901", "lat": 58.46, "priority": 2})  # No pano.
    pts.append({**pts[0], "point_id": "p2-902", "priority": 2})  # Same pano as p1-000.
    return pts


def fake_embed(images):
    assert all(isinstance(im, Image.Image) and im.size == (640, 640) for im in images)
    v = np.random.default_rng(len(images)).standard_normal((len(images), 8448))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def scan_for_images(root: Path) -> list[Path]:
    found = []
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix.lower() in IMAGE_SUFFIXES
                            or p.read_bytes()[:4].startswith(IMAGE_MAGIC)):
            found.append(p)
    return found


def run_mocked(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(f, "quality_rule", lambda view: None)  # Synthetic tiles are noise.
    limiter = f.RateLimiter(spacing=0.0)
    session = f.GuardedSession(limiter, f.Breaker(), deadline=time.time() + 60,
                               session=transport)
    return f.run(points(), tmp_path / "out", FakeLibrary(session), session, fake_embed,
                 "test", log_every=3)


def test_pipeline_writes_descriptors_only(tmp_path, monkeypatch):
    summary = run_mocked(tmp_path, FakeTransport(), monkeypatch)
    assert summary["stop_reason"] == "completed"
    assert summary["max_in_flight"] == 1
    assert summary["tiles_per_pano"] == {16: 6}  # Middle rows only, 6 unique panos.
    folder = tmp_path / "out" / f.INDEX_NAME
    views = [json.loads(x) for x in (folder / "views.jsonl").read_text().splitlines()]
    assert len(views) == 24 and {v["source"] for v in views} == {"gsv_internal"}
    assert all(v["attribution"] == "Google Street View, © Google" for v in views)
    assert "license" not in json.dumps(views).lower()
    desc = np.concatenate([np.load(p) for p in sorted(folder.glob("descriptors-*.npy"))])
    assert desc.dtype == np.float16 and desc.shape == (24, 8448)
    outcomes = [json.loads(x)["outcome"] for x in (folder / "points.jsonl").read_text()
                .splitlines()]
    assert outcomes.count("fetched") == 6 and "not_official" in outcomes
    assert "no_pano" in outcomes and "duplicate_pano" in outcomes
    assert scan_for_images(tmp_path) == []
    log = [json.loads(x) for x in (tmp_path / "out" / "pull_log.jsonl").read_text()
           .splitlines()]
    assert sum(r["request_count"] for r in log if r["request_type"] == "search") == 9
    assert log[-1]["stop_reason"] == "completed"


def test_pipeline_stops_on_429_without_further_requests(tmp_path, monkeypatch):
    transport = FakeTransport(fail_at=20, fail_status=429)
    summary = run_mocked(tmp_path, transport, monkeypatch)
    assert summary["stop_reason"] == "http_429"
    assert transport.calls == 20  # Nothing after the block.
    log = [json.loads(x) for x in (tmp_path / "out" / "pull_log.jsonl").read_text()
           .splitlines()]
    assert log[-1]["stop_reason"] == "http_429"
    assert scan_for_images(tmp_path) == []


def test_pipeline_stops_on_consent_page(tmp_path, monkeypatch):
    class Consent(FakeTransport):
        def request(self, method, url, timeout=None, **kw):
            self.calls += 1
            if self.calls == 5:
                return response(200, "text/html", b"<html>Before you continue</html>", url)
            return super().request(method, url, timeout, **kw)

    transport = Consent()
    summary = run_mocked(tmp_path, transport, monkeypatch)
    assert summary["stop_reason"].startswith("consent_or_captcha")
    assert transport.calls == 5


def test_pipeline_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "quality_rule", lambda view: None)
    session = f.GuardedSession(f.RateLimiter(spacing=0.0), f.Breaker(),
                               deadline=time.time() - 1, session=FakeTransport())
    summary = f.run(points(), tmp_path / "out", FakeLibrary(session), session, fake_embed, "t")
    assert summary["stop_reason"] == "deadline" and session.counts["search"] == 0


def test_guarded_session_adds_no_headers_to_searches():
    seen = {}

    class Spy(FakeTransport):
        def request(self, method, url, timeout=None, **kw):
            seen.update(kw.get("headers") or {})
            return super().request(method, url, timeout, **kw)

    s = f.GuardedSession(f.RateLimiter(spacing=0.0), f.Breaker(), time.time() + 9, Spy())
    s.post("https://search", data=b"x", headers={"content-type": "application/json+protobuf"})
    assert seen == {"content-type": "application/json+protobuf"}  # Library's own header only.
