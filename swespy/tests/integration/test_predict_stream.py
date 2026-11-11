"""POST /predict/stream: event order and schema, same result as /predict, cancellation."""

import asyncio
import io
import json
import os
import threading
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from swespy.api.routes.predict import event_stream


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def app(monkeypatch):
    os.environ["TESTING"] = "true"
    from swespy.api.main import create_app
    from swespy.config import get_settings

    get_settings.cache_clear()
    return create_app()


def lines(body: bytes) -> list[dict]:
    return [json.loads(x) for x in body.decode().splitlines() if x]


async def test_stream_event_order_schema_and_result_equals_predict(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"file": ("x.png", png(), "image/png")}
        streamed = await c.post("/predict/stream", files=files)
        plain = await c.post("/predict", files={"file": ("x.png", png(), "image/png")})
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("application/x-ndjson")
    events = lines(streamed.content)
    types = [e["type"] for e in events]
    assert types[:3] == ["stage", "candidates", "stage"]
    assert events[0]["stage"] == "retrieval" and events[2]["stage"] == "verification"
    assert types[3:-1] == ["verified"] * events[2]["count"] and types[-1] == "result"
    items = events[1]["items"]
    assert len(items) == 50 and set(items[0]) == {"id", "lat", "lng", "similarity"}
    assert all(set(e) == {"type", "id", "inliers", "score"} for e in events[3:-1])
    result = {k: v for k, v in events[-1].items() if k != "type"}
    expected = plain.json()
    for r in (result, expected):
        r.pop("processing_time_ms")
    assert result == expected


async def test_stream_validates_like_predict(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        bad = await c.post("/predict/stream", files={"file": ("x.txt", b"hi", "text/plain")})
    assert bad.status_code == 400


async def test_disconnect_cancels_the_work():
    progress = []
    stopped = threading.Event()

    class SlowLocator:
        def versions(self):
            return {}

        def locate_events(self, contents, region, cancelled):
            from swespy.model.locator import Cancelled

            yield {"type": "stage", "stage": "retrieval"}
            for n in range(100):  # Would take 5 s if never cancelled.
                time.sleep(0.05)
                if cancelled():
                    stopped.set()
                    raise Cancelled
                progress.append(n)
            yield {"type": "candidates", "items": []}

    calls = {"n": 0}

    async def disconnected():
        calls["n"] += 1
        return calls["n"] >= 2  # The client leaves right after the first event.

    got = [e async for e in event_stream(SlowLocator(), b"", None, disconnected)]
    assert [json.loads(e)["type"] for e in got] == ["stage"]
    assert await asyncio.to_thread(stopped.wait, 3.0), "work was not cancelled"
    assert len(progress) < 100


REAL = Path(__file__).resolve().parents[2]
DEV = REAL / "data/public_eval/recent_linkoping_streets_v1"


@pytest.mark.skipif(os.environ.get("SWESPY_RUN_REAL") != "1",
                    reason="Needs local weights, gallery and dev photos")
async def test_real_stream_result_equals_predict_for_three_dev_photos(monkeypatch):
    monkeypatch.chdir(REAL)
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1000")
    from swespy.api.main import build_locator, create_app
    from swespy.config import get_settings

    get_settings.cache_clear()
    application = create_app()
    application.state.locator = build_locator(get_settings())
    queries = json.loads((DEV / "manifest.json").read_text())["queries"]
    async with AsyncClient(transport=ASGITransport(app=application), base_url="http://t") as c:
        for q in [queries[i] for i in (0, 1, 6)]:
            raw = (DEV / q["path"]).read_bytes()
            plain = (await c.post("/predict", files={"file": ("q.jpg", raw, "image/jpeg")})).json()
            events = lines((await c.post("/predict/stream",
                                         files={"file": ("q.jpg", raw, "image/jpeg")})).content)
            result = {k: v for k, v in events[-1].items() if k != "type"}
            for r in (result, plain):
                r.pop("processing_time_ms")
            # One code path (locate_events) serves both routes, so they must agree exactly.
            assert result == plain, q["image_id"]
