"""Official-API Street View build: everything mocked, no network."""

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from eval import gsv_build as b
from swespy import gsv

FAKE_KEY = "AIzaFAKE_planted_key_9876543210zyxwvutsrq"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif", ".bmp", ".tif", ".tiff"}
M = 1 / 111_320


# ---------------------------------------------------------------- caps


def test_tile_daily_cap_is_exact_and_uncharged_is_not_counted(tmp_path):
    now = datetime(2026, 11, 10, 20, tzinfo=UTC)  # 12:00 Pacific.
    budget = gsv.Budget(tmp_path / "j.jsonl", now=lambda: now)
    assert budget.take("tile", 14_499)
    assert budget.remaining("tile") == 1
    assert not budget.take("tile", 2)  # Nothing reserved when it would cross the cap.
    assert budget.take("tile", 1)
    assert not budget.take("tile", 1)
    with pytest.raises(ValueError):
        budget.take("metadata", 1)  # Uncharged types can never touch the budget.
    assert budget.take("image", 9_500) and not budget.take("image", 1)


def test_daily_reset_is_midnight_pacific_and_total_cap(tmp_path):
    t = [datetime(2026, 11, 10, 7, 59, tzinfo=UTC)]  # 23:59 Pacific on Nov 9.
    budget = gsv.Budget(tmp_path / "j.jsonl", now=lambda: t[0])
    assert budget.take("tile", 14_500)
    t[0] = datetime(2026, 11, 10, 8, 1, tzinfo=UTC)  # 00:01 Pacific (09:01 Sweden).
    assert budget.remaining("tile") == 14_500
    # Total cap across days: 15 days of 14,500 would exceed 220,000.
    for day in range(1, 15):
        t[0] = datetime(2026, 11, 10, 20, tzinfo=UTC) + timedelta(days=day)
        budget.take("tile", 14_500)
    assert budget.tiles_total <= gsv.TILE_TOTAL_CAP
    t[0] = t[0] + timedelta(days=1)  # A fresh day: only the total cap limits now.
    assert budget.remaining("tile") == gsv.TILE_TOTAL_CAP - budget.tiles_total
    assert budget.take("tile", budget.remaining("tile"))
    assert budget.tiles_total == 220_000 and not budget.take("tile", 1)


def test_budget_survives_restart(tmp_path):
    now = datetime(2026, 11, 10, 20, tzinfo=UTC)
    gsv.Budget(tmp_path / "j.jsonl", now=lambda: now).take("tile", 14_000)
    again = gsv.Budget(tmp_path / "j.jsonl", now=lambda: now)
    assert again.remaining("tile") == 500


# ---------------------------------------------------------------- split


def plan_with(n):
    return {"panos": [{"pano_id": f"P{i}", "headings": [10.0, 190.0], "tiles": 12,
                       "route": None, "priority": 2} for i in range(n)]}


def test_split_never_overlaps_and_is_persistent():
    plan = plan_with(20)
    counts = b.assign(plan, tile_room=60, static_room=10, done=set())
    assert counts == {"tiles": 5, "static": 5}
    first = {p["pano_id"]: p["route"] for p in plan["panos"]}
    # Next day: more tile room, Static month spent. Earlier routes never change.
    b.assign(plan, tile_room=60 + 60, static_room=0, done=set())
    for p in plan["panos"]:
        if first[p["pano_id"]] is not None:
            assert p["route"] == first[p["pano_id"]]
    tiles = {p["pano_id"] for p in plan["panos"] if p["route"] == "tiles"}
    static = {p["pano_id"] for p in plan["panos"] if p["route"] == "static"}
    assert tiles and static and not tiles & static


# ---------------------------------------------------------------- mocked API


def jpeg(seed, size=512):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8).repeat(8, 0).repeat(8, 1)
    out = BytesIO()
    Image.fromarray(a).save(out, format="JPEG")
    return out.getvalue()


class FakeClient:
    def __init__(self, quota_after=None):
        self.calls = []
        self.quota_after = quota_after

    def request(self, method, url, params=None, json=None):
        self.calls.append(url)
        assert params["key"] == FAKE_KEY
        req = httpx.Request(method, url)
        if self.quota_after is not None and len(self.calls) > self.quota_after:
            return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED",
                                                       "message": f"quota for key {FAKE_KEY}"}},
                                  request=req)
        if url.endswith("createSession"):
            return httpx.Response(200, json={"session": "S"}, request=req)
        if url.endswith("panoIds"):
            ids = [f"P{round(loc['lat'] * 1e4)}" if loc["lat"] < 58.45 else ""
                   for loc in json["locations"]]
            return httpx.Response(200, json={"panoIds": ids}, request=req)
        if url.endswith("metadata"):
            pid = params["panoId"]
            return httpx.Response(200, json={
                "panoId": pid, "lat": 58.41, "lng": 15.62, "heading": 33.0,
                "imageWidth": 16384, "imageHeight": 8192, "tileWidth": 512, "tileHeight": 512,
                "date": "2026-05", "imageryType": "outdoor",
                "copyright": "© Someone" if pid.endswith("1") else "© 2026 Google"},
                request=req)
        return httpx.Response(200, content=jpeg(len(self.calls)),
                              headers={"content-type": "image/jpeg"}, request=req)


def points():
    pts = []
    for i in range(12):
        pts.append({"lat": 58.40 + i * 0.0003, "lng": 15.62, "priority": 1 + (i % 2),
                    "highway": "footway" if i % 2 == 0 else "residential", "bearing": 5.0,
                    "way_id": i, "gap": "empty"})
    pts.append({**pts[0], "lat": 58.46})  # No panorama.
    return pts


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "PLAN", tmp_path / "build_plan.json")
    monkeypatch.setattr(b, "quality_rule", lambda view: None)
    budget = gsv.Budget(tmp_path / "journal.jsonl")
    return tmp_path, budget


def fake_embed(images):
    assert all(im.size == (640, 640) for im in images)
    v = np.random.default_rng(len(images)).standard_normal((len(images), 8448))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def make_api(budget, client):
    api = b.Api(FAKE_KEY, budget, client=client, limiter=b.Limiter(1e6), sleep=lambda s: None)
    return api, b.LogBuffer(gsv.PullLog(budget.journal.parent / "pull_log.jsonl"), api)


def test_full_mocked_run_resume_attribution_and_no_images(env, capsys):
    root, budget = env
    client = FakeClient()
    api, logs = make_api(budget, client)
    api.create_session()
    plan = b.discover(api, logs, b.ordered_points(points()))
    assert all(p["pano_id"][-1] != "1" for p in plan["panos"])  # Photosphere dropped.
    index = b.Index(root / "gsv_index")
    out = b.build(api, logs, fake_embed, index, budget, static_reserve=0)
    assert out["stopped"] == {"tiles": "completed", "static": "completed"}
    rows, desc = index.load()
    assert len(rows) == len(desc) > 0
    assert all(r["attribution"] == "Google Street View, © Google" and r["source"] == "gsv"
               for r in rows)
    assert {r["route"] for r in rows} <= {"tiles", "static"}
    by_pano = {}
    for r in rows:
        by_pano.setdefault(r["pano_id"], set()).add(r["route"])
    assert all(len(v) == 1 for v in by_pano.values())  # Never both routes.
    for r in rows:  # 4 headings on footways, 2 sideways on roads.
        n = sum(1 for x in rows if x["pano_id"] == r["pano_id"])
        assert n == (4 if r["way_type"] == "footway" else 2)
    # Resume: a second run sends no billable request for finished views.
    sent_before = dict(api.counts.sent)
    b.build(api, logs, fake_embed, index, budget, static_reserve=0)
    assert api.counts.sent["tile"] == sent_before["tile"]
    assert api.counts.sent["image"] == sent_before.get("image", 0)
    found = [p for p in root.rglob("*") if p.is_file() and (
        p.suffix.lower() in IMAGE_SUFFIXES or p.read_bytes()[:3] == b"\xff\xd8\xff")]
    assert found == []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl"):
            assert FAKE_KEY not in p.read_text()
    printed = capsys.readouterr()
    assert FAKE_KEY not in printed.out + printed.err
    log = [json.loads(x) for x in (root / "pull_log.jsonl").read_text().splitlines()]
    assert {r["request_type"] for r in log} >= {"session", "panoids", "metadata", "tile"}
    assert tuple(log[0]) == gsv.PULL_LOG_FIELDS
    tiles_logged = sum(r["request_count"] for r in log if r["request_type"] == "tile")
    assert tiles_logged == budget.tiles_total  # Every reserved tile was sent and logged.


def test_quota_error_stops_cleanly_without_leaking_the_key(env):
    root, budget = env
    client = FakeClient()
    api, logs = make_api(budget, client)
    api.create_session()
    b.discover(api, logs, b.ordered_points(points()))
    client.quota_after = len(client.calls) + 5
    out = b.build(api, logs, fake_embed, b.Index(root / "gsv_index"), budget,
                  routes=("tiles",), static_reserve=0)
    assert out["stopped"]["tiles"].startswith("QuotaExceeded")
    assert FAKE_KEY not in json.dumps(out)
    for p in Path(root).rglob("*.json*"):
        assert FAKE_KEY not in p.read_text()


def test_tile_cap_stops_before_a_panorama_that_does_not_fit(env, monkeypatch):
    root, budget = env
    client = FakeClient()
    api, logs = make_api(budget, client)
    api.create_session()
    b.discover(api, logs, b.ordered_points(points()))
    monkeypatch.setattr(gsv, "TILE_DAILY_CAP", 40)
    out = b.build(api, logs, fake_embed, b.Index(root / "gsv_index"), budget,
                  routes=("tiles",), static_reserve=0)
    assert budget.tiles_total <= 40
    assert out["stopped"]["tiles"] in ("cap", "completed")


def test_headings_rule():
    assert b.headings_for("footway", 10.0) == [10.0, 100.0, 190.0, 280.0]
    assert b.headings_for("path", None) == [0.0, 90.0, 180.0, 270.0]
    assert b.headings_for("residential", 10.0) == [100.0, 280.0]


def test_order_follows_owner_list_and_skips_tracks():
    pts = [{"lat": 58.40 + i * 0.001, "lng": 15.6, "priority": p, "highway": h, "gap": "empty",
            "bearing": 0.0}
           for i, (p, h) in enumerate([(1, "path"), (3, "residential"), (2, "service"),
                                       (1, "footway"), (2, "residential"), (1, "track")])]
    order = [(p["priority"], p["highway"]) for p in b.ordered_points(pts)]
    assert order == [(2, "residential"), (1, "footway"), (2, "service"), (1, "path"),
                     (3, "residential")]


def test_tiles_needed_matches_simulation():
    pano = {"image_w": 16384, "image_h": 8192, "tile_w": 512, "tile_h": 512, "heading": 0.0}
    assert len(b.tiles_needed(pano, [0.0, 90.0, 180.0, 270.0])) == 16
    assert len(b.tiles_needed(pano, [55.0, 235.0])) == 12  # Unaligned: 3 columns per view.
    assert len(b.tiles_needed(pano, [45.0, 225.0])) == 8  # Aligned with tile edges.
