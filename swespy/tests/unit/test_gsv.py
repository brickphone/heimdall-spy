"""Street View rules and gap planning (no network)."""

import asyncio
import json
from datetime import UTC, datetime

import numpy as np
import pytest

from eval import gsv_plan as plan
from swespy import gsv

M = 1 / 111_320  # Degrees latitude per metre.
FAKE_KEY = "AIzaFAKE_planted_key_0123456789abcdefghi"


def test_gap_classification_on_synthetic_gallery():
    refs = [
        {"lat": 58.41, "lng": 15.62, "capture_year": 2016},  # Old, at the old-only point.
        {"lat": 58.42, "lng": 15.62, "capture_year": 2021},  # Recent.
        {"lat": 58.43, "lng": 15.62, "capture_year": 2015},
        {"lat": 58.43 + 10 * M, "lng": 15.62, "capture_year": 2019},  # Old + new mix.
        {"lat": 58.44, "lng": 15.62, "capture_year": None},  # Unknown year.
    ]
    grid = plan.RefGrid(refs)
    at = lambda lat: plan.classify(grid.near(lat, 15.62, plan.GAP_RADIUS_M))  # noqa: E731
    assert at(58.40) == "empty"
    assert at(58.41 + 20 * M) == "old-only"  # 20 m from a 2016 reference only.
    assert at(58.41 + 30 * M) == "empty"  # 30 m: beyond the 25 m radius.
    assert at(58.42) == "covered"
    assert at(58.43) == "covered"  # One reference within 25 m is from 2019.
    assert at(58.44) == "covered"  # Unknown year never counts as old.


def test_priorities_and_queue_order_and_cell_dedup():
    assert plan.priority_of("empty", "footway") == 1
    assert plan.priority_of("empty", "cycleway") == 1
    assert plan.priority_of("empty", "residential") == 2
    assert plan.priority_of("old-only", "footway") == 3
    assert plan.priority_of("covered", "footway") is None
    pts = [
        {"lat": 58.41, "lng": 15.62, "gap": "old-only", "priority": 3},
        {"lat": 58.42, "lng": 15.62, "gap": "empty", "priority": 2},
        {"lat": 58.43, "lng": 15.62, "gap": "empty", "priority": 1},
        {"lat": 58.43 + 2 * M, "lng": 15.62, "gap": "empty", "priority": 1},  # Same cell.
        {"lat": 58.45, "lng": 15.62, "gap": "covered", "priority": None},
    ]
    queue = plan.metadata_queue(pts)
    assert [p["priority"] for p in queue] == [1, 2, 3]
    assert all(p["gap"] != "covered" for p in queue)


def test_metadata_dedup_by_pano_id_and_copyright_filter():
    panos = {}
    meta = {"status": "OK", "pano_id": "P1", "location": {"lat": 58.4, "lng": 15.6},
            "date": "2023-06", "copyright": "© Google"}
    p3 = {"priority": 3, "highway": "residential", "bearing": 10.0}
    p1 = {"priority": 1, "highway": "footway", "bearing": 45.0}
    assert plan.merge_panos(panos, p3, meta) == "new"
    assert plan.merge_panos(panos, p3, meta) == "duplicate"
    assert plan.merge_panos(panos, p1, meta) == "duplicate"  # Upgraded to priority 1.
    assert panos["P1"]["priority"] == 1 and panos["P1"]["headings"] == [45.0, 135.0, 225.0,
                                                                        315.0]
    user = {**meta, "pano_id": "P2", "copyright": "© Anna Svensson"}
    assert plan.merge_panos(panos, p1, user) == "not_google"
    assert plan.merge_panos(panos, p1, {"status": "ZERO_RESULTS"}) == "zero_results"
    assert set(panos) == {"P1"}
    assert gsv.is_google_captured("© 2024 Google")
    assert gsv.is_google_captured("From the Owner, Photo by: Google")  # Map Tiles API.
    assert not gsv.is_google_captured("From the Owner, Photo by: Anna Svensson")
    assert not gsv.is_google_captured("© Googleplex fan")
    assert not gsv.is_google_captured(None)


def test_headings_relative_to_way():
    assert plan.headings(30.0, 1) == [30.0, 120.0, 210.0, 300.0]
    assert plan.headings(350.0, 2) == [350.0, 80.0, 170.0, 260.0]
    assert plan.headings(30.0, 3) == [120.0, 300.0]  # Sideways only.
    assert plan.headings(None, 1) == [0.0, 90.0, 180.0, 270.0]


def test_sampling_every_20_m_with_bearing():
    north = [(58.41, 15.62), (58.41 + 100 * M, 15.62)]
    samples = plan.sample_line(north)
    assert len(samples) == 6  # 0, 20, ..., 100 m.
    assert all(abs(b) < 0.5 or abs(b - 360) < 0.5 for _, _, b in samples)
    east = plan.sample_line([(58.41, 15.62), (58.41, 15.62 + 0.001)])
    assert all(abs(b - 90) < 0.5 for _, _, b in east)


def test_monthly_cap_allows_exactly_one_more_at_9499(tmp_path):
    log = gsv.PullLog(tmp_path / "pull_log.jsonl")
    now = datetime(2026, 11, 15, 12, tzinfo=UTC)
    log.append(request_type="image", request_count=9_499, success_count=9_499, bbox=None,
               priority=1, now=now)
    log.append(request_type="metadata", request_count=50_000, success_count=50_000,
               bbox=None, priority=1, now=now)  # Metadata is not capped.
    assert log.remaining_images(now=now) == 1
    log.reserve(1, now=now)
    with pytest.raises(gsv.CapReached):
        log.reserve(2, now=now)
    log.append(request_type="image", request_count=1, success_count=1, bbox=None, priority=1,
               now=now)
    with pytest.raises(gsv.CapReached):
        log.reserve(1, now=now)
    next_month = datetime(2026, 12, 2, 12, tzinfo=UTC)
    assert log.remaining_images(now=next_month) == gsv.MONTHLY_IMAGE_CAP
    with pytest.raises(ValueError):
        log.reserve(1, "metadata", now=now)


def test_pull_log_schema_and_no_key_leak(tmp_path, monkeypatch):
    monkeypatch.setenv(gsv.KEY_ENV, FAKE_KEY)
    assert gsv.api_key() == FAKE_KEY
    log = gsv.PullLog(tmp_path / "pull_log.jsonl")
    row = log.append(request_type="metadata", request_count=10, success_count=9,
                     bbox=(15.5, 58.3, 15.7, 58.5), priority=2)
    assert tuple(row) == gsv.PULL_LOG_FIELDS
    text = (tmp_path / "pull_log.jsonl").read_text()
    assert FAKE_KEY not in text and "http" not in text
    assert gsv.redact(f"https://x/y?location=1,2&key={FAKE_KEY}", FAKE_KEY).endswith(
        "key=[key]")
    with pytest.raises(ValueError):
        log.append(request_type="pano", request_count=1, success_count=1, bbox=None,
                   priority=1)


def test_metadata_stage_never_leaks_the_key(tmp_path, monkeypatch, capsys):
    """Mocked API: the key reaches only the request, never the log, state or output."""
    import httpx

    monkeypatch.setenv(gsv.KEY_ENV, FAKE_KEY)
    points = [{"lat": 58.41 + i * 50 * M, "lng": 15.62, "bearing": 0.0, "highway": "footway",
               "way_id": 1, "gap": "empty", "priority": 1} for i in range(4)]
    points.append({**points[0], "lat": 58.5, "gap": "covered", "priority": None})
    (tmp_path / "sample_points.json").write_text(json.dumps({"points": points}))
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        assert request.url.params["key"] == FAKE_KEY
        lat = float(request.url.params["location"].split(",")[0])
        assert lat < 58.5, "a covered point reached Google"
        copyright_text = "© Google" if len(seen) % 2 else "© Someone"
        return httpx.Response(200, json={
            "status": "OK", "pano_id": f"P{len(seen)}", "date": "2022-05",
            "location": {"lat": lat, "lng": 15.62}, "copyright": copyright_text})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    asyncio.run(plan.metadata_stage(tmp_path))
    assert len(seen) == 4
    panos = json.loads((tmp_path / "panos.json").read_text())["panos"]
    assert len(panos) == 2  # User photospheres dropped.
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text()
    out = capsys.readouterr()
    assert FAKE_KEY not in out.out + out.err
    rows = gsv.PullLog(tmp_path / "pull_log.jsonl").rows()
    assert rows[0]["request_type"] == "metadata" and rows[0]["request_count"] == 4


def test_attribution_on_every_gsv_candidate():
    label = gsv.candidate_label({"capture_date": "2023-06"})
    assert label["attribution"] == "Google Street View, © Google"
    assert label["source"] == "gsv" and label["license_url"] is None
    assert "creativecommons" not in json.dumps(label).lower()


def test_float16_storage_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.standard_normal((3, 8448)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.save(tmp_path / "d.npy", v.astype(np.float16))
    back = np.load(tmp_path / "d.npy").astype(np.float32)
    assert back.dtype == np.float32 and back.shape == v.shape
    assert np.abs(back - v).max() < 1e-3
    assert np.allclose(back @ back.T, v @ v.T, atol=1e-4)
