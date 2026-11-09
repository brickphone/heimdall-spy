"""Cross-view feasibility: viewpoint rules, polygon separation and eval-radius guards."""

import json

import pytest

from eval import crossview, leakage
from swespy.config import LINKOPING_AREA, TRAINING_CITIES, polygons_overlap

M = 1 / 111_320


def track(speed_mps, n=12, dt_s=2.0, pano=False):
    return [{"captured_at": int(i * dt_s * 1000), "lat": 59.0 + i * speed_mps * dt_s * M,
             "lng": 16.0, "is_pano": pano} for i in range(n)]


@pytest.mark.parametrize(("speed", "expected"), [
    (14.0, "car"),        # ~50 km/h
    (8.0, "car"),         # ~29 km/h
    (5.0, "bike"),        # ~18 km/h
    (1.3, "pedestrian"),  # walking
])
def test_speed_rules(speed, expected):
    assert crossview.classify(track(speed))["class"] == expected


def test_car_by_peaks_even_with_slow_median():
    frames = track(3.0, n=20)
    # Four fast stretches push the 90th percentile above 12 m/s.
    for i in range(16, 20):
        frames[i]["lat"] = frames[15]["lat"] + (i - 15) * 30 * M
    assert crossview.classify(frames)["class"] == "car"


def test_panorama_and_unknown_rules():
    assert crossview.classify(track(14.0, pano=True))["class"] == "panorama"
    assert crossview.classify(track(1.3), {"camera_type": "spherical"})["class"] == "panorama"
    assert crossview.classify(track(1.3, n=4))["class"] == "unknown"
    gappy = track(1.3, n=8, dt_s=120)  # Every step is a > 60 s gap.
    assert crossview.classify(gappy)["class"] == "unknown"


def test_phone_make_is_recorded_for_pedestrians():
    out = crossview.classify(track(1.0), {"make": "Apple", "model": "iPhone 13"})
    assert out["class"] == "pedestrian" and "phone make" in out["rule"]


def test_training_polygons_do_not_overlap_linkoping():
    assert set(TRAINING_CITIES) == {"Norrköping", "Uppsala", "Örebro", "Västerås", "Umeå"}
    for city in TRAINING_CITIES.values():
        assert not polygons_overlap(city["polygon"], LINKOPING_AREA)
    square = ((15.60, 58.40), (15.64, 58.40), (15.64, 58.42), (15.60, 58.42))
    assert polygons_overlap(square, LINKOPING_AREA)  # The check does detect overlap.
    assert crossview.outside_linkoping(58.59, 16.18)  # Norrköping
    assert not crossview.outside_linkoping(58.41, 15.62)  # Linköping centre


def test_one_km_guard_with_sealed_test_counts(tmp_path, monkeypatch):
    dev = {"queries": [{"image_id": "commons-1", "lat": 58.59, "lng": 16.18,
                        "capture_date": "2025-01-01", "creator": "a"}]}
    test = {"fresh_commons_ids": ["commons-2"],
            "images": [{"image_id": "commons-2", "lat": 59.86, "lng": 17.64,
                        "capture_date": "2025-01-01", "creator": "b"}]}
    (tmp_path / "dev.json").write_text(json.dumps(dev))
    (tmp_path / "test.json").write_text(json.dumps(test))
    monkeypatch.setattr(leakage, "DEV_MANIFEST", tmp_path / "dev.json")
    monkeypatch.setattr(leakage, "TEST_MANIFEST", tmp_path / "test.json")
    rows = [{"id": "near-dev", "lat": 58.59 + 500 * M, "lng": 16.18},
            {"id": "near-test", "lat": 59.86 - 900 * M, "lng": 17.64},
            {"id": "far", "lat": 59.27, "lng": 15.21}]
    kept, report = crossview.guard_eval_radius(rows)
    assert [r["id"] for r in kept] == ["far"]
    assert report == {"dev_excluded": 1, "test_sealed_excluded_count_only": 1}
