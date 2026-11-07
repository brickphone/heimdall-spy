"""Pair tiers and sampling must not pretend adjacent captures are independent tests."""

import pytest

from eval.paired_views import balanced, cell, creator_key, proposals

pytest.importorskip("utm", reason="Optional benchmark geospatial dependency")


def item(
    uid, lat=58.41, lng=15.62, heading=0, sequence="capture-a", captured_at=1000, creator="la2"
):
    return {
        "uid": uid,
        "lat": lat,
        "lng": lng,
        "heading": heading,
        "sequence": sequence,
        "captured_at": captured_at,
        "creator": creator,
    }


def test_viewpoint_pairs_are_not_mislabelled_as_cross_capture():
    a = item("a")
    b = item("b", lat=58.4101, heading=40)
    pair = proposals([b, a])[0]
    assert pair["pair_type"] == "within_capture_viewpoint"
    b.update(sequence="capture-b", captured_at=3 * 86400000)
    pair = proposals([a, b])[0]
    assert pair["pair_type"] == "cross_day"
    assert proposals([a, b]) == proposals([b, a])
    b["heading"] = 180
    assert not proposals([a, b])  # Nearby cameras pointing apart are not automatic positives.


def test_pairs_do_not_reuse_images_or_exceed_the_place_cap():
    rows = [item(str(i), lat=58.41 + i * 0.00001, heading=i * 10) for i in range(10)]
    pairs = proposals(rows)
    ids = [p[k] for p in pairs for k in ("a", "b")]
    assert len(ids) == len(set(ids))
    assert len(pairs) <= 2


def test_holdout_sampling_caps_creators_and_spreads_spatial_cells():
    rows = [
        item("same1", creator="LA2"),
        item("same2", creator="la2", lat=58.4101),
        item("far1", creator="other", lng=15.65),
        item("far2", creator="third", lng=15.68),
    ]
    selected = balanced(rows, 3, 1)
    assert len(selected) == 3
    assert len({creator_key(r["creator"]) for r in selected}) == 3
    assert len({cell(r, 1000) for r in selected}) == 3
