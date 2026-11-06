"""City acquisition must protect held-out sequences and balance crowded areas."""

import pytest

from eval.dense_linkoping import normalized
from eval.expand_linkoping import BBOX, select, tiles


def row(image_id, sequence="reference", x=100, heading=0, captured_at=100):
    return {"image_id": str(image_id), "sequence": sequence, "easting": x,
            "northing": 100, "heading": heading, "captured_at": captured_at}


def test_city_selection_excludes_test_sequences_and_existing_ids():
    rows = [row(1, "held-out"), row(2), row(3, x=200), row(4, x=300)]
    chosen = select(rows, {"held-out"}, {"2"}, 10)
    assert {r["image_id"] for r in chosen} == {"3", "4"}


def test_city_selection_balances_locations_and_keeps_newest_view():
    crowded = [row(i, x=i*20) for i in range(30)]
    rows = crowded + [row(100, x=2000), row(101, x=4000)]
    chosen = select(rows, set(), set(), 3)
    assert {int(r["easting"]//1000) for r in chosen} == {0, 2, 4}
    chosen = select([row(1), row(2, captured_at=200), row(3, heading=90)],
                    set(), set(), 10)
    assert {r["image_id"] for r in chosen} == {"2", "3"}
    assert chosen == select(list(reversed([row(1), row(2, captured_at=200),
                                          row(3, heading=90)])), set(), set(), 10)


def test_expanded_bounds_accept_suburbs_without_changing_central_default():
    pytest.importorskip("utm", reason="Optional benchmark dependency")
    raw = {"id": "123", "sequence": "capture", "creator": {"username": "author"},
           "geometry": {"coordinates": [15.55, 58.40]}, "captured_at": 100,
           "is_pano": False, "thumb_1024_url": "https://example.org/photo.jpg"}
    assert normalized(raw) is None
    assert normalized(raw, BBOX) is not None
    boxes = tiles()
    assert len(boxes) == 80
    assert boxes[0][:2] == BBOX[:2]
    assert boxes[-1][2:] == BBOX[2:]
