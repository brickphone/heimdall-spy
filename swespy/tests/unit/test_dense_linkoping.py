"""Dense acquisition must reserve entire sequences and preserve multiple headings."""

import pytest

from eval.dense_linkoping import formatted_link, split_metadata


def item(image_id, sequence, x=100, y=100, heading=0):
    return {"image_id": str(image_id), "sequence": sequence, "easting": x, "northing": y,
            "heading": heading, "captured_at": 123456, "utm_zone": 33, "utm_letter": "V",
            "lat": 58.41, "lng": 15.62}


def test_capture_sequences_and_previous_queries_stay_out_of_gallery() -> None:
    rows = [item(i, f"new-{i}", x=i*10) for i in range(40)]
    rows += [item(100, "old-ref"), item(101, "old-query")]
    gallery, queries, reserved = split_metadata(rows, {"old-ref"}, {"old-query"}, 12)
    assert len(gallery) == 12
    assert not {r["sequence"] for r in gallery} & reserved
    assert "old-query" not in {r["sequence"] for r in gallery}
    assert not {r["sequence"] for r in queries} & {"old-ref", "old-query"}
    assert split_metadata(rows, {"old-ref"}, {"old-query"}, 12) == (gallery, queries, reserved)


def test_gallery_retains_distinct_headings_at_same_camera_position() -> None:
    rows = [item(i, "old-ref", heading=i*90) for i in range(4)]
    rows += [item(100+i, f"new-{i}") for i in range(12)]
    gallery, queries, _ = split_metadata(rows, {"old-ref"}, set(), 20)
    assert {r["heading"] for r in gallery} == {0, 90, 180, 270}
    assert all(r["sequence"] != "old-ref" for r in queries)


def test_formatted_image_cannot_replace_existing_file(tmp_path) -> None:
    original = tmp_path / "originals/123.jpg"
    original.parent.mkdir()
    original.write_bytes(b"placeholder for link-only test")
    row = item(123, "reference")
    relative = formatted_link(tmp_path, row, "database")
    assert (tmp_path / relative).samefile(original)
    assert formatted_link(tmp_path, row, "database") == relative
    (tmp_path / relative).unlink()
    (tmp_path / relative).write_bytes(b"user data")
    with pytest.raises(ValueError, match="existing formatted"):
        formatted_link(tmp_path, row, "database")
    assert (tmp_path / relative).read_bytes() == b"user data"
