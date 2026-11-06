"""Leakage and provenance checks for the automatic street sanity test."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval import mapillary_check as check


def test_missing_sequence_or_invalid_coordinates_are_rejected() -> None:
    raw = {
        "id": "123", "sequence": "capture", "creator": {"username": "author"},
        "captured_at": 12345, "geometry": {"coordinates": [15.62, 58.41]},
    }
    assert check.record(raw) is not None
    assert check.record({**raw, "sequence": None}) is None
    assert check.record({**raw, "geometry": {"coordinates": [float("nan"), 58.41]}}) is None
    assert check.record({**raw, "creator": {}}) is None


def test_separate_image_from_reference_sequence_is_still_excluded() -> None:
    item = {"image_id": "2", "sequence": "same-trip", "lat": 58.41, "lng": 15.62}
    bbox = (15.58, 58.39, 15.66, 58.43)
    assert not check.eligible(item, {"1"}, {"same-trip"}, bbox)
    assert not check.eligible({**item, "image_id": "1"}, {"1"}, set(), bbox)
    assert not check.eligible({**item, "lat": 59.0}, set(), set(), bbox)
    assert check.eligible(item, {"1"}, {"different-trip"}, bbox)


def test_reencoded_copy_is_detected_without_identical_bytes(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    first = tmp_path / "original.png"
    second = tmp_path / "reencoded.jpg"
    unrelated = tmp_path / "unrelated.png"
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(first)
    with Image.open(first) as image:
        image.save(second, quality=85)
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(unrelated)
    a, b = check.signature(first), check.signature(second)
    assert a["sha256"] != b["sha256"]
    assert check.duplicate(b, [a])
    assert not check.duplicate(check.signature(unrelated), [a])


def test_changed_index_fails_before_any_api_inference(tmp_path: Path, monkeypatch) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    path = photos / "123.jpg"
    Image.new("RGB", (20, 20), "red").save(path)
    item = {
        "filename": path.name, "image_id": "123", "sequence": "test-trip",
        "lat": 58.41, "lng": 15.62, **check.signature(path),
    }
    manifest = {
        "queries": [item], "reference_sequences": ["reference-trip"],
        "index_sha256": "frozen",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    closed = []

    class Store:
        def __init__(self, settings):
            self.client = self

        def close(self):
            closed.append(True)

    monkeypatch.setattr(check, "VectorStore", Store)
    monkeypatch.setattr(check, "index_fingerprint", lambda store: "changed")

    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject changed index before calling the API")

    monkeypatch.setattr(check.httpx, "AsyncClient", forbidden)
    monkeypatch.setenv("TESTING", "false")
    check.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="Reference index changed"):
            asyncio.run(check.evaluate(manifest_path))
        assert closed == [True]
    finally:
        check.get_settings.cache_clear()
