"""Provenance and duplicate-screening helpers shared by acquisition scripts."""

from pathlib import Path

import numpy as np
from PIL import Image

from eval import common


def test_missing_sequence_or_invalid_coordinates_are_rejected() -> None:
    raw = {
        "id": "123", "sequence": "capture", "creator": {"username": "author"},
        "captured_at": 12345, "geometry": {"coordinates": [15.62, 58.41]},
    }
    assert common.record(raw) is not None
    assert common.record({**raw, "sequence": None}) is None
    assert common.record({**raw, "geometry": {"coordinates": [float("nan"), 58.41]}}) is None
    assert common.record({**raw, "creator": {}}) is None


def test_reencoded_copy_is_detected_without_identical_bytes(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    first = tmp_path / "original.png"
    second = tmp_path / "reencoded.jpg"
    unrelated = tmp_path / "unrelated.png"
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(first)
    with Image.open(first) as image:
        image.save(second, quality=85)
    Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(unrelated)
    a, b = common.signature(first), common.signature(second)
    assert a["sha256"] != b["sha256"]
    assert common.duplicate(b, [a])
    assert not common.duplicate(common.signature(unrelated), [a])
