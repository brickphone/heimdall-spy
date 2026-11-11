"""Automatic search variants: generation rules, merge and the manual-region tie rule."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from swespy.model import variants as va


def street(width: int = 640, height: int = 480) -> Image.Image:
    """Textured image (passes the sky/ground rule everywhere)."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def person_on_right(width: int = 640, height: int = 480, share: float = 0.5) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, round(width * (1 - share)):] = True
    return mask


def test_whole_mode_is_only_the_photo() -> None:
    variants = va.make_variants(street(), person_on_right(), mode="whole")
    assert [v.id for v in variants] == ["whole"]


def test_no_people_skips_the_masked_variant() -> None:
    variants = va.make_variants(street(), np.zeros((480, 640), dtype=bool))
    ids = [v.id for v in variants]
    assert "masked" not in ids
    # Without people the largest clean rectangle is the whole photo, already variant (a).
    assert "crop:clean" not in ids
    assert ids == ["whole", "crop:left", "crop:center", "crop:right", "crop:band"]


def test_full_mode_crop_count_size_and_regions() -> None:
    variants = va.make_variants(street(), person_on_right())
    ids = [v.id for v in variants]
    assert ids[:2] == ["whole", "masked"]
    crops = [v for v in variants if v.id.startswith("crop:")]
    assert 1 <= len(crops) <= va.MAX_CROPS
    for v in crops:
        assert va.region_area(v.region) >= va.CROP_MIN_AREA
        w, h = v.image.size
        assert w == pytest.approx(640 * v.region["width"], abs=1)
        assert h == pytest.approx(480 * v.region["height"], abs=1)
    clean = next(v for v in crops if v.id == "crop:clean")
    assert clean.region["x"] + clean.region["width"] <= 0.5 + 1e-9  # Left of the person.
    assert va.person_fraction(person_on_right(), clean.region) < va.CLEAN_MAX_PERSON


def test_masked_variant_fills_people_with_the_mean_colour() -> None:
    image = street()
    mask = person_on_right()
    masked = np.asarray(va.make_variants(image, mask)[1].image)
    mean = np.asarray(image).reshape(-1, 3).mean(0).round()
    assert (masked[mask] == mean).all()
    assert (masked[~mask] == np.asarray(image)[~mask]).all()


def test_sky_or_ground_crops_are_skipped() -> None:
    # A flat left strip (like sky or bare ground) is dropped; textured crops are kept.
    image = street()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, round(640 * va.STRIP_WIDTH), 480), fill=(150, 180, 220))
    ids = [v.id for v in va.make_variants(image, np.zeros((480, 640), dtype=bool))]
    assert "crop:left" not in ids
    assert "crop:right" in ids


def test_no_clean_rectangle_when_people_fill_the_photo() -> None:
    assert va.largest_clean_rect(np.ones((100, 100), dtype=bool)) is None


def test_manual_region_is_variant_a_and_automatic_variants_still_run() -> None:
    region = {"x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0}
    variants = va.make_variants(street(), person_on_right(), region)
    assert variants[0].id == "manual" and variants[0].region == region
    assert variants[0].image.size == (320, 480)
    assert any(v.id.startswith("crop:") for v in variants)


def test_merge_keeps_each_references_best_variant() -> None:
    merged = va.merge([([1, 2], [0.5, 0.4]), ([2, 3], [0.7, 0.3])], [0, 1])
    assert merged == [(2, 0.7, 1), (1, 0.5, 0), (3, 0.3, 1)]


def test_manual_region_wins_ties() -> None:
    # The same references at the same similarity from a crop (listed first) and the
    # manual region: the manual region keeps them.
    merged = va.merge([([7, 8], [0.6, 0.6]), ([8, 7], [0.6, 0.6])], [2, 0])
    assert sorted(merged) == [(7, 0.6, 1), (8, 0.6, 1)]
    # Ordering ties between references go to the manual region's reference.
    merged = va.merge([([5], [0.6]), ([6], [0.6])], [2, 0])
    assert merged == [(6, 0.6, 1), (5, 0.6, 0)]


def test_analyze_suggests_a_region_only_above_the_threshold() -> None:
    out = va.analyze(street(), person_on_right(share=0.5))
    assert out["suggest"] and out["person_share"] == pytest.approx(0.5)
    assert out["suggested_region"]["x"] == 0.0
    few = va.analyze(street(), person_on_right(share=0.1))
    assert not few["suggest"]
