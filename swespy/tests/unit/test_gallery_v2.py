"""Part 4 units: panorama crops, thinning, leakage guard, quality rules, provenance."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval import densify, leakage, panorama, quality

DAY_MS = 86_400_000


def test_panorama_crop_headings_and_count():
    headings = panorama.crop_headings(350.0)
    assert len(headings) == panorama.CROP_COUNT == 8
    assert [o for o, _ in headings] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert [h for _, h in headings][:3] == [350.0, 35.0, 80.0]
    assert all(h is None for _, h in panorama.crop_headings(None, 4))


def test_panorama_crop_faces_the_right_direction():
    # Equirectangular image: left half red, right half blue; the centre column faces 0°.
    pano = np.zeros((256, 512, 3), dtype=np.uint8)
    pano[:, :256] = (255, 0, 0)
    pano[:, 256:] = (0, 0, 255)
    image = Image.fromarray(pano)
    crops = panorama.split(image, 0.0)
    assert len(crops) == 8
    east = np.asarray(crops[2][0])  # +90°: clockwise = right half = blue.
    west = np.asarray(crops[6][0])  # +270° = -90°: left half = red.
    assert east[..., 2].mean() > 200 and east[..., 0].mean() < 50
    assert west[..., 0].mean() > 200 and west[..., 2].mean() < 50
    assert crops[0][0].size == (panorama.CROP_SIZE, panorama.CROP_SIZE)


def frame(i, lat, lng, heading, seq="s", t=0, pano=False):
    return {"source_id": str(i), "sequence": seq, "captured_at": t + i, "lat": lat,
            "lng": lng, "heading": heading, "is_pano": pano}


def test_thinning_drops_close_similar_frames_only():
    step = 1 / 111_320  # About 1 m north.
    rows = [frame(0, 58.41, 15.62, 10), frame(1, 58.41 + step, 15.62, 15),  # 1 m, 5°: drop
            frame(2, 58.41 + 2 * step, 15.62, 90),  # 2 m but 80° turn: keep
            frame(3, 58.41 + 5 * step, 15.62, 10),  # 5 m, same heading: drop (10 m rule)
            frame(6, 58.41 + 12 * step, 15.62, 10),  # 12 m: keep
            frame(4, 58.41, 15.62, 10, seq="other"),  # Other sequence: keep
            frame(5, 58.41, 15.62, 10, pano=True)]  # Panorama: always keep
    kept = {r["source_id"] for r in densify.thin_sequences(rows)}
    assert kept == {"0", "2", "4", "5", "6"}
    assert densify.THIN_DISTANCE_M == 10.0 and densify.THIN_HEADING_DEG == 20.0


def write_manifest(path: Path, rows, key="queries", ids=None):
    data = {key: rows}
    if ids is not None:
        data["fresh_commons_ids"] = ids
    path.write_text(json.dumps(data))
    return path


def eval_photo(image_id="commons-7", day="2025-05-01", creator="Alice"):
    return {"image_id": image_id, "lat": 58.41, "lng": 15.62, "capture_date": day,
            "creator": creator, "sha256": "abc", "phash": 0, "dhash": 0}


def ref(ref_id, **kw):
    base = {"ref_id": ref_id, "source": "mapillary", "source_id": ref_id, "lat": 58.41,
            "lng": 15.62, "captured_at": 0, "creator": "bob"}
    return {**base, **kw}


def stockholm_ms(day, hour=12):
    from datetime import datetime

    return int(datetime.fromisoformat(f"{day}T{hour:02d}:00").replace(
        tzinfo=leakage.LOCAL).timestamp() * 1000)


def guard_cases():
    near = 30 / 111_320
    return [
        ref("same-photo", source="wikimedia_commons", source_id="7",
            captured_at=stockholm_ms("2020-01-01")),
        ref("near-same-day", lat=58.41 + near, captured_at=stockholm_ms("2025-05-01", 15)),
        ref("same-author-day", lat=58.50, creator="ALICE",
            captured_at=stockholm_ms("2025-05-01", 9)),
        ref("near-other-year", lat=58.41 + near, captured_at=stockholm_ms("2024-05-01")),
        ref("same-author-other-day", lat=58.50, creator="alice",
            captured_at=stockholm_ms("2025-05-03")),
        ref("far-same-day", lat=58.45, captured_at=stockholm_ms("2025-05-01")),
        ref("pixel-copy", lat=58.45, captured_at=stockholm_ms("2019-01-01"), sha256="zzz",
            phash=1, dhash=0),
    ]


def test_dev_guard_excludes_each_rule_and_keeps_others(tmp_path):
    manifest = write_manifest(tmp_path / "dev.json", [eval_photo()])
    kept, report = leakage.guard_dev(guard_cases(), manifest)
    assert {r["ref_id"] for r in kept} == {"near-other-year", "same-author-other-day",
                                           "far-same-day"}
    assert report["by_rule"] == {"eval_photo_itself": 2, "near_in_space_and_time": 1,
                                 "same_photographer_session": 1}
    assert {e["reference"] for e in report["exclusions"]} == {
        "same-photo", "near-same-day", "same-author-day", "pixel-copy"}


def test_sealed_guard_returns_only_counts(tmp_path):
    manifest = write_manifest(tmp_path / "test.json",
                              [eval_photo(), eval_photo("commons-99", creator="Zed")],
                              key="images", ids=["commons-7"])
    kept, counts = leakage.guard_sealed(guard_cases(), manifest)
    assert len(kept) == 3
    assert counts == {"excluded": 4, "eval_photo_itself": 2, "near_in_space_and_time": 1,
                      "same_photographer_session": 1}
    assert all(isinstance(v, int) for v in counts.values())  # No IDs or coordinates.


def test_quality_rules():
    from PIL import ImageFilter

    flat = Image.new("RGB", (400, 300), (120, 160, 220))  # Sky only.
    rng = np.random.default_rng(0)
    # Sharp but low-detail: two flat regions with one crisp horizon edge.
    horizon = np.full((300, 400, 3), 200, dtype=np.uint8)
    horizon[150:] = 60
    horizon[150:, :, 1] = 90
    blocks = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8).repeat(10, 0).repeat(10, 1)
    busy = Image.fromarray(blocks)
    assert quality.image_rule(quality.measures(flat)) == "sky_or_ground_only"
    assert quality.image_rule(quality.measures(busy)) is None
    assert quality.measures(Image.fromarray(horizon))["blur_effect"] < 0.47  # Sharp edge.
    blurred = busy.filter(ImageFilter.GaussianBlur(3))
    assert quality.image_rule(quality.measures(blurred)) == "blurry"
    assert quality.indoor_text("File:Domkyrkan interiör.jpg")
    assert not quality.indoor_text("File:Stora torget, Linköping.jpg")


def test_commons_time_precision():
    ms, precision = densify.commons_time("2024-06-01 14:30:00")
    assert precision == "exact" and leakage.local_day(ms) == "2024-06-01"
    assert densify.commons_time("2024-06-01")[1] == "day"
    assert densify.commons_time("unknown") is None


def test_mapillary_rows_prefer_computed_geometry(tmp_path):
    folder = tmp_path / "mapillary" / "entities"
    folder.mkdir(parents=True)
    entity = {"id": "1", "sequence": "s", "captured_at": 1_600_000_000_000,
              "creator": {"username": "u"}, "is_pano": False,
              "geometry": {"coordinates": [15.60, 58.40]}, "compass_angle": 10.0,
              "computed_geometry": {"coordinates": [15.61, 58.41]},
              "computed_compass_angle": 370.0}
    raw_only = {**entity, "id": "2", "computed_geometry": None}
    (folder / "a.json").write_text(json.dumps([entity, raw_only]))
    rows = {r["source_id"]: r for r in densify.mapillary_rows(tmp_path)}
    assert (rows["1"]["lat"], rows["1"]["location_source"], rows["1"]["heading"]) == \
        (58.41, "mapillary_computed", 10.0)
    assert (rows["2"]["lat"], rows["2"]["location_source"]) == (58.40, "mapillary_raw")
    assert rows["1"]["capture_year"] == 2020


@pytest.mark.parametrize("bad", ["license", "time"])
def test_commons_rows_reject_non_commercial_and_undated(bad):
    ext = {"LicenseUrl": "https://creativecommons.org/licenses/by-sa/4.0",
           "GPSLatitude": "58.41", "GPSLongitude": "15.62",
           "DateTimeOriginal": "2024-06-01 14:30", "Artist": "A"}
    if bad == "license":
        ext["LicenseUrl"] = "https://creativecommons.org/licenses/by-nc/2.0"
    else:
        ext["DateTimeOriginal"] = ""
    page = {"pageid": 5, "title": "File:x.jpg", "imageinfo": [{
        "mime": "image/jpeg", "descriptionurl": "u", "url": "u",
        "extmetadata": {k: {"value": v} for k, v in ext.items()}}]}
    row, reason = densify.commons_row(page)
    assert row is None and reason in ("license_not_commercial", "no_capture_time")


def test_near_duplicate_means_same_image_not_similar_view():
    m = 1 / 111_320
    kept = {"sha256": "a", "phash": 0, "dhash": 0, "source": "mapillary", "lat": 58.41,
            "lng": 15.62}
    look_alike = {**kept, "sha256": "b", "phash": 1}
    assert not quality.is_duplicate({**look_alike, "lat": 58.41 + 20 * m}, [kept])
    assert quality.is_duplicate({**look_alike, "lat": 58.41 + 3 * m}, [kept])
    assert quality.is_duplicate({**look_alike, "lat": 58.41 + 20 * m,
                                 "source": "wikimedia_commons"}, [kept])
    assert quality.is_duplicate({**kept, "lat": 58.41 + 25 * m}, [kept])  # Same bytes.
    assert not quality.is_duplicate({**look_alike, "phash": 2**40 - 1, "dhash": 2**40 - 1,
                                     "lat": 58.41 + 3 * m}, [kept])


def test_build_stages_end_to_end_with_mocked_network(tmp_path, monkeypatch):
    """download -> crops -> filter -> index on a tiny plan; no network, no model."""
    import asyncio
    import hashlib
    import io

    import httpx

    rng = np.random.default_rng(3)

    def jpeg(w, h):
        blocks = rng.integers(0, 256, (h // 8, w // 8, 3), dtype=np.uint8)
        out = io.BytesIO()
        Image.fromarray(blocks.repeat(8, 0).repeat(8, 1)).save(out, format="JPEG")
        return out.getvalue()

    base = {"captured_at": stockholm_ms("2019-06-01"), "capture_year": 2019,
            "license_url": densify.CC_BY_SA_4, "location_source": "mapillary_computed",
            "creator": "someone", "source_url": "u", "sequence": "s"}
    plan = [
        {**base, "source": "mapillary", "source_id": "11", "ref_id": "mapillary-11",
         "lat": 58.40, "lng": 15.60, "heading": 90.0, "is_pano": False},
        {**base, "source": "mapillary", "source_id": "12", "ref_id": "mapillary-12",
         "lat": 58.42, "lng": 15.64, "heading": 10.0, "is_pano": True},
        {**base, "source": "kartaview", "source_id": "k1", "ref_id": "kartaview-k1",
         "lat": 58.39, "lng": 15.58, "heading": None, "is_pano": False,
         "location_source": "kartaview_gps", "download_url": "https://cdn/k1.jpg"},
    ]
    (tmp_path / "plan.json").write_text(json.dumps({"references": plan}))
    (tmp_path / "plan-report.json").write_text("{}")

    async def fake_get(client, url, *, params=None, headers=None, attempts=6):
        if "graph.mapillary.com" in url:
            data = [{"id": i, "thumb_1024_url": f"https://cdn/{i}.jpg",
                     "thumb_2048_url": f"https://cdn/{i}-pano.jpg"}
                    for i in params["image_ids"].split(",")]
            return httpx.Response(200, json={"data": data}, request=httpx.Request("GET", url))
        size = (2048, 1024) if "pano" in url else (1024, 768)
        return httpx.Response(200, content=jpeg(*size), request=httpx.Request("GET", url))

    monkeypatch.setattr(densify, "get_with_backoff", fake_get)
    monkeypatch.setattr(densify, "legacy_images", lambda: {})
    asyncio.run(densify.download(tmp_path))
    with Image.open(densify.image_path(tmp_path, plan[0])) as im:
        assert max(im.size) == densify.IMAGE_MAX_SIDE  # Stored at 640 px.
    with Image.open(densify.image_path(tmp_path, plan[1])) as im:
        assert im.size == (2048, 1024)  # Panorama source kept for cropping.

    densify.crops(tmp_path)
    crops = json.loads((tmp_path / "pano-crops.json").read_text())["references"]
    assert len(crops) == 8 and crops[2]["heading"] == 100.0
    densify.filter_stage(tmp_path)
    kept = json.loads((tmp_path / "filtered.json").read_text())["references"]
    assert {r["ref_id"] for r in kept} >= {"mapillary-11", "kartaview-k1"}
    assert not any(r.get("is_pano") for r in kept)  # Only crops of the panorama remain.

    # Fake an encoded cache so the index stage can run without MegaLoc.
    identity = {"weights_sha256": "x"}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = tmp_path / "megaloc" / "descriptors" / key
    cache.mkdir(parents=True)
    for r in kept:
        np.save(cache / f"{r['sha256']}.npy", np.ones(8448, dtype=np.float32) / 8448**0.5)
    (tmp_path / "encoder-identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(densify, "get_settings",
                        lambda: type("S", (), {"megaloc_root": str(tmp_path / "megaloc")})())
    densify.index(tmp_path)
    saved = json.loads((tmp_path / "gallery-index.json").read_text())
    assert saved["build_parameters"]["thinning"]["distance_m"] == 10.0
    assert saved["build_parameters"]["image_storage"]["max_side_px"] == 640
    assert len(saved["references_in_index_order"]) == len(kept)
