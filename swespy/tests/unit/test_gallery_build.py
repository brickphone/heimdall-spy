"""Staged, pipelined gallery build with mocked network and a fake encoder."""

import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from eval import densify, gallery_build

RNG = np.random.default_rng(5)


def jpeg(w, h):
    blocks = RNG.integers(0, 256, (h // 8, w // 8, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(blocks.repeat(8, 0).repeat(8, 1)).save(out, format="JPEG")
    return out.getvalue()


async def fake_get(client, url, *, params=None, headers=None, attempts=6):
    if "graph.mapillary.com" in url:
        data = [{"id": i, "thumb_1024_url": f"https://cdn/{i}.jpg",
                 "thumb_2048_url": f"https://cdn/{i}-pano.jpg"}
                for i in params["image_ids"].split(",")]
        return httpx.Response(200, json={"data": data}, request=httpx.Request("GET", url))
    size = (2048, 1024) if "pano" in url else (1024, 768)
    return httpx.Response(200, content=jpeg(*size), request=httpx.Request("GET", url))


class FakeEmbedder:
    identity = {"weights_sha256": "fake"}

    def __init__(self, cache: Path):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    def transform(self, image):
        return np.asarray(image.resize((8, 8)), dtype=np.float32).ravel()

    def batch_tensors(self, tensors):
        self.calls += len(tensors)
        v = np.ones((len(tensors), 8448), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def ref(i, year, pano=False):
    return {"source": "mapillary", "source_id": str(100 + i), "ref_id": f"mapillary-{100 + i}",
            "sequence": f"s{i}", "creator": "c", "captured_at": 1_600_000_000_000 + i,
            "capture_year": year, "lat": 58.40 + i * 0.002, "lng": 15.60 + i * 0.002,
            "location_source": "mapillary_computed", "heading": 30.0, "is_pano": pano,
            "license_url": densify.CC_BY_SA_4, "source_url": "u"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    plan = [ref(0, 2019), ref(1, 2015), ref(2, 2016, pano=True), ref(3, 2021), ref(4, 2014)]
    (tmp_path / "plan.json").write_text(json.dumps({"references": plan}))
    (tmp_path / "plan-report.json").write_text("{}")
    monkeypatch.setattr(densify, "get_with_backoff", fake_get)
    monkeypatch.setattr(densify, "legacy_images", lambda: {})
    monkeypatch.setattr(gallery_build, "CHUNK", 2)
    root_cache = tmp_path / "megaloc"
    import hashlib

    key = hashlib.sha256(json.dumps(FakeEmbedder.identity, sort_keys=True).encode()).hexdigest()
    embedder = FakeEmbedder(root_cache / "descriptors" / key)
    settings = type("S", (), {"megaloc_root": str(root_cache)})()
    monkeypatch.setattr(gallery_build, "get_settings", lambda: settings)
    return tmp_path, embedder


def test_stage_split():
    assert gallery_build.stage_of(ref(0, 2017)) == "historic"
    assert gallery_build.stage_of(ref(0, 2017, pano=True)) == "recent"
    assert gallery_build.stage_of(ref(0, 2018)) == "recent"
    assert gallery_build.stage_of({**ref(0, 2010), "source": "wikimedia_commons"}) == "recent"


def test_recent_then_historic_with_resume(setup):
    root, embedder = setup
    with pytest.raises(ValueError, match="recent"):
        gallery_build.build(root, "historic", "cpu", "float32", 4, lambda: embedder)
    recent = gallery_build.build(root, "recent", "cpu", "float32", 4, lambda: embedder)
    saved = json.loads(recent.read_text())
    ids = {r["ref_id"] for r in saved["references_in_index_order"]}
    assert "mapillary-100" in ids and "mapillary-103" in ids
    assert sum(r["ref_id"].startswith("mapillary-102-crop") for r in saved[
        "references_in_index_order"]) == 8  # 2016 panorama is in the recent stage.
    assert not ids & {"mapillary-101", "mapillary-104"}  # Pre-2018 frames wait.
    encoded = embedder.calls
    # Resume: an unencoded chunk is re-queued; already cached descriptors are not redone.
    marker = gallery_build.encoded_marker(root, "recent-0000")
    marker.unlink()
    gallery_build.build(root, "recent", "cpu", "float32", 4, lambda: embedder)
    assert marker.exists() and embedder.calls == encoded
    historic = json.loads(gallery_build.build(
        root, "historic", "cpu", "float32", 4, lambda: embedder).read_text())
    all_ids = {r["ref_id"] for r in historic["references_in_index_order"]}
    assert ids | {"mapillary-101", "mapillary-104"} == all_ids
    assert historic["stages_included"] == ["recent", "historic"]
    assert historic["build_parameters"]["thinning"]["distance_m"] == 10.0


def test_deferred_sealed_guard_is_applied_on_finalize(setup, monkeypatch):
    from eval import leakage
    from swespy.model.locator import Gallery

    root, embedder = setup
    index = gallery_build.build(root, "recent", "cpu", "float32", 4, lambda: embedder,
                                defer_sealed=True)
    saved = json.loads(index.read_text())
    assert saved["sealed_guard"] == "pending"
    with pytest.raises(ValueError, match="sealed"):
        Gallery(index)
    # Synthetic test manifest whose only photo is a byte-copy of one gallery reference.
    planted = saved["references_in_index_order"][0]
    manifest = root / "test-manifest.json"
    manifest.write_text(json.dumps({"fresh_commons_ids": ["commons-1"], "images": [{
        "image_id": "commons-1", "lat": 0.0, "lng": 0.0, "capture_date": "2001-01-01",
        "creator": "x", "sha256": planted["sha256"], "phash": None, "dhash": None}]}))
    real = leakage.guard_sealed
    monkeypatch.setattr(leakage, "guard_sealed", lambda refs: real(refs, manifest))
    result = gallery_build.finalize_sealed(root, "recent")
    assert result["excluded"] == 1 and result["eval_photo_itself"] == 1
    final = json.loads(index.read_text())
    assert final["sealed_guard"] == "applied_on_mac"
    assert planted["ref_id"] not in {r["ref_id"] for r in final["references_in_index_order"]}
    assert Gallery(index).refs  # Loadable once finalized.
