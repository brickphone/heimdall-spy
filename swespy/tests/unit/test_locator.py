"""The shared inference path: polygon-restricted gallery search and ranked output."""

import json
from pathlib import Path

import numpy as np
import pytest

from swespy.model import locator as loc
from swespy.model.megaloc import MODEL_IDENTITY_KEYS, file_hash

IDENTITY = {key: "pinned" for key in MODEL_IDENTITY_KEYS}


def gallery_files(tmp_path: Path) -> Path:
    import faiss

    refs = [
        {"image_id": "centre", "lat": 58.4108, "lng": 15.6214},
        {"image_id": "ryd", "lat": 58.4080, "lng": 15.5650},
        {"image_id": "outside", "lat": 58.30, "lng": 15.62},  # South of the polygon.
    ]
    vectors = np.eye(3, 4, dtype=np.float32)
    index = faiss.IndexFlatL2(4)
    index.add(vectors)
    faiss.write_index(index, str(tmp_path / "gallery.faiss"))
    manifest = tmp_path / "gallery-index.json"
    manifest.write_text(json.dumps({
        "index_sha256": file_hash(tmp_path / "gallery.faiss"), "dimension": 4,
        "references_in_index_order": refs, "model_identity": {**IDENTITY, "device": "mps"},
    }))
    return manifest


class FakeEmbedder:
    identity = {**IDENTITY, "device": "cpu"}  # Device may differ; descriptor space may not.

    def embed_image(self, image) -> np.ndarray:
        return np.array([0, 1, 0, 0], dtype=np.float32)

    def batch_images(self, images) -> np.ndarray:
        return np.stack([self.embed_image(image) for image in images])


def test_gallery_excludes_references_outside_project_area(tmp_path: Path) -> None:
    gallery = loc.Gallery(gallery_files(tmp_path))
    assert [r["image_id"] for r in gallery.refs] == ["centre", "ryd"]
    assert gallery.excluded_outside_area == 1
    # The excluded reference's vector would be the perfect match; it must not come back.
    indices, sims = gallery.search(np.array([0, 0, 1, 0], dtype=np.float32), 5)
    assert indices.shape == (1, 2)


def test_locator_ranks_and_reports_top_candidates(tmp_path: Path) -> None:
    locator = loc.GeoLocator(FakeEmbedder(), loc.Gallery(gallery_files(tmp_path)), top_k=2)
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    result = locator.locate_bytes(buf.getvalue())
    assert (result["predicted_lat"], result["predicted_lng"]) == (58.4080, 15.5650)
    assert [c["image_id"] for c in result["candidates"]] == ["ryd", "centre"]
    assert result["candidates"][0]["similarity"] == pytest.approx(1.0)
    assert result["candidates"][1]["similarity"] == pytest.approx(0.0)
    assert result["ranked_indices"] == [1, 0]


def test_locator_rejects_a_different_descriptor_space(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    embedder.identity = {**IDENTITY, "weights_sha256": "other"}
    with pytest.raises(ValueError, match="different MegaLoc assets"):
        loc.GeoLocator(embedder, loc.Gallery(gallery_files(tmp_path)))


def test_changed_index_is_rejected(tmp_path: Path) -> None:
    manifest = gallery_files(tmp_path)
    (tmp_path / "gallery.faiss").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        loc.Gallery(manifest)


def test_descriptor_cache_gallery_format(tmp_path: Path) -> None:
    import hashlib

    folder = tmp_path / "descriptors"
    folder.mkdir()
    refs = []
    for i, (lat, lng) in enumerate([(58.4108, 15.6214), (58.4080, 15.5650), (58.30, 15.62)]):
        vector = np.zeros(4, dtype=np.float32)
        vector[i] = 1
        np.save(folder / f"h{i}.npy", vector)
        refs.append({"image_id": f"r{i}", "lat": lat, "lng": lng, "sha256": f"h{i}"})
    manifest = tmp_path / "gallery-index.json"
    manifest.write_text(json.dumps({
        "format": "descriptor_cache_v1", "dimension": 4, "descriptor_dir": str(folder),
        "references_in_index_order": refs, "model_identity": IDENTITY,
        "index_sha256": hashlib.sha256("\n".join(r["sha256"] for r in refs).encode()).hexdigest(),
    }))
    gallery = loc.Gallery(manifest)
    assert [r["image_id"] for r in gallery.refs] == ["r0", "r1"]
    indices, sims = gallery.search(np.array([0, 1, 0, 0], dtype=np.float32), 2)
    assert indices[0].tolist() == [1, 0] and sims[0][0] == pytest.approx(1.0)
    refs[0]["sha256"] = "tampered"
    manifest.write_text(manifest.read_text().replace('"h0"', '"tampered"'))
    with pytest.raises(ValueError, match="checksum"):
        loc.Gallery(manifest)


def test_locator_reranks_with_geometric_verification(tmp_path: Path) -> None:
    from io import BytesIO

    from PIL import Image

    class FakeVerifier:
        def verify_iter(self, image, references):
            # Only the second-ranked reference ("centre") shows the same scene.
            for key, path in references:
                yield {"inliers": 120 if "h0" in str(path) or key == "h0" else 0, "matches": 150}

        def verify_multi(self, images, pairs):
            yield from self.verify_iter(images[0], [(key, ref) for _, key, ref in pairs])

        def verify(self, image, references):
            return list(self.verify_iter(image, references))

    manifest = gallery_files(tmp_path)
    saved = json.loads(manifest.read_text())
    for i, r in enumerate(saved["references_in_index_order"]):
        r.update({"sha256": f"h{i}", "path": f"img{i}.jpg"})
    manifest.write_text(json.dumps(saved))
    locator = loc.GeoLocator(FakeEmbedder(), loc.Gallery(manifest), top_k=2, depth=2,
                             verifier=FakeVerifier(), weight=0.5, min_inliers=15)
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    result = locator.locate_bytes(buf.getvalue())
    assert [c["image_id"] for c in result["candidates"]] == ["centre", "ryd"]
    assert result["candidates"][0]["inliers"] == 120
    assert result["confidence_state"] == "ok"
    assert (result["predicted_lat"], result["predicted_lng"]) == (58.4108, 15.6214)


def test_mmap_gallery_equals_in_memory_index(tmp_path: Path) -> None:
    import hashlib

    rng = np.random.default_rng(1)
    folder = tmp_path / "descriptors"
    folder.mkdir()
    refs = []
    for i in range(40):
        vector = rng.normal(size=16).astype(np.float32)
        np.save(folder / f"h{i}.npy", vector / np.linalg.norm(vector))
        lat = 58.30 if i % 7 == 0 else 58.4108  # Every 7th lies outside the polygon.
        refs.append({"image_id": f"r{i}", "lat": lat, "lng": 15.6214, "sha256": f"h{i}"})
    manifest = tmp_path / "gallery-index.json"
    manifest.write_text(json.dumps({
        "format": "descriptor_cache_v1", "dimension": 16, "descriptor_dir": str(folder),
        "references_in_index_order": refs, "model_identity": IDENTITY,
        "index_sha256": hashlib.sha256("\n".join(r["sha256"] for r in refs).encode()).hexdigest(),
    }))
    mapped = loc.Gallery(manifest)
    flat = loc.Gallery(manifest, mmap=False)
    assert isinstance(mapped.vectors, np.memmap) and mapped.index is None
    queries = rng.normal(size=(5, 16)).astype(np.float32)
    for got, want in zip(mapped.search(queries, 10), flat.search(queries, 10), strict=True):
        np.testing.assert_array_equal(got, want)
    # A different polygon filter makes the matrix stale; it is rebuilt, not reused.
    path, sidecar = loc.vectors_file(manifest)
    key = json.loads(sidecar.read_text())["key"]
    wide = ((15.0, 58.0), (16.0, 58.0), (16.0, 59.0), (15.0, 59.0))
    assert len(loc.Gallery(manifest, wide).refs) == 40
    assert json.loads(sidecar.read_text())["key"] != key
