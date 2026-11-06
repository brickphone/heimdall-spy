"""Fixed-input and descriptor checks for the experiment, without downloading model weights."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval import megaloc_compare as compare


def comparison_files(tmp_path: Path) -> tuple[Path, Path]:
    photos = tmp_path / "photos"
    photos.mkdir()
    image = photos / "123.jpg"
    Image.new("RGB", (20, 20), "red").save(image)
    query = {
        "filename": image.name, "image_id": "123", "sequence": "query-sequence",
        "lat": 58.41, "lng": 15.62, "sha256": compare.file_hash(image),
    }
    manifest = {
        "queries": [query], "reference_sequences": ["reference-sequence"],
        "index_sha256": "frozen",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    clip = {
        "manifest_sha256": compare.file_hash(manifest_path), "index_sha256": "frozen",
        "top_k": 5, "hit_radius_m": 100, "queries": [query],
    }
    clip_path = tmp_path / "clip.json"
    clip_path.write_text(json.dumps(clip))
    return manifest_path, clip_path


@pytest.mark.parametrize("change", ["manifest", "index", "bytes", "coordinates", "top_k"])
def test_incomparable_clip_report_rejected(tmp_path: Path, change: str) -> None:
    manifest, path = comparison_files(tmp_path)
    compare.validate_comparison(manifest, path)
    clip = json.loads(path.read_text())
    if change == "manifest":
        clip["manifest_sha256"] = "different"
    elif change == "index":
        clip["index_sha256"] = "different"
    elif change == "bytes":
        clip["queries"][0]["sha256"] = "different"
    elif change == "coordinates":
        clip["queries"][0]["lat"] += 0.01
    else:
        clip["top_k"] = 10
    path.write_text(json.dumps(clip))
    with pytest.raises(ValueError):
        compare.validate_comparison(manifest, path)


@pytest.mark.parametrize("kind", ["zero", "nan", "wrong_dimension"])
def test_invalid_descriptors_rejected(kind: str) -> None:
    vectors = np.zeros((1, compare.DIM), dtype=np.float32)
    if kind == "nan":
        vectors[0, 0] = float("nan")
    elif kind == "wrong_dimension":
        vectors = np.ones((1, 768), dtype=np.float32)
    with pytest.raises(ValueError, match="descriptors"):
        compare.checked_vectors(vectors, 1)


def test_contiguous_adapter_preserves_convolution_inputs_and_results() -> None:
    import torch

    values = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    noncontiguous = values.transpose(-1, -2)
    conv = torch.nn.Conv2d(3, 4, kernel_size=1)
    (adapted,) = compare.contiguous_conv_inputs(conv, (noncontiguous,))
    assert adapted.is_contiguous()
    assert torch.equal(adapted, noncontiguous)
    torch.testing.assert_close(conv(adapted), conv(noncontiguous))


def test_cached_descriptors_resume_and_reject_changed_image(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (20, 20), "red").save(path)
    digest = compare.file_hash(path)
    embedder = object.__new__(compare.MegaLocEmbedder)
    embedder.cache = tmp_path / "cache"
    embedder.cache.mkdir()
    vector = np.zeros((1, compare.DIM), dtype=np.float32)
    vector[0, 0] = 1
    embedder.batch = lambda paths: vector.copy()
    assert np.array_equal(embedder.embed([path], [digest], 1), vector)

    def forbidden(paths):
        raise AssertionError("A valid descriptor should be loaded from cache")

    embedder.batch = forbidden
    assert np.array_equal(embedder.embed([path], [digest], 1), vector)
    path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="Image bytes changed"):
        embedder.embed([path], [digest], 1)


def test_modified_upstream_source_rejected_before_weight_loading(
    tmp_path: Path, monkeypatch,
) -> None:
    folder = tmp_path / "assets" / compare.SOURCE_REVISION
    folder.mkdir(parents=True)
    for name in ("megaloc_model.py", "LICENSE", "README.md"):
        (folder / name).write_text("modified source")

    def forbidden(**kwargs):
        raise AssertionError("Source must be verified before downloading/loading weights")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    with pytest.raises(ValueError, match="source checksum"):
        compare.load_assets(tmp_path)
