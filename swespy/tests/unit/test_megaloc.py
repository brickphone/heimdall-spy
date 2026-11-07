"""Fixed-input and descriptor checks for MegaLoc, without downloading model weights."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from swespy.model import megaloc


@pytest.mark.parametrize("kind", ["zero", "nan", "wrong_dimension"])
def test_invalid_descriptors_rejected(kind: str) -> None:
    vectors = np.zeros((1, megaloc.DIM), dtype=np.float32)
    if kind == "nan":
        vectors[0, 0] = float("nan")
    elif kind == "wrong_dimension":
        vectors = np.ones((1, 768), dtype=np.float32)
    with pytest.raises(ValueError, match="descriptors"):
        megaloc.checked_vectors(vectors, 1)


def test_contiguous_adapter_preserves_convolution_inputs_and_results() -> None:
    import torch

    values = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    noncontiguous = values.transpose(-1, -2)
    conv = torch.nn.Conv2d(3, 4, kernel_size=1)
    (adapted,) = megaloc.contiguous_conv_inputs(conv, (noncontiguous,))
    assert adapted.is_contiguous()
    assert torch.equal(adapted, noncontiguous)
    torch.testing.assert_close(conv(adapted), conv(noncontiguous))


def test_cached_descriptors_resume_and_reject_changed_image(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (20, 20), "red").save(path)
    digest = megaloc.file_hash(path)
    embedder = object.__new__(megaloc.MegaLocEmbedder)
    embedder.cache = tmp_path / "cache"
    embedder.cache.mkdir()
    vector = np.zeros((1, megaloc.DIM), dtype=np.float32)
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
    folder = tmp_path / "assets" / megaloc.SOURCE_REVISION
    folder.mkdir(parents=True)
    for name in ("megaloc_model.py", "LICENSE", "README.md"):
        (folder / name).write_text("modified source")

    def forbidden(**kwargs):
        raise AssertionError("Source must be verified before downloading/loading weights")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    with pytest.raises(ValueError, match="source checksum"):
        megaloc.load_assets(tmp_path)
