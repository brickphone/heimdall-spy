from pathlib import Path

import numpy as np
import torch
from PIL import Image

from swespy.config import Settings
from swespy.model.embedder import CLIPEmbedder


class FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def get_image_features(self, pixel_values):
        _ = pixel_values
        out = torch.ones((1, 768), dtype=torch.float32)
        out[0, 0] = 2.0
        return out


class FakeProcessor:
    def __call__(self, images, return_tensors="pt", padding=False):
        _ = (images, return_tensors, padding)
        return {"pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32)}


def test_embed_image_normalized_vector(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("swespy.model.embedder.CLIPModel.from_pretrained", lambda _name: FakeModel())
    monkeypatch.setattr(
        "swespy.model.embedder.CLIPProcessor.from_pretrained",
        lambda _name: FakeProcessor(),
    )

    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(image_path)

    embedder = CLIPEmbedder(Settings(mapillary_access_token="x"))
    vector = embedder.embed_image(image_path)

    assert vector.shape == (768,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
