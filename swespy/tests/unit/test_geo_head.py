"""Sequence isolation, checkpoint/encoder compatibility and actual trained-head inference."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from swespy.model.geo_head import GeoHead, fit_head, sequence_split
from swespy.model.trainer import GeoClassifier, Trainer


def test_sequence_split_keeps_classes_known_and_capture_groups_disjoint():
    minimal = sequence_split(["a"] * 3, ["one", "two", "three"])
    assert [len(fold) for fold in minimal] == [1, 1, 1]
    labels = ["a", "b"] * 12 + ["rare"]
    sequences = [f"capture-{i // 2}" for i in range(24)] + ["only-rare"]
    folds = sequence_split(labels, sequences)
    assert set(np.concatenate(folds)) == set(range(len(labels)))
    groups = [{sequences[i] for i in f} for f in folds]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    training_labels = {labels[i] for i in folds[0]}
    assert {labels[i] for i in folds[1]} <= training_labels
    assert {labels[i] for i in folds[2]} <= training_labels
    assert "only-rare" in groups[0]
    frame = pd.DataFrame({"s2_cell": labels, "sequence": sequences})
    train, val, test = Trainer.split_by_cell(frame)
    assert set(val.s2_cell) <= set(train.s2_cell) and set(test.s2_cell) <= set(train.s2_cell)
    with pytest.raises(ValueError, match="capture sequences"):
        Trainer.split_by_cell(frame.drop(columns="sequence"))


def test_trained_descriptor_head_safe_roundtrip_and_encoder_mismatch(tmp_path):
    labels = ["a", "b"] * 12
    features = np.array([[1.0, 0.0], [0.0, 1.0]] * 12, dtype=np.float32)
    sequences = [f"capture-{i // 2}" for i in range(24)]
    folds = sequence_split(labels, sequences)
    targets = np.array([0, 1] * 12)
    head, report = fit_head(
        features,
        targets,
        folds,
        ["a", "b"],
        [[np.float64(58.4), np.float64(15.6)], [np.float64(58.5), np.float64(15.7)]],
        {"encoder": "fixture"},
        epochs=10,
    )
    path = tmp_path / "head.pt"
    torch.save(head.checkpoint, path)
    restored = GeoHead.load(path, {"encoder": "fixture"})
    np.testing.assert_array_equal(head.probabilities(features), restored.probabilities(features))
    assert report["validation_accuracy"] == 1
    assert restored.locations(features[:2], 1)[0][0]["cell"] == "a"
    with pytest.raises(ValueError, match="identities differ"):
        GeoHead.load(path, {"encoder": "different"})


def test_legacy_clip_checkpoint_is_loaded_and_backbone_stays_frozen(tmp_path, monkeypatch):
    class FakeCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = nn.Linear(2, 2, bias=False)
            self.config = SimpleNamespace(projection_dim=2)
            with torch.no_grad():
                self.vision_model.weight.copy_(torch.eye(2))

        def get_image_features(self, pixel_values):
            return self.vision_model(pixel_values)

    monkeypatch.setattr("swespy.model.trainer.CLIPModel.from_pretrained", lambda _: FakeCLIP())
    model = GeoClassifier("fixture", 2)
    model.train()
    assert not model.clip.training
    assert all(not p.requires_grad for p in model.clip.parameters())
    x = torch.tensor([[1.0, 2.0]])
    model(x).sum().backward()
    assert model.head.weight.grad is not None and model.clip.vision_model.weight.grad is None
    path = tmp_path / "legacy-head.pt"
    torch.save(
        {
            "format": "clip_cell_classifier_v1",
            "head_state": model.head.state_dict(),
            "clip_model_name": "fixture",
            "label_to_idx": {"a": 0, "b": 1},
        },
        path,
    )
    loaded = GeoClassifier.load(path)
    torch.testing.assert_close(model(x), loaded.model(x))
    assert loaded.label_to_idx == {"a": 0, "b": 1}
