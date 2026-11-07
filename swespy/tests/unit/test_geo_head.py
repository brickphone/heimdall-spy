"""Sequence isolation, checkpoint/encoder compatibility and actual trained-head inference."""


import numpy as np
import pytest
import torch

from swespy.model.geo_head import GeoHead, fit_head, sequence_split


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
    with pytest.raises(ValueError, match="capture sequences"):
        sequence_split(labels, [""] * len(labels))


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
