"""Trainable geographic head over frozen descriptors, with explicit checkpoint inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def sequence_split(
    labels: list[str],
    sequences: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out entire captures while ensuring every output class has training examples."""
    if len(labels) != len(sequences) or not labels or any(not s for s in sequences):
        raise ValueError("Nonempty labels and capture sequences are required for every row")
    groups = sorted(set(sequences))
    if len(groups) < 3:
        raise ValueError("At least three capture sequences are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_end = max(1, min(len(groups) - 2, int(len(groups) * 0.7)))
    val_end = max(train_end + 1, min(len(groups) - 1, int(len(groups) * 0.85)))
    train, val, test = (
        set(groups[:train_end]),
        set(groups[train_end:val_end]),
        set(groups[val_end:]),
    )
    seen = {label for label, group in zip(labels, sequences, strict=True) if group in train}
    for label in sorted(set(labels) - seen):
        options = {
            group for target, group in zip(labels, sequences, strict=True) if target == label
        }
        if not options & train:
            group = min(options, key=lambda s: groups.index(s))
            train.add(group)
            val.discard(group)
            test.discard(group)
    if not val or not test:
        raise ValueError("Insufficient independent captures for closed-set validation/test")
    folds = tuple(
        np.array([i for i, s in enumerate(sequences) if s in fold], dtype=np.int64)
        for fold in (train, val, test)
    )
    return folds  # type: ignore[return-value]


def grid_label(easting: float, northing: float, size: int = 1000) -> str:
    return f"{int(easting // size)}:{int(northing // size)}"


class GeoHead:
    """CPU inference from an explicitly loaded, encoder-bound geographic checkpoint."""

    def __init__(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("format") != "frozen_descriptor_geo_head_v1":
            raise ValueError("Unsupported geographic checkpoint")
        self.checkpoint = checkpoint
        self.labels: list[str] = checkpoint["labels"]
        self.head = nn.Linear(checkpoint["dimension"], len(self.labels))
        self.head.load_state_dict(checkpoint["state_dict"], strict=True)
        self.head.eval()

    @classmethod
    def load(cls, path: Path, encoder_identity: dict[str, Any]) -> GeoHead:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["encoder_identity"] != encoder_identity:
            raise ValueError("Trained head and image encoder identities differ")
        return cls(saved)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if (
            features.ndim != 2
            or features.shape[1] != self.head.in_features
            or not np.isfinite(features).all()
        ):
            raise ValueError("Invalid features for trained geographic head")
        with torch.inference_mode():
            return torch.softmax(self.head(torch.from_numpy(features)), dim=1).numpy()

    def locations(self, features: np.ndarray, top_k: int = 5) -> list[list[dict[str, Any]]]:
        probs = self.probabilities(features)
        centers = self.checkpoint["centers"]
        return [
            [
                {
                    "cell": self.labels[i],
                    "lat": centers[i][0],
                    "lng": centers[i][1],
                    "score": float(row[i]),
                }
                for i in np.argsort(-row, kind="stable")[:top_k]
            ]
            for row in probs
        ]

    def rerank(
        self,
        features: np.ndarray,
        candidates: np.ndarray,
        similarities: np.ndarray,
        references: list[dict[str, Any]],
        weight: float | None = None,
    ) -> np.ndarray:
        weight = self.checkpoint["fusion_weight"] if weight is None else weight
        probabilities = self.probabilities(features)
        label_to_idx = {label: i for i, label in enumerate(self.labels)}
        classes = np.array(
            [
                label_to_idx[
                    grid_label(r["easting"], r["northing"], self.checkpoint["grid_size_m"])
                ]
                for r in references
            ]
        )
        scores = similarities + weight * np.log(
            np.maximum(probabilities[np.arange(len(features))[:, None], classes[candidates]], 1e-9)
        )
        order = np.argsort(-scores, axis=1, kind="stable")
        return np.take_along_axis(candidates, order, axis=1)


def fit_head(
    features: np.ndarray,
    targets: np.ndarray,
    folds: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: list[str],
    centers: list[list[float]],
    encoder_identity: dict[str, Any],
    seed: int = 42,
    epochs: int = 60,
) -> tuple[GeoHead, dict[str, Any]]:
    """Train a small head; select weights on Mapillary validation only, never public queries."""
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(targets, dtype=np.int64))
    train, val, test = [torch.from_numpy(a) for a in folds]
    model = nn.Linear(x.shape[1], len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    frequencies = torch.bincount(y[train], minlength=len(labels)).float()
    if torch.any(frequencies == 0):
        raise ValueError("Every output class must have training examples")
    loss_fn = nn.CrossEntropyLoss(weight=frequencies.rsqrt())
    best_score, best_epoch, best_state = (-1.0, float("-inf")), 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        ordered = train[torch.randperm(len(train))]
        for indices in ordered.split(256):
            optimizer.zero_grad()
            loss = loss_fn(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x[val])
            val_loss = float(nn.functional.cross_entropy(logits, y[val]))
            accuracy = float((logits.argmax(1) == y[val]).float().mean())
        history.append(
            {"epoch": epoch + 1, "validation_accuracy": accuracy, "validation_loss": val_loss}
        )
        score = (accuracy, -val_loss)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"Head epoch {epoch + 1}: validation accuracy {accuracy:.3f}", flush=True)
        if epoch + 1 - best_epoch >= 8:
            break
    assert best_state is not None
    checkpoint = {
        "format": "frozen_descriptor_geo_head_v1",
        "dimension": x.shape[1],
        "labels": labels,
        "centers": [[float(value) for value in center] for center in centers],
        "grid_size_m": 1000,
        "encoder_identity": encoder_identity,
        "state_dict": best_state,
        "fusion_weight": 0.0,
        "seed": seed,
        "best_epoch": best_epoch,
    }
    head = GeoHead(checkpoint)
    with torch.inference_mode():
        test_accuracy = float((head.head(x[test]).argmax(1) == y[test]).float().mean())
    return head, {
        "history": history,
        "best_epoch": best_epoch,
        "validation_accuracy": best_score[0],
        "internal_test_accuracy": test_accuracy,
        "fold_counts": [len(train), len(val), len(test)],
        "seed": seed,
        "optimizer": "AdamW lr0.03 weight_decay0.01; inverse-sqrt class weights; "
        "batch256; <=60 epochs; patience8",
    }
