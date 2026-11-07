"""Fine-tuning training logic for geolocation classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from swespy.config import Settings
from swespy.model.geo_head import sequence_split


class GeoDataset(Dataset):
    """Dataset for image-to-cell classification."""

    def __init__(
        self, df: pd.DataFrame, processor: CLIPProcessor, label_to_idx: dict[str, int]
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        image = Image.open(row["local_path"]).convert("RGB")
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        label = self.label_to_idx[row["s2_cell"]]
        return pixel_values, label


class GeoClassifier(nn.Module):
    """CLIP vision encoder with linear classification head."""

    def __init__(self, clip_model_name: str, num_cells: int) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.head = nn.Linear(self.clip.config.projection_dim, num_cells)

    def train(self, mode: bool = True) -> GeoClassifier:
        super().train(mode)
        self.clip.eval()  # Frozen features must not change with training/dropout mode.
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Compute logits for S2 classes."""
        with torch.no_grad():
            features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.head(features)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> TrainArtifacts:
        """Explicit inference path for the trained head, using its recorded class order."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != "clip_cell_classifier_v1":
            raise ValueError(
                "Unsupported classifier checkpoint; retrain with the repaired contract"
            )
        labels = saved["label_to_idx"]
        if sorted(labels.values()) != list(range(len(labels))):
            raise ValueError("Invalid classifier class mapping")
        model = cls(saved["clip_model_name"], len(labels)).to(device)
        model.head.load_state_dict(saved["head_state"], strict=True)
        model.eval()
        return TrainArtifacts(model, labels)


@dataclass
class TrainArtifacts:
    model: GeoClassifier
    label_to_idx: dict[str, int]


class Trainer:
    """Trainer for CLIP-based geolocation classification."""

    def __init__(self, settings: Settings, project_root: Path | None = None) -> None:
        self.settings = settings
        self.project_root = project_root or Path.cwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def split_by_cell(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Keep output cells known while holding out complete capture sequences.

        Historical method name retained for callers. Splitting output classes
        themselves made validation labels impossible to resolve.
        """
        if (
            not {"s2_cell", "sequence"}.issubset(df.columns)
            or df[["s2_cell", "sequence"]].isna().any().any()
        ):
            raise ValueError(
                "Training metadata requires S2 cells and capture sequences; "
                "do not substitute a random frame split"
            )
        folds = sequence_split(
            df["s2_cell"].astype(str).tolist(), df["sequence"].astype(str).tolist()
        )
        return tuple(df.iloc[indices].copy() for indices in folds)  # type: ignore[return-value]

    def _make_loader(
        self,
        df: pd.DataFrame,
        processor: CLIPProcessor,
        label_to_idx: dict[str, int],
        shuffle: bool,
    ) -> DataLoader:
        dataset = GeoDataset(df=df, processor=processor, label_to_idx=label_to_idx)
        return DataLoader(
            dataset, batch_size=self.settings.batch_size, shuffle=shuffle, num_workers=0
        )

    def train(self, epochs: int = 5) -> TrainArtifacts:
        """Train and persist the best model checkpoint."""
        meta_path = self.project_root / "data" / "metadata" / "images.parquet"
        df = pd.read_parquet(meta_path)
        train_df, val_df, _test_df = self.split_by_cell(df)
        torch.manual_seed(42)

        labels = sorted(train_df["s2_cell"].unique().tolist())
        label_to_idx = {label: idx for idx, label in enumerate(labels)}

        processor = CLIPProcessor.from_pretrained(self.settings.clip_model_name)
        model = GeoClassifier(self.settings.clip_model_name, num_cells=len(label_to_idx)).to(
            self.device
        )

        train_loader = self._make_loader(train_df, processor, label_to_idx, shuffle=True)
        val_loader = self._make_loader(val_df, processor, label_to_idx, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.head.parameters(),
            lr=self.settings.train_learning_rate,
            weight_decay=self.settings.train_weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        checkpoint_dir = self.project_root / "data" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best_model.pt"

        if best_path.exists():
            raise ValueError("Refusing to overwrite an existing classifier checkpoint")
        best_val_acc = -1.0
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for pixel_values, targets in train_loader:
                pixel_values = pixel_values.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = model(pixel_values)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            avg_loss = running_loss / max(1, len(train_loader))
            val_acc = self._evaluate_accuracy(model, val_loader)
            logger.info(
                "Epoch {}/{} - loss: {:.4f} - val_acc: {:.4f}", epoch + 1, epochs, avg_loss, val_acc
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "format": "clip_cell_classifier_v1",
                        "head_state": {
                            k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()
                        },
                        "clip_model_name": self.settings.clip_model_name,
                        "label_to_idx": label_to_idx,
                    },
                    best_path,
                )
                logger.info("Saved new best checkpoint to {}", best_path)

        if not best_path.exists():
            raise RuntimeError("No trained checkpoint was produced")
        return GeoClassifier.load(best_path, str(self.device))

    def _evaluate_accuracy(self, model: GeoClassifier, loader: DataLoader) -> float:
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for pixel_values, targets in loader:
                pixel_values = pixel_values.to(self.device)
                targets = targets.to(self.device)
                logits = model(pixel_values)
                preds = logits.argmax(dim=1)
                total += targets.numel()
                correct += (preds == targets).sum().item()
        return (correct / total) if total else 0.0
