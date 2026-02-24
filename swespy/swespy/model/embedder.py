"""CLIP embedding extraction."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from swespy.config import Settings


class CLIPEmbedder:
    """Embeds images using a HuggingFace CLIP model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(settings.clip_model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(settings.clip_model_name)

    @staticmethod
    def _normalize(vectors: torch.Tensor) -> torch.Tensor:
        return vectors / vectors.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

    def embed_image(self, image_path: Path) -> np.ndarray:
        """Create a normalized embedding for one image on disk."""
        image = Image.open(image_path).convert("RGB")
        return self.embed_pil(image)

    def embed_pil(self, image: Image.Image) -> np.ndarray:
        """Create a normalized embedding for one PIL image."""
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=pixel_values)
            normalized = self._normalize(features)[0]
        return normalized.detach().cpu().numpy().astype(np.float32)

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Create a normalized embedding from in-memory bytes."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.embed_pil(image)

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        """Create normalized embeddings for a batch of image paths."""
        batches: list[np.ndarray] = []
        batch_size = self.settings.batch_size

        for i in range(0, len(image_paths), batch_size):
            current_paths = image_paths[i : i + batch_size]
            images = [Image.open(path).convert("RGB") for path in current_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                features = self.model.get_image_features(pixel_values=pixel_values)
                normalized = self._normalize(features)
            batches.append(normalized.detach().cpu().numpy().astype(np.float32))

        if not batches:
            return np.zeros((0, self.settings.embedding_dim), dtype=np.float32)

        return np.vstack(batches)
