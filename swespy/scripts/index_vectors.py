"""Create and index CLIP embeddings in Qdrant."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from swespy.config import get_settings
from swespy.model.embedder import CLIPEmbedder
from swespy.vector_store.qdrant_client import VectorStore

if __name__ == "__main__":
    settings = get_settings()
    root = Path.cwd()
    metadata = pd.read_parquet(root / "data" / "metadata" / "images.parquet")

    store = VectorStore(settings)
    store.create_collection()
    if store.client.count(collection_name=store.collection_name, exact=True).count:
        raise RuntimeError(
            "Refusing to overwrite a populated collection. Preserve the existing index "
            "and explicitly plan any replacement or incremental indexing."
        )

    if metadata.empty:
        raise RuntimeError("Reference metadata is empty; nothing can be indexed")
    # Resolve from stable image IDs so Windows-created metadata works in Linux containers.
    image_paths = [root / "data" / "raw" / f"{image_id}.jpg" for image_id in metadata["image_id"]]
    if any(not path.is_file() for path in image_paths):
        raise RuntimeError("Reference images are missing from data/raw; indexing was not started")
    embedder = CLIPEmbedder(settings)
    batches = []
    for start in range(0, len(image_paths), settings.batch_size):
        current = image_paths[start : start + settings.batch_size]
        batches.append(embedder.embed_batch(current))
        logger.info("Embedded {}/{} reference images", start + len(current), len(image_paths))
    embeddings = np.vstack(batches)
    if (
        embeddings.shape != (len(image_paths), settings.embedding_dim)
        or not np.isfinite(embeddings).all()
        or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    ):
        raise RuntimeError("Invalid reference embeddings; nothing was upserted")
    payloads = metadata[["image_id", "lat", "lng", "s2_cell"]].to_dict("records")
    store.upsert_embeddings(embeddings, payloads)
