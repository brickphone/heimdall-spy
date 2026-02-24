"""Create and index CLIP embeddings in Qdrant."""

from pathlib import Path

import pandas as pd

from swespy.config import get_settings
from swespy.model.embedder import CLIPEmbedder
from swespy.vector_store.qdrant_client import VectorStore


if __name__ == "__main__":
    settings = get_settings()
    root = Path.cwd()
    metadata = pd.read_parquet(root / "data" / "metadata" / "images.parquet")

    embedder = CLIPEmbedder(settings)
    store = VectorStore(settings)
    store.create_collection()

    image_paths = [Path(path) for path in metadata["local_path"].tolist()]
    embeddings = embedder.embed_batch(image_paths)
    payloads = metadata[["image_id", "lat", "lng", "s2_cell"]].to_dict("records")
    store.upsert_embeddings(embeddings, payloads)
