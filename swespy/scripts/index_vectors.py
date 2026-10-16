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
    embeddings = embedder.embed_batch(image_paths)
    payloads = metadata[["image_id", "lat", "lng", "s2_cell"]].to_dict("records")
    store.upsert_embeddings(embeddings, payloads)
