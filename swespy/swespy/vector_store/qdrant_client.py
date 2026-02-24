"""Qdrant vector store client."""

from __future__ import annotations

from itertools import count

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from swespy.config import Settings


class VectorStore:
    """Qdrant-backed vector store for image embeddings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection_name = settings.qdrant_collection_name

    def create_collection(self) -> None:
        """Create collection if it does not exist."""
        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.collection_name in existing:
            logger.info("Collection '{}' already exists", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.settings.embedding_dim, distance=Distance.COSINE),
        )
        logger.info("Created collection '{}'", self.collection_name)

    def upsert_embeddings(self, embeddings: np.ndarray, payloads: list[dict]) -> None:
        """Upload embeddings and payloads to Qdrant in batches."""
        if embeddings.shape[0] != len(payloads):
            raise ValueError("Embeddings and payloads length mismatch")

        ids = count(1)
        batch_size = self.settings.vector_upsert_batch_size
        for i in range(0, len(payloads), batch_size):
            batch_embeddings = embeddings[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]
            points = [
                PointStruct(id=next(ids), vector=vector.tolist(), payload=payload)
                for vector, payload in zip(batch_embeddings, batch_payloads, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Upserted {} vectors", len(payloads))

    def query(self, embedding: np.ndarray, top_k: int) -> list[dict]:
        """Search nearest neighbors and return payloads with scores."""
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )

        results: list[dict] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "lat": float(payload.get("lat", 0.0)),
                    "lng": float(payload.get("lng", 0.0)),
                    "s2_cell": str(payload.get("s2_cell", "")),
                    "similarity_score": float(hit.score),
                    "image_id": str(payload.get("image_id", "")),
                }
            )
        return results

    @staticmethod
    def aggregate_prediction(results: list[dict]) -> dict[str, float]:
        """Aggregate top matches into a weighted coordinate prediction."""
        if not results:
            return {"lat": 0.0, "lng": 0.0, "confidence": 0.0}

        weights = np.array([max(0.0, r["similarity_score"]) for r in results], dtype=np.float64)
        if float(weights.sum()) == 0.0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        lats = np.array([r["lat"] for r in results], dtype=np.float64)
        lngs = np.array([r["lng"] for r in results], dtype=np.float64)
        pred_lat = float(np.dot(weights, lats))
        pred_lng = float(np.dot(weights, lngs))
        confidence = float(np.mean([r["similarity_score"] for r in results]))
        return {"lat": pred_lat, "lng": pred_lng, "confidence": confidence}
