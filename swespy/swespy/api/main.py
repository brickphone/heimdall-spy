"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
import sys
from typing import Any

from fastapi import FastAPI
from loguru import logger
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from swespy.api.middleware import request_id_middleware
from swespy.api.routes.health import router as health_router
from swespy.api.routes.predict import router as predict_router
from swespy.config import get_settings
from swespy.model.embedder import CLIPEmbedder
from swespy.vector_store.qdrant_client import VectorStore


class DummyEmbedder:
    """Fast lightweight embedder for tests."""

    def embed_bytes(self, _: bytes) -> Any:
        import numpy as np

        vec = np.zeros(768, dtype=np.float32)
        vec[0] = 1.0
        return vec


class DummyVectorStore:
    """Fast lightweight vector store for tests."""

    def query(self, embedding: Any, top_k: int) -> list[dict[str, Any]]:
        _ = embedding
        return [
            {"lat": 59.3293, "lng": 18.0686, "similarity_score": 0.9, "s2_cell": "89c25", "image_id": "1"}
            for _ in range(top_k)
        ]

    def aggregate_prediction(self, results: list[dict[str, Any]]) -> dict[str, float]:
        _ = results
        return {"lat": 59.3293, "lng": 18.0686, "confidence": 0.9}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and vector store at app startup."""
    settings = get_settings()
    if settings.testing:
        app.state.embedder = DummyEmbedder()
        app.state.vector_store = DummyVectorStore()
        logger.info("Running in testing mode with dummy components")
    else:
        app.state.embedder = CLIPEmbedder(settings)
        app.state.vector_store = VectorStore(settings)
        app.state.vector_store.create_collection()
        logger.info("Loaded embedder and vector store")

    app.state.top_k_results = settings.top_k_results
    app.state.allowed_image_types = set(settings.allowed_image_types)
    app.state.max_upload_size_mb = settings.max_upload_size_mb
    yield


def create_app() -> FastAPI:
    """Construct FastAPI application."""
    settings = get_settings()
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.api_rate_limit_per_minute}/minute"],
    )

    app = FastAPI(title="SweSpy API", version="0.1.0", lifespan=lifespan)
    app.state.limiter = limiter

    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.middleware("http")(request_id_middleware)

    app.include_router(health_router)
    app.include_router(predict_router)

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    return app


app = create_app()
