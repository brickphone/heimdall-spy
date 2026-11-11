"""FastAPI application entrypoint."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from swespy.api.middleware import request_id_middleware
from swespy.api.routes.field_photos import router as field_photos_router
from swespy.api.routes.health import router as health_router
from swespy.api.routes.predict import router as predict_router
from swespy.api.routes.references import router as references_router
from swespy.config import Settings, get_settings


class DummyLocator:
    """Fast lightweight locator for tests; never loads model weights or an index."""

    gallery = None

    def versions(self) -> dict[str, Any]:
        return {"model": {"name": "dummy"}, "gallery": {"file": None}, "rerank": {}}

    def locate_bytes(self, contents: bytes,
                     region: dict[str, float] | None = None) -> dict[str, Any]:
        _ = contents, region
        return {
            "predicted_lat": 58.4108, "predicted_lng": 15.6214,
            "prediction_method": "top-ranked reference", "confidence": 0.9,
            "confidence_status": "dummy",
            "confidence_state": "ok",
            "confidence_reasons": [],
            "candidates": [
                {"rank": i, "image_id": str(i), "lat": 58.4108, "lng": 15.6214,
                 "similarity": 0.9, "creator": None, "source_url": None, "license_url": None}
                for i in range(1, 6)
            ],
            "searched_variant": "manual" if region else "whole",
            "searched_region": region or {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            "person_share": 0.0,
            "alternative_region": {"x": 0.3, "y": 0.0, "width": 0.4, "height": 1.0},
            "variants": [{"id": "manual" if region else "whole", "region": region,
                          "top_similarity": 0.9}],
            "timings_ms": {},
        }

    def analyze_bytes(self, contents: bytes) -> dict[str, Any]:
        _ = contents
        return {"person_share": 0.0, "suggested_region": None, "suggest": False,
                "variants": [{"id": "whole", "region": {"x": 0.0, "y": 0.0, "width": 1.0,
                                                        "height": 1.0}}]}

    def locate_events(self, contents: bytes, region: dict[str, float] | None = None,
                      cancelled: Any = lambda: False) -> Any:
        """The same event sequence as GeoLocator.locate_events, with fixed values."""
        yield {"type": "stage", "stage": "retrieval"}
        yield {"type": "candidates", "items": [
            {"id": n, "lat": 58.41 + n * 1e-4, "lng": 15.62, "similarity": 0.9 - n * 0.01}
            for n in range(50)]}
        yield {"type": "stage", "stage": "verification", "count": 5}
        for n in range(5):
            yield {"type": "verified", "id": n, "inliers": 40 - n, "score": 0.9}
        yield {"type": "_result", "result": self.locate_bytes(contents, region)}


def build_locator(settings: Settings) -> Any:
    """Load MegaLoc and the polygon-restricted gallery: the one inference path."""
    from swespy.model.locator import Gallery, GeoLocator
    from swespy.model.megaloc import MegaLocEmbedder

    gallery = Gallery(Path(settings.gallery_index), settings.area_polygon)
    embedder = MegaLocEmbedder(Path(settings.megaloc_root), settings.megaloc_device)
    segmenter = None
    if settings.variants_mode != "whole":
        from swespy.model.variants import PersonSegmenter

        segmenter = PersonSegmenter(settings.segment_device)
    variants = {"segmenter": segmenter, "variants_mode": settings.variants_mode}
    if not settings.rerank_enabled:
        return GeoLocator(embedder, gallery, settings.top_k_results, settings.retrieval_depth,
                          **variants)
    from swespy.model import rerank as rr

    verifier = rr.GeometricVerifier(settings.rerank_device)
    return GeoLocator(embedder, gallery, settings.top_k_results, settings.retrieval_depth,
                      verifier=verifier, weight=rr.DEFAULT_WEIGHT,
                      min_inliers=rr.DEFAULT_MIN_INLIERS, rerank_k=settings.rerank_k, **variants)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and gallery at app startup."""
    settings = get_settings()
    if settings.testing:
        app.state.locator = DummyLocator()
        logger.info("Running in testing mode with a dummy locator")
    elif not hasattr(app.state, "locator"):
        app.state.locator = build_locator(settings)
        logger.info("Loaded MegaLoc and {} gallery references",
                    len(app.state.locator.gallery.refs))

    app.state.top_k_results = settings.top_k_results
    app.state.allowed_image_types = set(settings.allowed_image_types)
    app.state.max_upload_size_mb = settings.max_upload_size_mb
    app.state.field_photos_dir = settings.field_photos_dir
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
    app.state.top_k_results = settings.top_k_results
    app.state.allowed_image_types = set(settings.allowed_image_types)
    app.state.max_upload_size_mb = settings.max_upload_size_mb
    app.state.field_photos_dir = settings.field_photos_dir
    if settings.testing:
        app.state.locator = DummyLocator()

    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.middleware("http")(request_id_middleware)

    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(references_router)
    app.include_router(field_photos_router)

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    return app


app = create_app()
