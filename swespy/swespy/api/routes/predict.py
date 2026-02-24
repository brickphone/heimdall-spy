"""Predict route."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger

from swespy.api.schemas import MatchResult, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    """Infer GPS coordinates from an uploaded image."""
    allowed_types = request.app.state.allowed_image_types
    max_file_bytes = request.app.state.max_upload_size_mb * 1024 * 1024

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported")

    contents = await file.read()
    if len(contents) > max_file_bytes:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    embedder = request.app.state.embedder
    vector_store = request.app.state.vector_store
    top_k = request.app.state.top_k_results

    start = time.perf_counter()
    embedding = await asyncio.to_thread(embedder.embed_bytes, contents)
    results: list[dict[str, Any]] = await asyncio.to_thread(
        vector_store.query,
        embedding,
        top_k,
    )
    aggregate = vector_store.aggregate_prediction(results)
    elapsed_ms = (time.perf_counter() - start) * 1000

    image_hash = hashlib.sha256(contents).hexdigest()[:16]
    logger.info(
        "image_hash={} predicted_lat={} predicted_lng={} confidence={} latency_ms={:.2f}",
        image_hash,
        aggregate["lat"],
        aggregate["lng"],
        aggregate["confidence"],
        elapsed_ms,
    )

    top_matches = [MatchResult(**result) for result in results]
    return PredictResponse(
        predicted_lat=aggregate["lat"],
        predicted_lng=aggregate["lng"],
        confidence=max(0.0, min(1.0, aggregate["confidence"])),
        top_matches=top_matches,
        processing_time_ms=elapsed_ms,
    )
