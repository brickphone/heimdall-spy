"""Predict route."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger

from swespy.api.schemas import MatchResult, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    """Locate an uploaded image. The upload is processed in memory and never stored."""
    allowed_types = request.app.state.allowed_image_types
    max_file_bytes = request.app.state.max_upload_size_mb * 1024 * 1024

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported")

    contents = await file.read()
    if len(contents) > max_file_bytes:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(request.app.state.locator.locate_bytes, contents)
    except OSError as exc:  # PIL cannot decode the upload.
        raise HTTPException(status_code=400, detail="Unreadable image") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000
    # No image hash or coordinates are logged, so logs cannot link a photo to a place.
    logger.info("predict latency_ms={:.2f}", elapsed_ms)

    return PredictResponse(
        predicted_lat=result["predicted_lat"],
        predicted_lng=result["predicted_lng"],
        prediction_method=result["prediction_method"],
        confidence=result["confidence"],
        confidence_status=result["confidence_status"],
        top_matches=[
            MatchResult(similarity_score=c["similarity"], **{
                k: c[k] for k in ("rank", "image_id", "lat", "lng", "creator", "source_url",
                                  "license_url")})
            for c in result["candidates"]
        ],
        processing_time_ms=elapsed_ms,
    )
