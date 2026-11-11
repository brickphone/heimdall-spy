"""Predict route."""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from loguru import logger

from swespy.api.schemas import MatchResult, PredictResponse
from swespy.model.imageio import UnsupportedImage

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    region: str | None = Form(None),
) -> PredictResponse:
    """Locate an uploaded image. The upload is processed in memory and never stored.

    ``region`` (optional): JSON {"x", "y", "width", "height"} normalised to the upright
    photo; only that part is searched.
    """
    allowed_types = request.app.state.allowed_image_types
    max_file_bytes = request.app.state.max_upload_size_mb * 1024 * 1024

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG and HEIC are supported")

    contents = await file.read()
    if len(contents) > max_file_bytes:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    selected = None
    if region:
        try:
            selected = json.loads(region)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Region must be JSON") from exc
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(request.app.state.locator.locate_bytes, contents,
                                         selected)
    except UnsupportedImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000
    # No image hash or coordinates are logged, so logs cannot link a photo to a place.
    logger.info("predict latency_ms={:.2f}", elapsed_ms)

    return PredictResponse(
        predicted_lat=result["predicted_lat"],
        predicted_lng=result["predicted_lng"],
        prediction_method=result["prediction_method"],
        confidence=result["confidence"],
        confidence_status=result["confidence_status"],
        confidence_state=result.get("confidence_state", "unavailable"),
        confidence_reasons=result.get("confidence_reasons", []),
        top_matches=[
            MatchResult(similarity_score=c["similarity"], inliers=c.get("inliers"),
                        score=c.get("score"), source=c.get("source"),
                        capture_year=c.get("capture_year"), **{
                k: c[k] for k in ("rank", "image_id", "lat", "lng", "creator", "source_url",
                                  "license_url")})
            for c in result["candidates"]
        ],
        processing_time_ms=elapsed_ms,
        region=selected,
        versions=request.app.state.locator.versions(),
    )
