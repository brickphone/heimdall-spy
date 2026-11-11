"""Predict routes: one-shot (/predict) and streamed progress (/predict/stream)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from swespy.api.schemas import MatchResult, PredictResponse
from swespy.model.imageio import UnsupportedImage

router = APIRouter(tags=["predict"])
POLL_S = 0.25  # How often the stream checks for a disconnected client while work runs.


async def read_upload(request: Request, file: UploadFile, region: str | None
                      ) -> tuple[bytes, dict[str, float] | None]:
    """Shared validation for both routes. Raises HTTP 400 with a plain message."""
    if file.content_type not in request.app.state.allowed_image_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG and HEIC are supported")
    contents = await file.read()
    if len(contents) > request.app.state.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")
    selected = None
    if region:
        try:
            selected = json.loads(region)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Region must be JSON") from exc
    return contents, selected


def to_response(result: dict[str, Any], elapsed_ms: float, region: dict[str, float] | None,
                versions: dict[str, Any]) -> PredictResponse:
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
        region=region,
        versions=versions,
    )


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
    contents, selected = await read_upload(request, file, region)
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(request.app.state.locator.locate_bytes, contents,
                                         selected)
    except UnsupportedImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000
    # No image hash or coordinates are logged, so logs cannot link a photo to a place.
    logger.info("predict latency_ms={:.2f}", elapsed_ms)
    return to_response(result, elapsed_ms, selected, request.app.state.locator.versions())


async def event_stream(locator: Any, contents: bytes, region: dict[str, float] | None,
                       is_disconnected: Callable[[], Awaitable[bool]]) -> AsyncIterator[bytes]:
    """NDJSON progress events from ``locator.locate_events``, run in a worker thread.

    If the client disconnects (or the response is closed), ``cancel`` is set and the
    locator stops at its next check, so no further work is done for this request.
    """
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    done = object()
    start = time.perf_counter()

    def put(item: Any) -> None:
        loop.call_soon_threadsafe(events.put_nowait, item)

    def work() -> None:
        from swespy.model.locator import Cancelled

        try:
            for event in locator.locate_events(contents, region, cancel.is_set):
                put(event)
        except Cancelled:
            logger.info("predict stream cancelled by the client")
        except UnsupportedImage as exc:
            put({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - reported to the client, details logged.
            logger.exception("predict stream failed: {}", type(exc).__name__)
            put({"type": "error", "message": "Locating failed. Try again."})
        finally:
            put(done)

    worker = loop.run_in_executor(None, work)
    try:
        while True:
            try:
                event = await asyncio.wait_for(events.get(), timeout=POLL_S)
            except TimeoutError:
                if await is_disconnected():
                    break
                continue
            if event is done:
                break
            if event["type"] == "_result":
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("predict stream latency_ms={:.2f}", elapsed_ms)
                response = to_response(event["result"], elapsed_ms, region, locator.versions())
                event = {"type": "result", **response.model_dump()}
            yield (json.dumps(event) + "\n").encode()
            if await is_disconnected():
                break
    finally:
        cancel.set()  # Stops the worker if the client left or the stream was closed early.
        del worker


@router.post("/predict/stream")
async def predict_stream(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    region: str | None = Form(None),
) -> StreamingResponse:
    """Like /predict, but streams progress as NDJSON (one JSON object per line).

    Events, in order: stage "retrieval"; "candidates" (top retrieval depth: id, lat, lng,
    similarity only, never images or sources); stage "verification" (count); one
    "verified" per re-ranked candidate (id, inliers, score) as it completes; "result"
    (exactly /predict's response); or "error" (message). Disconnecting cancels the work.
    """
    contents, selected = await read_upload(request, file, region)
    return StreamingResponse(
        event_stream(request.app.state.locator, contents, selected, request.is_disconnected),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
