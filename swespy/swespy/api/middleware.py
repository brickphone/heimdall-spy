"""Custom API middleware."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from loguru import logger


async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach request ID and log request lifecycle."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id={} method={} path={} status={} latency_ms={:.2f}",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response
