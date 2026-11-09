"""Liveness and readiness."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness: the process is up and serving HTTP."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: the model and gallery index are loaded (503 until then)."""
    locator = getattr(request.app.state, "locator", None)
    if locator is None:
        return JSONResponse({"ready": False}, status_code=503)
    return JSONResponse({"ready": True, "versions": locator.versions()})
