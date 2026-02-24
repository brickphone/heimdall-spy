"""Health route."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health endpoint."""
    return {"status": "ok"}
