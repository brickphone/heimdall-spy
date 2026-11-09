"""Gallery reference images, for showing a match next to the query (with attribution)."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(tags=["references"])


@router.get("/references/{image_id}/image")
async def reference_image(request: Request, image_id: str) -> FileResponse:
    """The stored 640 px reference image. Looked up by ID in the loaded gallery only."""
    gallery = getattr(getattr(request.app.state, "locator", None), "gallery", None)
    path = gallery.image_path(image_id) if gallery is not None else None
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown reference")
    ref = gallery.reference(image_id)
    headers = {"X-Source-Url": ref.get("source_url") or "",
               "X-License-Url": ref.get("license_url") or "",
               "Cache-Control": "public, max-age=86400"}
    return FileResponse(path, media_type="image/jpeg", headers=headers)
