"""Opt-in storage of field photos for a FUTURE evaluation set.

Called only when the user ticks "save this photo for later evaluation". Field photos are
never added to the gallery, the dev set or any tuning. /predict itself stores nothing.
The saved image is the decoded, upright RGB pixels re-encoded as JPEG: no EXIF/GPS.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from swespy.model.imageio import UnsupportedImage, decode_upload

router = APIRouter(tags=["field-photos"])


@router.post("/field-photos")
async def save_field_photo(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    prediction: str = Form(...),
    true_lat: float = Form(...),
    true_lng: float = Form(...),
) -> dict[str, str]:
    if not (math.isfinite(true_lat) and math.isfinite(true_lng)
            and -90 <= true_lat <= 90 and -180 <= true_lng <= 180):
        raise HTTPException(status_code=400, detail="Invalid true location")
    try:
        predicted = json.loads(prediction)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="prediction must be JSON") from exc
    contents = await file.read()
    if len(contents) > request.app.state.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    try:
        image = decode_upload(contents)
    except UnsupportedImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    digest = hashlib.sha256(contents).hexdigest()
    saved_at = datetime.now(UTC)
    folder = Path(request.app.state.field_photos_dir) / (
        f"{saved_at.strftime('%Y%m%dT%H%M%SZ')}-{digest[:10]}")
    folder.mkdir(parents=True, exist_ok=False)
    image.save(folder / "photo.jpg", format="JPEG", quality=95)  # Pixels only.
    record = {
        "saved_at": saved_at.isoformat(), "upload_sha256": digest,
        "true_location": {"lat": true_lat, "lng": true_lng,
                          "source": "clicked on the demo map by the photographer"},
        "prediction": predicted,
        "use": "future evaluation set only; never gallery, dev set or tuning",
    }
    (folder / "record.json").write_text(json.dumps(record, indent=1))
    return {"saved": folder.name}
