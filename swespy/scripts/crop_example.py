"""Manual-crop example for the About page, measured live against the local API.

Every crop tried is recorded (no cherry-picking): the whole photo plus each manual
region, with the error to the photo's true location and the confidence state. The dev
photo (Wikimedia Commons, CC BY-SA 4.0) is copied resized with its credit. Regions are
cropped locally to lossless PNG, which is exactly what POST /predict's region does.
Usage (API running on :8000): .venv/bin/python scripts/crop_example.py
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import httpx

from swespy.model.imageio import crop_region, decode_upload

DEV = Path("data/public_eval/recent_linkoping_streets_v1")
OUT_JSON = Path("web/src/generated/crop_example.json")
OUT_IMG = Path("web/public/about")
API = "http://127.0.0.1:8000/predict"
TRIED = {
    "commons-199099878": [("the brick building behind the van",
                           {"x": 0.0, "y": 0.0, "width": 0.40, "height": 0.45})],
    "commons-199596662": [("the station building", {"x": 0.0, "y": 0.18, "width": 0.26,
                                                    "height": 0.55}),
                          ("building, flags and trees", {"x": 0.0, "y": 0.08, "width": 0.75,
                                                         "height": 0.42})],
}
SHOWN = "commons-199099878"


def haversine(a: float, b: float, c: float, d: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(a), math.radians(c)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def main() -> None:
    queries = {q["image_id"]: q for q in json.loads((DEV / "manifest.json").read_text())["queries"]}
    out = {"attempts": [], "shown": SHOWN}
    for image_id, crops in TRIED.items():
        q = queries[image_id]
        raw = (DEV / q["path"]).read_bytes()
        for label, region in [("whole photo", None), *crops]:
            body, mime = raw, "image/jpeg"
            if region:
                buf = io.BytesIO()
                crop_region(decode_upload(raw), region).save(buf, format="PNG")
                body, mime = buf.getvalue(), "image/png"
            r = httpx.post(API, files={"file": ("q", body, mime)}, timeout=180).json()
            top = r["top_matches"][0]
            out["attempts"].append({
                "image_id": image_id, "label": label, "region": region,
                "error_m": round(haversine(q["lat"], q["lng"], r["predicted_lat"],
                                           r["predicted_lng"])),
                "state": r["confidence_state"], "top_inliers": top["inliers"]})
    q = queries[SHOWN]
    out["photo"] = {"file": f"/about/{SHOWN}.jpg", "creator": q.get("creator"),
                    "license_url": q.get("license_url"), "source_url": q.get("source_url")}
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    image = decode_upload((DEV / q["path"]).read_bytes())
    image.thumbnail((960, 960))
    image.save(OUT_IMG / f"{SHOWN}.jpg", format="JPEG", quality=85)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["attempts"], indent=1))


if __name__ == "__main__":
    main()
