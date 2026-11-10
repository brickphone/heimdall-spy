"""Street View internal-backend pilot, Part 3: tile-rendered views vs the official API.

For 50 pilot views (one per panorama, seeded draw) fetch the same view (pano ID,
heading, FOV 90, 640x640, pitch 0) from the official Street View Static API with the
owner's key, in memory only, encode it with MegaLoc (fp32, CPU like the pod) and compare
it with the pilot's stored descriptor. The 50 image requests are reserved against the
official monthly cap and logged in data/gsv/pull_log.jsonl (official route), together
with the free metadata checks. Nothing is written except descriptors-free summary numbers.
"""

from __future__ import annotations

import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from swespy import gsv

PILOT = Path("data/gsv_internal/gsv_internal_pilot")
OFFICIAL_LOG = Path("data/gsv/pull_log.jsonl")
STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"
N_VIEWS = 50
SEED = 20261110


def pilot_views() -> tuple[list[dict[str, Any]], np.ndarray]:
    views = [json.loads(x) for x in (PILOT / "views.jsonl").read_text().splitlines()]
    chunks = sorted(PILOT.glob("descriptors-*.npy"))
    desc = np.concatenate([np.load(p) for p in chunks]).astype(np.float32)
    assert len(desc) == len(views) and all(v["row"] == i for i, v in enumerate(views))
    return views, desc


def choose(views: list[dict[str, Any]], n: int = N_VIEWS, seed: int = SEED) -> list[int]:
    rng = random.Random(seed)
    by_pano: dict[str, list[int]] = {}
    for i, v in enumerate(views):
        by_pano.setdefault(v["pano_id"], []).append(i)
    panos = sorted(by_pano)
    rng.shuffle(panos)
    return [rng.choice(by_pano[p]) for p in panos[:n]]


def main() -> None:
    import httpx
    from PIL import Image

    from swespy.model.megaloc import MegaLocEmbedder

    key = gsv.api_key()
    views, desc = pilot_views()
    picks = choose(views)
    log = gsv.PullLog(OFFICIAL_LOG)
    log.reserve(len(picks), "image")  # Raises before any request if the cap would be crossed.
    lats = [views[i]["lat"] for i in picks]
    lngs = [views[i]["lng"] for i in picks]
    bbox = (min(lngs), min(lats), max(lngs), max(lats))
    embedder = MegaLocEmbedder(Path("data/megaloc"), "cpu")
    meta_ok = img_ok = 0
    images: list[Any] = []
    kept: list[int] = []
    official_meta: list[dict[str, Any]] = []
    with httpx.Client(timeout=30) as client:
        for i in picks:
            v = views[i]
            try:
                m = client.get(f"{STATIC_URL}/metadata", params={"pano": v["pano_id"], "key": key})
                meta = m.json() if m.status_code == 200 else {"status": f"http_{m.status_code}"}
            except httpx.HTTPError as exc:
                meta = {"status": type(exc).__name__}  # No message: it can contain the URL.
            if meta.get("status") != "OK":
                official_meta.append({"pano_id": v["pano_id"], "status": meta.get("status")})
                continue
            meta_ok += 1
            official_meta.append({"pano_id": v["pano_id"], "status": "OK",
                                  "copyright": meta.get("copyright"), "date": meta.get("date")})
            try:
                r = client.get(STATIC_URL, params={
                    "pano": v["pano_id"], "heading": v["heading"], "fov": 90, "pitch": 0,
                    "size": "640x640", "return_error_code": "true", "key": key})
            except httpx.HTTPError as exc:
                official_meta[-1]["image"] = type(exc).__name__
                continue
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
                official_meta[-1]["image"] = f"http_{r.status_code}"
                continue
            img_ok += 1
            with Image.open(BytesIO(r.content)) as im:
                images.append(im.convert("RGB").copy())  # In memory only.
            kept.append(i)
    # Every request that was sent is logged; image requests count against the cap.
    log.append(request_type="metadata", request_count=len(picks), success_count=meta_ok,
               bbox=bbox, priority=None)
    log.append(request_type="image", request_count=meta_ok, success_count=img_ok, bbox=bbox,
               priority=None)
    official = embedder.batch_images(images) if images else np.zeros((0, desc.shape[1]))
    del images
    pilot = desc[kept]
    pilot /= np.linalg.norm(pilot, axis=1, keepdims=True)
    same = np.sum(official * pilot, axis=1)
    scores = official @ (desc / np.linalg.norm(desc, axis=1, keepdims=True)).T
    rank1 = int(sum(int(np.argmax(scores[j])) == kept[j] for j in range(len(kept))))
    other_heading, other_pano = [], []
    for j, i in enumerate(kept):
        pano = views[i]["pano_id"]
        for k, w in enumerate(views):
            if k == i:
                continue
            (other_heading if w["pano_id"] == pano else other_pano).append(scores[j, k])
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    result = {
        "views_requested": len(picks), "metadata_ok": meta_ok, "images_ok": img_ok,
        "official_copyrights": sorted({m.get("copyright") for m in official_meta
                                       if m.get("copyright")}),
        "cosine_same_view": {"mean": float(same.mean()), "median": q(same, 50),
                             "p5": q(same, 5), "min": float(same.min())},
        "cosine_same_pano_other_heading": {"median": q(other_heading, 50),
                                           "p95": q(other_heading, 95)},
        "cosine_other_panos": {"median": q(other_pano, 50), "p99": q(other_pano, 99),
                               "max": float(np.max(other_pano))},
        "official_view_retrieves_its_pilot_view_top1": f"{rank1}/{len(kept)}",
        "failures": [m for m in official_meta if m.get("status") != "OK" or "image" in m],
        "used_this_month": log.used_this_month("image"),
        "cap": gsv.MONTHLY_IMAGE_CAP,
    }
    (Path("data/gsv_internal") / "official_comparison.json").write_text(
        json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
