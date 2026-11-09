"""Opt-in: with re-ranking on, the API gives the same answers as the evaluation path.

Compares POST /predict (MegaLoc + ALIKED/LightGlue verification of the top K=5, shared
``rerank.finalize``) with the evaluation result for the same photo, computed from the
cached dev inliers (``data/gallery_v2/rerank/dev_pairs_k50.json``, first 5 candidates).
Needs local weights, the historic gallery and the dev photos:

    SWESPY_RUN_REAL=1 PYTORCH_ENABLE_MPS_FALLBACK=1 MEGALOC_DEVICE=mps RERANK_DEVICE=mps \
        .venv/bin/python -m pytest tests/integration/test_api_rerank_matches_eval.py
"""

import json
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/public_eval/recent_linkoping_streets_v1"
PAIRS = ROOT / "data/gallery_v2/rerank/dev_pairs_k50.json"
N_PHOTOS = 10
INLIER_TOLERANCE = 3  # RANSAC/MPS numerics may move a count by a few inliers.


@pytest.mark.skipif(os.environ.get("SWESPY_RUN_REAL") != "1",
                    reason="Needs local weights, gallery, dev photos and cached dev inliers")
async def test_api_rerank_path_equals_eval_path(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("TOP_K_RESULTS", "5")
    from swespy.api.main import build_locator, create_app
    from swespy.config import get_settings
    from swespy.model import rerank as rr

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.rerank_enabled and settings.rerank_k == 5
    app = create_app()
    app.state.locator = build_locator(settings)
    cached = json.loads(PAIRS.read_text())["queries"]
    queries = json.loads((DEV / "manifest.json").read_text())["queries"][:N_PHOTOS]
    problems = []
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for q in queries:
            files = {"file": ("q.jpg", (DEV / q["path"]).read_bytes(), "image/jpeg")}
            got = (await client.post("/predict", files=files)).json()
            want = rr.finalize(cached[q["image_id"]][: settings.rerank_k], rr.DEFAULT_WEIGHT,
                               rr.DEFAULT_MIN_INLIERS, top_k=5)
            want_inliers = {c["image_id"]: c["inliers"] for c in want["top"]}
            got_ids = [m["image_id"] for m in got["top_matches"]]
            if got_ids[0] != want["top"][0]["image_id"]:
                problems.append((q["image_id"], "top-1 differs"))
            if set(got_ids) != set(want_inliers):
                problems.append((q["image_id"], "top-5 set differs"))
            if got["confidence_state"] != want["confidence"]["state"]:
                problems.append((q["image_id"], "confidence state differs"))
            for m in got["top_matches"]:
                if abs(m["inliers"] - want_inliers.get(m["image_id"], -99)) > INLIER_TOLERANCE:
                    problems.append((q["image_id"], f"inliers {m['inliers']} vs "
                                                    f"{want_inliers.get(m['image_id'])}"))
            if abs(got["predicted_lat"] - want["predicted_lat"]) > 1e-5 or \
                    abs(got["predicted_lng"] - want["predicted_lng"]) > 1e-5:
                problems.append((q["image_id"], "predicted location differs"))
    assert not problems, problems
