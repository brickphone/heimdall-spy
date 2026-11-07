"""Opt-in: the API and the evaluation script give the same answers for the dev photos.

Requires the local MegaLoc weights, the exported gallery index, the development photos
and ``eval/results/baseline_v2.json``. Run with:

    SWESPY_RUN_REAL=1 PYTORCH_ENABLE_MPS_FALLBACK=1 MEGALOC_DEVICE=mps \
        .venv/bin/python -m pytest tests/integration/test_api_matches_eval.py
"""

import json
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "data/public_eval/recent_linkoping_streets_v1"
BASELINE = ROOT / "eval/results/baseline_v2.json"


@pytest.mark.skipif(os.environ.get("SWESPY_RUN_REAL") != "1",
                    reason="Needs local MegaLoc weights, gallery index and dev photos")
async def test_api_path_equals_eval_path(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("TOP_K_RESULTS", "5")
    from swespy.api.main import build_locator, create_app
    from swespy.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.state.locator = build_locator(get_settings())
    expected = {row["image_id"]: row for row in json.loads(BASELINE.read_text())["queries"]}
    queries = json.loads((DEV / "manifest.json").read_text())["queries"]
    mismatches = []
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for q in queries:
            files = {"file": ("q.jpg", (DEV / q["path"]).read_bytes(), "image/jpeg")}
            response = await client.post("/predict", files=files)
            assert response.status_code == 200
            got = response.json()
            want = expected[q["image_id"]]
            ids = [m["image_id"] for m in got["top_matches"]]
            if ids != [m["image_id"] for m in want["top5"]] or abs(
                got["top_matches"][0]["similarity_score"] - want["top5"][0]["similarity"]
            ) > 1e-3:
                mismatches.append(q["image_id"])
    assert not mismatches, f"API and eval disagree for {len(mismatches)} photos: {mismatches}"
