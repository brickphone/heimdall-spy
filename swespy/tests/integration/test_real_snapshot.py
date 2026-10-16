"""Opt-in real model/index snapshots; requires private fixtures captured once."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("SWESPY_RUN_REAL_SNAPSHOT") != "1",
    reason="Real snapshot pending private photos/labels, reference index and one-time capture",
)
def test_real_pipeline_snapshot() -> None:
    """Compare fixed private queries through actual CLIP/Qdrant/API inference."""
    env = {**os.environ, "TESTING": "false"}
    subprocess.run(
        [sys.executable, "-B", "-m", "eval.run_eval", "--check-snapshot"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=True,
    )
