"""Re-ranking score combination, location prediction and low-confidence logic."""

import math

import pytest

from swespy.model import rerank as rr

M = 1 / 111_320  # About one metre of latitude.


def cand(i, sim, inliers, dlat_m=0.0):
    return {"index": i, "similarity": sim, "inliers": inliers, "lat": 58.41 + dlat_m * M,
            "lng": 15.62}


def test_combined_score_and_reordering():
    assert rr.combined_scores([0.5, 0.4], [0, 100], 0.1) == pytest.approx(
        [0.5, 0.4 + 0.1 * math.log1p(100)])
    ranked = rr.rerank([cand(0, 0.50, 0), cand(1, 0.45, 80)], weight=0.05)
    assert [c["index"] for c in ranked] == [1, 0]  # Geometry overrides a small cosine gap.
    assert [c["index"] for c in rr.rerank([cand(0, 0.5, 0), cand(1, 0.45, 80)], 0.0)] == [0, 1]


def test_predicted_location_ignores_far_candidates():
    ranked = rr.rerank([cand(0, 0.9, 50, 0), cand(1, 0.8, 40, 40), cand(2, 0.7, 30, 5000)], 0.0)
    lat, lng, used = rr.predict_location(ranked)
    assert used == 2  # The candidate 5 km away is an outlier.
    assert 58.41 < lat < 58.41 + 40 * M


def test_low_confidence_states():
    good = rr.rerank([cand(0, 0.9, 60), cand(1, 0.8, 40, 50)], 0.01)
    assert rr.confidence_state(good, 15)["state"] == "ok"
    few = rr.rerank([cand(0, 0.9, 3), cand(1, 0.8, 2, 50)], 0.01)
    assert rr.confidence_state(few, 15)["state"] == "low"
    split = rr.rerank([cand(0, 0.9, 60), cand(1, 0.8, 50, 2000)], 0.01)
    state = rr.confidence_state(split, 15)
    assert state["state"] == "low" and "away" in state["reasons"][0]
    # A far runner-up without geometric support is not a disagreement.
    noise = rr.rerank([cand(0, 0.9, 60), cand(1, 0.8, 0, 2000)], 0.01)
    assert rr.confidence_state(noise, 15)["state"] == "ok"


def test_degenerate_matches_count_as_no_geometry():
    import numpy as np
    import torch

    verifier = object.__new__(rr.GeometricVerifier)
    verifier.match_device = "cpu"
    pts = torch.tensor([[[10.0, 10.0]] * 20])  # Identical points: no fundamental matrix.
    verifier.matcher = lambda data: {"matches": [torch.tensor(np.c_[np.arange(20),
                                                                    np.arange(20)])]}
    out = verifier.inliers({"keypoints": pts}, {"keypoints": pts})
    assert out == {"matches": 20, "inliers": 0}
