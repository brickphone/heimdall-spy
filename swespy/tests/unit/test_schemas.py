"""Cosine roundoff must not disable rejection of invalid result scores."""

import pytest
from pydantic import ValidationError

from swespy.api.schemas import MatchResult


@pytest.mark.parametrize("score", [-0.1, 1.01, float("nan"), float("inf")])
def test_invalid_similarity_still_rejected(score: float) -> None:
    with pytest.raises(ValidationError):
        MatchResult(lat=58.41, lng=15.62, s2_cell="test", similarity_score=score)
