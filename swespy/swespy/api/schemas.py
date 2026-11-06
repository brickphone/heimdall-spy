"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MatchResult(BaseModel):
    """A single vector match candidate."""

    lat: float
    lng: float
    similarity_score: float = Field(ge=0.0, le=1.0)
    s2_cell: str

    @field_validator("similarity_score", mode="before")
    @classmethod
    def clamp_cosine_roundoff(cls, value: object) -> object:
        """Allow tiny floating-point overshoot from cosine search results."""
        if isinstance(value, float) and 1.0 < value <= 1.0 + 1e-6:
            return 1.0
        return value


class PredictResponse(BaseModel):
    """Prediction response payload."""

    predicted_lat: float
    predicted_lng: float
    confidence: float = Field(ge=0.0, le=1.0)
    top_matches: list[MatchResult]
    processing_time_ms: float
