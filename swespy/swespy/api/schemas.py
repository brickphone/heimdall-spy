"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MatchResult(BaseModel):
    """One ranked gallery reference."""

    rank: int = Field(ge=1)
    image_id: str
    lat: float
    lng: float
    similarity_score: float = Field(ge=-1.0, le=1.0)
    creator: str | None = None
    source_url: str | None = None
    license_url: str | None = None

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
    prediction_method: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_status: str
    top_matches: list[MatchResult]
    processing_time_ms: float
