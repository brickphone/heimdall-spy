"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchResult(BaseModel):
    """A single vector match candidate."""

    lat: float
    lng: float
    similarity_score: float = Field(ge=0.0, le=1.0)
    s2_cell: str


class PredictResponse(BaseModel):
    """Prediction response payload."""

    predicted_lat: float
    predicted_lng: float
    confidence: float = Field(ge=0.0, le=1.0)
    top_matches: list[MatchResult]
    processing_time_ms: float
