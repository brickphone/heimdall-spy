"""API request/response schemas."""

from __future__ import annotations

from typing import Any

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
    source: str | None = None
    capture_year: int | None = None
    inliers: int | None = None
    score: float | None = None
    variant: str | None = None  # The search variant that retrieved this reference.

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
    confidence_state: str = "unavailable"
    confidence_reasons: list[str] = []
    top_matches: list[MatchResult]
    processing_time_ms: float
    region: dict[str, float] | None = None
    # Automatic region search: the variant whose verified match won ("whole", "manual",
    # "masked" or "crop:<name>"), the part of the photo it covers (normalised), the share
    # of the photo covered by people, and an alternative region to try.
    searched_variant: str = "whole"
    searched_region: dict[str, float] | None = None
    person_share: float | None = None
    alternative_region: dict[str, float] | None = None
    variants: list[dict[str, Any]] = []
    timings_ms: dict[str, float] = {}
    versions: dict[str, Any] = {}


class AnalyzeResponse(BaseModel):
    """People in the photo and the suggested region; no retrieval."""

    person_share: float | None
    suggested_region: dict[str, float] | None
    suggest: bool  # People cover more than 30%: pre-draw the suggested region.
    variants: list[dict[str, Any]]
    processing_time_ms: float
