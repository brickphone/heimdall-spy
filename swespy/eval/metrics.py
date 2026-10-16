"""Geographic retrieval metrics independent of inference implementation."""

from __future__ import annotations

import math
import statistics
from typing import Any

from eval.dataset import VIEWPOINTS, Photo


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return great-circle distance in metres using a 6,371 km Earth radius."""
    a = (
        math.sin(math.radians(lat2 - lat1) / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    )
    a = max(0.0, min(1.0, a))
    return 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def observation(photo: Photo, prediction: dict[str, Any], hit_radius_m: float) -> dict[str, Any]:
    """Measure pin error and whether any returned alternative is within the hit radius."""
    lat, lng = float(prediction["predicted_lat"]), float(prediction["predicted_lng"])
    confidence = float(prediction["confidence"])
    if not (math.isfinite(lat) and math.isfinite(lng) and -90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("Inference returned invalid coordinates")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Inference returned invalid confidence")
    matches = prediction["top_matches"]
    for match in matches:
        match_lat, match_lng = float(match["lat"]), float(match["lng"])
        if not (
            math.isfinite(match_lat)
            and math.isfinite(match_lng)
            and -90 <= match_lat <= 90
            and -180 <= match_lng <= 180
        ):
            raise ValueError("Inference returned invalid alternative coordinates")
    candidate_distances = [distance_m(photo.lat, photo.lng, m["lat"], m["lng"]) for m in matches]
    if any(not math.isfinite(d) for d in candidate_distances):
        raise ValueError("Inference returned invalid alternative coordinates")
    return {
        "viewpoint_type": photo.viewpoint_type,
        "error_m": distance_m(photo.lat, photo.lng, lat, lng),
        "top_k_hit": any(d <= hit_radius_m for d in candidate_distances),
        "candidate_count": len(matches),
        "confidence": confidence,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return overall and per-viewpoint metrics; empty groups have null metrics."""

    def group(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        return {
            "count": n,
            "median_error_m": statistics.median(r["error_m"] for r in items) if n else None,
            **{
                f"pct_within_{radius}m": 100 * sum(r["error_m"] <= radius for r in items) / n
                if n
                else None
                for radius in (100, 500, 2000)
            },
            "top_k_hit_pct": 100 * sum(r["top_k_hit"] for r in items) / n if n else None,
        }

    return {
        "overall": group(rows),
        "by_viewpoint": {
            v: group([r for r in rows if r["viewpoint_type"] == v]) for v in VIEWPOINTS
        },
    }


def print_table(metrics: dict[str, Any]) -> None:
    """Print counts, median metres, threshold percentages and geographic top-k hits."""
    print(
        f"{'viewpoint':<14} {'n':>5} {'median m':>12} {'<=100m %':>10} "
        f"{'<=500m %':>10} {'<=2km %':>10} {'top-k %':>10}"
    )
    for name, group in [("overall", metrics["overall"]), *metrics["by_viewpoint"].items()]:
        values = [
            group[key]
            for key in (
                "median_error_m",
                "pct_within_100m",
                "pct_within_500m",
                "pct_within_2000m",
                "top_k_hit_pct",
            )
        ]
        formatted = " ".join(f"{v:>10.2f}" if v is not None else f"{'n/a':>10}" for v in values)
        print(f"{name:<14} {group['count']:>5}   {formatted}")
