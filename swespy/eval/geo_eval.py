"""Model-independent evaluation statistics: confidence intervals, diagnostics and baselines.

Everything here works on coordinates and ranked candidate lists only, so the same
functions score retrieval, re-ranking and constant baselines identically.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

import numpy as np

RADII_M = (25, 100, 500, 2000)
TOP_KS = (1, 5, 20, 50)
DIAGNOSTIC_RADII_M = (100, 500)
COVERAGE_RADIUS_M = 100
RETRIEVAL_DEPTH = 50
# Flag a development set as clustered when one photographer or one 1 km disc holds
# more than this share of the photos.
CLUSTER_SHARE_THRESHOLD = 0.25
EARTH_RADIUS_M = 6_371_000


def wilson_interval(successes: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        raise ValueError("Wilson interval needs at least one trial")
    if not 0 <= successes <= n:
        raise ValueError("Successes must lie between 0 and n")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(successes: float, n: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, n)
    return {"count": successes, "n": n, "rate": successes / n, "ci95": [low, high]}


def haversine_m(lat: float, lng: float, lats: np.ndarray, lngs: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance from one point to many, in metres."""
    phi1, phi2 = math.radians(lat), np.radians(lats)
    a = (np.sin((phi2 - phi1) / 2) ** 2
         + math.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lngs - lng) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def failure_type(
    nearest_reference_m: float,
    ranked_distances_m: list[float],
    coverage_m: float = COVERAGE_RADIUS_M,
    depth: int = RETRIEVAL_DEPTH,
) -> str:
    """Classify a query by where the true location was lost.

    - no_coverage: no gallery reference within ``coverage_m`` of the truth.
    - success: the top-1 reference lies within ``coverage_m``.
    - ranking: a reference within ``coverage_m`` is in the top ``depth`` but not top 1.
    - retrieval: covered, but no reference within ``coverage_m`` in the top ``depth``.
    """
    if not ranked_distances_m:
        raise ValueError("At least one ranked candidate is required")
    if nearest_reference_m > coverage_m:
        return "no_coverage"
    if ranked_distances_m[0] <= coverage_m:
        return "success"
    if any(d <= coverage_m for d in ranked_distances_m[:depth]):
        return "ranking"
    return "retrieval"


def diagnose(
    true_lat: float,
    true_lng: float,
    predicted: tuple[float, float],
    ranked_coords: list[tuple[float, float]],
    nearest_reference_m: float,
) -> dict[str, Any]:
    """Per-photo error and coverage/ranking diagnostics."""
    if not ranked_coords:
        raise ValueError("At least one ranked candidate is required")
    lats = np.array([c[0] for c in ranked_coords])
    lngs = np.array([c[1] for c in ranked_coords])
    ranked = haversine_m(true_lat, true_lng, lats, lngs).tolist()
    error = float(haversine_m(true_lat, true_lng, np.array([predicted[0]]),
                              np.array([predicted[1]]))[0])
    return {
        "error_m": error,
        "nearest_reference_m": nearest_reference_m,
        "rank_of_first_within_m": {
            str(r): next((i + 1 for i, d in enumerate(ranked) if d <= r), None)
            for r in DIAGNOSTIC_RADII_M
        },
        "true_in_top_k": {
            f"{r}m": {str(k): any(d <= r for d in ranked[:k]) for k in TOP_KS}
            for r in DIAGNOSTIC_RADII_M
        },
        "failure_type": failure_type(nearest_reference_m, ranked),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-photo rows into rates with Wilson 95% CIs."""
    n = len(rows)
    if not n:
        raise ValueError("Cannot summarize an empty evaluation")
    types = Counter(r["failure_type"] for r in rows)
    return {
        "query_count": n,
        "median_error_m": statistics.median(r["error_m"] for r in rows),
        "within": {f"{radius}m": rate(sum(r["error_m"] <= radius for r in rows), n)
                   for radius in RADII_M},
        "coverage": {f"{radius}m": rate(sum(r["nearest_reference_m"] <= radius for r in rows), n)
                     for radius in (*RADII_M[:3], 50)},
        "true_in_top_k": {
            f"{r}m": {str(k): rate(sum(row["true_in_top_k"][f"{r}m"][str(k)] for row in rows), n)
                      for k in TOP_KS}
            for r in DIAGNOSTIC_RADII_M
        },
        "failure_type": {t: rate(types.get(t, 0), n)
                         for t in ("success", "ranking", "retrieval", "no_coverage")},
    }


def majority_location(refs: list[dict[str, Any]], cell_m: int = 1000) -> dict[str, Any]:
    """The mean position of the most populated gallery cell; uses no evaluation labels."""
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in refs:
        cells.setdefault((int(r["easting"] // cell_m), int(r["northing"] // cell_m)), []).append(r)
    key, members = max(cells.items(), key=lambda kv: (len(kv[1]), kv[0]))
    return {"lat": statistics.fmean(r["lat"] for r in members),
            "lng": statistics.fmean(r["lng"] for r in members),
            "cell": list(key), "cell_m": cell_m, "reference_count": len(members)}


def constant_baseline(queries: list[dict[str, Any]], lat: float, lng: float) -> dict[str, Any]:
    errors = [float(haversine_m(q["lat"], q["lng"], np.array([lat]), np.array([lng]))[0])
              for q in queries]
    n = len(errors)
    return {"median_error_m": statistics.median(errors),
            "within": {f"{radius}m": rate(sum(e <= radius for e in errors), n)
                       for radius in RADII_M}}


def random_reference_baseline(
    queries: list[dict[str, Any]], refs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Exact expectation of guessing a uniformly random gallery reference.

    The expected success count is fractional; its Wilson interval uses that count
    directly, which approximates the interval of a single random draw.
    """
    lats = np.array([r["lat"] for r in refs])
    lngs = np.array([r["lng"] for r in refs])
    per_query = [haversine_m(q["lat"], q["lng"], lats, lngs) for q in queries]
    n = len(queries)
    return {
        "median_error_m": statistics.median(float(np.median(d)) for d in per_query),
        "within": {f"{radius}m": rate(float(sum(np.mean(d <= radius) for d in per_query)), n)
                   for radius in RADII_M},
        "note": "Exact expectation over all references; count is fractional",
    }


def distribution(queries: list[dict[str, Any]], radius_m: float = 1000) -> dict[str, Any]:
    """Photographer and spatial concentration of an evaluation set, with explicit flags."""
    n = len(queries)
    creators = Counter(q["creator"] for q in queries)
    lats = np.array([q["lat"] for q in queries])
    lngs = np.array([q["lng"] for q in queries])
    neighbours = [int(np.sum(haversine_m(q["lat"], q["lng"], lats, lngs) <= radius_m))
                  for q in queries]
    densest = int(np.argmax(neighbours))
    top_creator, top_count = creators.most_common(1)[0]
    share_disc = neighbours[densest] / n
    flags = []
    top3 = sum(c for _, c in creators.most_common(3))
    if top_count / n > CLUSTER_SHARE_THRESHOLD:
        flags.append(f"photographer_concentration: {top_creator} took {top_count}/{n}")
    if top3 / n > 2 * CLUSTER_SHARE_THRESHOLD:
        flags.append(f"photographer_concentration: three photographers took {top3}/{n}")
    if share_disc > CLUSTER_SHARE_THRESHOLD:
        flags.append(f"spatial_clustering: {neighbours[densest]}/{n} photos within "
                     f"{radius_m:.0f} m of one point")
    return {
        "photo_count": n,
        "photographer_count": len(creators),
        "photos_per_photographer": dict(creators.most_common()),
        "largest_photographer_share": top_count / n,
        "top3_photographer_share": top3 / n,
        "effective_photographers": 1 / sum((c / n) ** 2 for c in creators.values()),
        "densest_point": {"lat": float(lats[densest]), "lng": float(lngs[densest]),
                          "image_id": queries[densest].get("image_id")},
        f"share_within_{radius_m:.0f}m_of_densest_point": share_disc,
        "median_distance_to_densest_point_m": float(np.median(
            haversine_m(float(lats[densest]), float(lngs[densest]), lats, lngs))),
        "clustering_threshold_share": CLUSTER_SHARE_THRESHOLD,
        "clustered": bool(flags),
        "flags": flags,
    }
