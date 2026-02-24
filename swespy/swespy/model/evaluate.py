"""Evaluation metrics for geolocation inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate haversine distance in kilometers."""
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def evaluate_model(model: Any, test_dataloader: Any, cell_to_latlong: dict[int, tuple[float, float]]) -> dict[str, float]:
    """Evaluate model with geolocation benchmark metrics."""
    model.eval()
    errors: list[float] = []

    with torch.no_grad():
        for pixel_values, true_lat, true_lng in test_dataloader:
            logits = model(pixel_values)
            pred_cells = logits.argmax(dim=1).detach().cpu().numpy()
            true_lat_np = np.array(true_lat)
            true_lng_np = np.array(true_lng)

            for idx, cell_idx in enumerate(pred_cells):
                pred_lat, pred_lng = cell_to_latlong[int(cell_idx)]
                error = haversine_distance(
                    float(true_lat_np[idx]),
                    float(true_lng_np[idx]),
                    float(pred_lat),
                    float(pred_lng),
                )
                errors.append(error)

    if not errors:
        return {
            "median_error_km": 0.0,
            "mean_error_km": 0.0,
            "pct_within_25km": 0.0,
            "pct_within_100km": 0.0,
            "pct_within_200km": 0.0,
        }

    errors_np = np.array(errors)
    return {
        "median_error_km": float(np.median(errors_np)),
        "mean_error_km": float(np.mean(errors_np)),
        "pct_within_25km": float(np.mean(errors_np <= 25.0)),
        "pct_within_100km": float(np.mean(errors_np <= 100.0)),
        "pct_within_200km": float(np.mean(errors_np <= 200.0)),
    }
