"""S2 cell utilities."""

from __future__ import annotations

import pandas as pd
from s2geometry import pywraps2 as s2


def lat_lng_to_cell_id(lat: float, lng: float, level: int) -> str:
    """Convert a coordinate to an S2 cell token at a given level."""
    lat_lng = s2.S2LatLng.FromDegrees(lat, lng)
    cell_id = s2.S2CellId.FromLatLng(lat_lng).parent(level)
    return cell_id.ToToken()


def cell_id_to_lat_lng(cell_token: str) -> tuple[float, float]:
    """Return center point (lat, lng) for an S2 cell token."""
    cell_id = s2.S2CellId.FromToken(cell_token)
    center = cell_id.ToLatLng()
    return center.lat().degrees(), center.lng().degrees()


def assign_cells_to_dataframe(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """Assign S2 cells to dataframe rows using `lat`/`lng` columns."""
    if not {"lat", "lng"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'lat' and 'lng' columns")

    enriched = df.copy()
    enriched["s2_cell"] = enriched.apply(
        lambda row: lat_lng_to_cell_id(float(row["lat"]), float(row["lng"]), level),
        axis=1,
    )
    return enriched
