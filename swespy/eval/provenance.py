"""Provenance contract for every stored reference image (Part 3.4) and license allowlist."""

from __future__ import annotations

import math
from typing import Any

REQUIRED = ("source", "source_id", "license_url", "captured_at", "capture_year", "lat", "lng",
            "location_source")
# heading is required as a key but may be None when the source does not record it.
OPTIONAL_KNOWN = ("heading",)
SOURCES = ("mapillary", "kartaview", "wikimedia_commons", "self_collected")
LOCATION_SOURCES = ("mapillary_computed", "mapillary_raw", "gpx_interpolated",
                    "commons_exif_low_precision", "kartaview_gps")
# Licences that allow commercial use and derivatives (crops/resizes are derivatives).
COMMERCIAL_LICENSE_PREFIXES = (
    "https://creativecommons.org/licenses/by/",
    "https://creativecommons.org/licenses/by-sa/",
    "https://creativecommons.org/publicdomain/zero/",
    "https://creativecommons.org/publicdomain/mark/",
)
SELF_COLLECTED_LICENSE = "self-collected:owner-consent"


def license_allowed(url: str) -> bool:
    url = url.replace("http://", "https://")
    return url == SELF_COLLECTED_LICENSE or url.startswith(COMMERCIAL_LICENSE_PREFIXES)


def problems(row: dict[str, Any]) -> list[str]:
    """Return every provenance violation for one gallery row (empty list = valid)."""
    found = [f"missing:{k}" for k in (*REQUIRED, *OPTIONAL_KNOWN) if k not in row]
    found += [f"empty:{k}" for k in REQUIRED if k in row and row[k] in (None, "")]
    if row.get("source") not in SOURCES:
        found.append("unknown_source")
    if row.get("location_source") not in LOCATION_SOURCES:
        found.append("unknown_location_source")
    if row.get("license_url") and not license_allowed(str(row["license_url"])):
        found.append("license_not_commercial")
    try:
        lat, lng = float(row["lat"]), float(row["lng"])
        if not (math.isfinite(lat) and math.isfinite(lng)):
            found.append("bad_coordinates")
    except (KeyError, TypeError, ValueError):
        found.append("bad_coordinates")
    heading = row.get("heading")
    if heading is not None and not (isinstance(heading, int | float) and 0 <= heading < 360):
        found.append("bad_heading")
    return found


def from_legacy_mapillary(ref: dict[str, Any]) -> dict[str, Any]:
    """Map a pre-contract Mapillary gallery row (raw geometry) onto the provenance fields."""
    from datetime import UTC, datetime

    return {**ref, "source": "mapillary", "source_id": ref["image_id"],
            "heading": ref.get("heading"), "location_source": "mapillary_raw",
            "capture_year": datetime.fromtimestamp(ref["captured_at"] / 1000, UTC).year}
