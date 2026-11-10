"""Google Street View: rules shared by every part that touches it.

Recorded decision: Google Street View is approved for indexing, risk accepted by
Mattis Hüttenrauch on 2026-11-09. Mandatory rules that follow from it, enforced here:

- Descriptors only. Images are held in memory for encoding and discarded; never written
  to disk, logs or caches.
- Google imagery is never labelled with a Creative Commons or other licence. The source
  is always shown as ``ATTRIBUTION``.
- Every pull is logged (``PullLog``), without keys, image data or URLs.
- Street View data is a retrieval index only: never training, fine-tuning or pair mining.
- At most ``MONTHLY_IMAGE_CAP`` image requests per billing month unless the owner
  explicitly approves paid usage. Metadata requests are free and not capped.

The API key comes from the environment (``GOOGLE_MAPS_API_KEY``) or the macOS Keychain
(service ``google_maps_key``). It is never stored in the repo, printed or logged, and
it never appears in an exception message (``GsvRequestError`` carries no URL).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ATTRIBUTION = "Google Street View, © Google"
SOURCE = "gsv"
KEY_ENV = "GOOGLE_MAPS_API_KEY"
KEYCHAIN_SERVICE = "google_maps_key"
MONTHLY_IMAGE_CAP = 9_500
# Billable request types and their monthly caps. Street View tiles (Map Tiles API) are a
# separate SKU; they stay at 0 (not allowed) until the owner approves a tile cap.
MONTHLY_CAPS = {"image": MONTHLY_IMAGE_CAP, "tile": 0}
# Google Maps Platform billing months follow Pacific Time, so the cap does too.
BILLING_TZ = ZoneInfo("America/Los_Angeles")
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
# Google-captured panoramas (car, trekker, on-foot) carry "© Google"; user-contributed
# photospheres carry the contributor's name instead.
GOOGLE_COPYRIGHT = re.compile(r"^©\s*(\d{4}\s+)?Google( LLC)?$")
REQUEST_TYPES = ("metadata", "image", "tile")
PULL_LOG_FIELDS = (
    "utc", "bbox", "request_type", "request_count", "success_count",
    "cumulative_month", "priority",
)

# httpx logs request URLs (which contain the key) at INFO; keep it quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class MissingKey(RuntimeError):
    """No API key in the environment or the Keychain."""


class CapReached(RuntimeError):
    """The monthly image-request cap would be exceeded."""


class GsvRequestError(RuntimeError):
    """A failed Google request. Deliberately carries no URL or parameters."""


def api_key() -> str:
    """The key from the environment or the Keychain. Never print or log the result."""
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        out = None
    if out and out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    raise MissingKey(
        f"No Google Maps key: set {KEY_ENV} or add a Keychain item with service "
        f"'{KEYCHAIN_SERVICE}'")


def is_google_captured(copyright_text: str | None) -> bool:
    return bool(copyright_text) and bool(GOOGLE_COPYRIGHT.match(copyright_text.strip()))


def billing_month(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(BILLING_TZ).strftime("%Y-%m")


def redact(text: str, key: str) -> str:
    """Remove the key (and any key=... parameter) from text before it is shown anywhere."""
    if key:
        text = text.replace(key, "[key]")
    return re.sub(r"key=[^&\s]+", "key=[key]", text)


@dataclass
class PullLog:
    """Append-only log, one row per batch. No keys, image data or URLs."""

    path: Path

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def used_this_month(self, request_type: str = "image", now: datetime | None = None) -> int:
        month = billing_month(now)
        return sum(
            r["request_count"] for r in self.rows()
            if r["request_type"] == request_type
            and billing_month(datetime.fromisoformat(r["utc"])) == month
        )

    def remaining(self, request_type: str = "image", now: datetime | None = None) -> int:
        cap = MONTHLY_CAPS[request_type]
        return max(0, cap - self.used_this_month(request_type, now))

    def remaining_images(self, now: datetime | None = None) -> int:
        return self.remaining("image", now)

    def reserve(self, n: int, request_type: str = "image", now: datetime | None = None) -> None:
        """Raise before sending ``n`` billable requests that would cross the cap."""
        if request_type not in MONTHLY_CAPS:
            raise ValueError(f"{request_type} requests are not billable")
        if n > self.remaining(request_type, now):
            raise CapReached(
                f"{n} {request_type} requests would exceed the monthly cap of "
                f"{MONTHLY_CAPS[request_type]} "
                f"({self.used_this_month(request_type, now)} used in {billing_month(now)})")

    def append(
        self, *, request_type: str, request_count: int, success_count: int,
        bbox: tuple[float, float, float, float] | None, priority: int | None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if request_type not in REQUEST_TYPES:
            raise ValueError(f"Unknown request type {request_type}")
        when = now or datetime.now(UTC)
        previous = self.used_this_month(request_type, when)
        row = {
            "utc": when.astimezone(UTC).isoformat(),
            "bbox": [round(v, 6) for v in bbox] if bbox else None,
            "request_type": request_type,
            "request_count": int(request_count),
            "success_count": int(success_count),
            "cumulative_month": previous + int(request_count),
            "priority": priority,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row


def candidate_label(meta: dict[str, Any]) -> dict[str, Any]:
    """Display fields for a Street View candidate: fixed attribution, never a licence."""
    return {
        "source": SOURCE,
        "attribution": ATTRIBUTION,
        "license_url": None,
        "creator": "Google",
        "capture_date": meta.get("capture_date"),
    }
