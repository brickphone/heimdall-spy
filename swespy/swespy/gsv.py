"""Google Street View: rules shared by every part that touches it.

Recorded decision: Google Street View is approved for indexing, risk accepted by
Mattis Hüttenrauch on 2026-11-09. Mandatory rules that follow from it, enforced here:

- Descriptors only. Images are held in memory for encoding and discarded; never written
  to disk, logs or caches.
- Google imagery is never labelled with a Creative Commons or other licence. The source
  is always shown as ``ATTRIBUTION``.
- Every pull is logged (``PullLog``), without keys, image data or URLs.
- Street View data is a retrieval index only: never training, fine-tuning or pair mining.
- Billable caps (owner decision 2026-11-10), enforced by ``Budget`` from a reservation
  journal written *before* each billable request is sent (crash-safe):
  Street View tiles at most ``TILE_DAILY_CAP`` per Pacific day and ``TILE_TOTAL_CAP``
  across all runs; Static API images at most ``MONTHLY_IMAGE_CAP`` per billing month.
  Uncharged requests (session, panoIds, metadata) are logged but never counted.

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
from collections.abc import Callable
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
TILE_DAILY_CAP = 14_500  # Below Google's 15,000/day quota; days reset at midnight Pacific.
TILE_TOTAL_CAP = 220_000
# Monthly caps by billable type ("tile" has no monthly cap; see Budget for its caps).
MONTHLY_CAPS = {"image": MONTHLY_IMAGE_CAP, "tile": TILE_TOTAL_CAP}
UNCHARGED = ("session", "panoids", "metadata")
# Google Maps Platform billing months follow Pacific Time, so the cap does too.
BILLING_TZ = ZoneInfo("America/Los_Angeles")
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
# Google-captured panoramas (car, trekker, on-foot) carry "© Google" (Static API) or
# "From the Owner, Photo by: Google" (Map Tiles API); user-contributed photospheres carry
# the contributor's name instead.
GOOGLE_COPYRIGHT = re.compile(
    r"^(©\s*(\d{4}\s+)?Google( LLC)?|(From the Owner, )?Photo by: Google( LLC)?)$")
REQUEST_TYPES = ("session", "panoids", "metadata", "image", "tile")
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


def billing_day(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(BILLING_TZ).strftime("%Y-%m-%d")


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


class Budget:
    """Crash-safe billable-request budget.

    Every billable request is reserved with ``take`` *before* it is sent; the reservation
    is appended to a journal immediately, and the budget is always computed from that
    journal. A crash can therefore only over-count (never under-count) what was sent.
    """

    def __init__(self, journal: Path, now: Callable[[], datetime] | None = None):
        import threading

        self.journal = journal
        self.now = now or (lambda: datetime.now(UTC))
        self.lock = threading.Lock()
        self.tiles_by_day: dict[str, int] = {}
        self.tiles_total = 0
        self.images_by_month: dict[str, int] = {}
        if journal.exists():
            for line in journal.read_text().splitlines():
                if line:
                    r = json.loads(line)
                    self._add(r["type"], r["n"], datetime.fromisoformat(r["utc"]))

    def _add(self, kind: str, n: int, when: datetime) -> None:
        if kind == "tile":
            day = billing_day(when)
            self.tiles_by_day[day] = self.tiles_by_day.get(day, 0) + n
            self.tiles_total += n
        elif kind == "image":
            month = billing_month(when)
            self.images_by_month[month] = self.images_by_month.get(month, 0) + n
        else:
            raise ValueError(f"{kind} requests are not billable")

    def remaining(self, kind: str) -> int:
        now = self.now()
        if kind == "tile":
            return max(0, min(TILE_DAILY_CAP - self.tiles_by_day.get(billing_day(now), 0),
                              TILE_TOTAL_CAP - self.tiles_total))
        if kind == "image":
            return max(0, MONTHLY_IMAGE_CAP - self.images_by_month.get(billing_month(now), 0))
        raise ValueError(f"{kind} requests are not billable")

    def take(self, kind: str, n: int = 1) -> bool:
        """Reserve ``n`` billable requests; False (nothing reserved) if a cap would be hit."""
        with self.lock:
            if n > self.remaining(kind):
                return False
            when = self.now()
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a") as f:
                f.write(json.dumps({"utc": when.astimezone(UTC).isoformat(), "type": kind,
                                    "n": int(n)}) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._add(kind, n, when)
            return True

    def summary(self) -> dict[str, Any]:
        now = self.now()
        return {"tiles_today": self.tiles_by_day.get(billing_day(now), 0),
                "tiles_total": self.tiles_total,
                "tiles_total_remaining": TILE_TOTAL_CAP - self.tiles_total,
                "static_this_month": self.images_by_month.get(billing_month(now), 0),
                "billing_day_pacific": billing_day(now)}


def candidate_label(meta: dict[str, Any]) -> dict[str, Any]:
    """Display fields for a Street View candidate: fixed attribution, never a licence."""
    return {
        "source": SOURCE,
        "attribution": ATTRIBUTION,
        "license_url": None,
        "creator": "Google",
        "capture_date": meta.get("capture_date"),
    }
