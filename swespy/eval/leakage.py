"""Leakage guard between evaluation photos and gallery references.

Rules (applied to every candidate reference):

- ``eval_photo_itself``: the reference *is* an evaluation photo (same source ID or
  SHA-256 / perceptual-hash duplicate of one).
- ``near_in_space_and_time``: within 50 m **and** within 30 minutes of an eval photo's
  capture time. Eval photos whose time is only known to the day use a conservative
  window: the whole local (Europe/Stockholm) calendar day, widened by 30 minutes.
- ``same_photographer_session``: same photographer (case-insensitive name match
  across sources) on the same local calendar day.

``guard_dev`` reports which references matched which dev photo. ``guard_sealed``
reads the test manifest internally and returns **only counts per rule**: it never
prints, logs, returns or saves which references or locations matched.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from eval.geo_eval import haversine_m

RADIUS_M = 50.0
WINDOW = timedelta(minutes=30)
LOCAL = ZoneInfo("Europe/Stockholm")
RULES = ("eval_photo_itself", "near_in_space_and_time", "same_photographer_session")
DEV_MANIFEST = Path("data/public_eval/recent_linkoping_streets_v1/manifest.json")
TEST_MANIFEST = Path("data/paired_views/recent_pilot_v2/manifest.json")


def person(name: Any) -> str:
    return " ".join(str(name or "").casefold().split())


def local_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).astimezone(LOCAL).date().isoformat()


def eval_window(photo: dict[str, Any]) -> tuple[datetime, datetime, str]:
    """Return (start, end, local day) of the protected capture interval."""
    precision = photo.get("capture_time_precision") or (
        "day" if "capture_date" in photo else "exact")
    if precision == "day":
        day = datetime.fromisoformat(photo["capture_date"]).replace(tzinfo=LOCAL)
        return day - WINDOW, day + timedelta(days=1) + WINDOW, photo["capture_date"]
    moment = datetime.fromtimestamp(photo["captured_at"] / 1000, UTC)
    return moment - WINDOW, moment + WINDOW, local_day(photo["captured_at"])


def _match(ref: dict[str, Any], photo: dict[str, Any], window: tuple[datetime, datetime, str],
           distance_m: float) -> str | None:
    if (ref.get("source"), ref.get("source_id")) == (photo["source"], photo["source_id"]) or (
        ref.get("sha256") and ref.get("sha256") == photo.get("sha256")
    ) or (
        ref.get("phash") is not None and photo.get("phash") is not None
        and ((ref["phash"] ^ photo["phash"]).bit_count() <= 6
             or (ref["dhash"] ^ photo["dhash"]).bit_count() <= 5)
    ):
        return "eval_photo_itself"
    start, end, day = window
    taken = datetime.fromtimestamp(ref["captured_at"] / 1000, UTC)
    if distance_m <= RADIUS_M and start <= taken <= end:
        return "near_in_space_and_time"
    if person(ref.get("creator")) and person(ref.get("creator")) == person(photo.get("creator")) \
            and local_day(ref["captured_at"]) == day:
        return "same_photographer_session"
    return None


def _scan(refs: list[dict[str, Any]], photos: list[dict[str, Any]]
          ) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]]]:
    """Return kept references and (ref index, photo index, rule) for each exclusion."""
    if not refs:
        return [], []
    lats = np.array([r["lat"] for r in refs])
    lngs = np.array([r["lng"] for r in refs])
    first: dict[int, tuple[int, str]] = {}
    for j, photo in enumerate(photos):
        window = eval_window(photo)
        distances = haversine_m(photo["lat"], photo["lng"], lats, lngs)
        for i, ref in enumerate(refs):
            if i in first:
                continue
            rule = _match(ref, photo, window, float(distances[i]))
            if rule:
                first[i] = (j, rule)
    kept = [r for i, r in enumerate(refs) if i not in first]
    return kept, [(i, j, rule) for i, (j, rule) in sorted(first.items())]


def eval_photos(manifest: dict[str, Any], ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows = manifest.get("queries") or manifest.get("images") or []
    photos = []
    for row in rows:
        if ids is not None and row.get("image_id") not in ids:
            continue
        source_id = str(row["image_id"]).removeprefix("commons-")
        photos.append({**row, "source": "wikimedia_commons", "source_id": source_id})
    return photos


def guard_dev(refs: list[dict[str, Any]], manifest_path: Path = DEV_MANIFEST
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Visible dev-set guard: returns kept references and a detailed exclusion report."""
    photos = eval_photos(json.loads(manifest_path.read_text()))
    kept, hits = _scan(refs, photos)
    return kept, {
        "eval_photo_count": len(photos),
        "excluded": len(hits),
        "by_rule": {rule: sum(h[2] == rule for h in hits) for rule in RULES},
        "exclusions": [{"reference": refs[i].get("ref_id", refs[i].get("source_id")),
                        "dev_photo": photos[j]["image_id"], "rule": rule}
                       for i, j, rule in hits],
    }


def guard_sealed(refs: list[dict[str, Any]], manifest_path: Path = TEST_MANIFEST
                 ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sealed test-set guard: kept references plus counts per rule, nothing else."""
    manifest = json.loads(manifest_path.read_text())
    ids = set(manifest["fresh_commons_ids"])
    kept, hits = _scan(refs, eval_photos(manifest, ids))
    counts = Counter(rule for _, _, rule in hits)
    return kept, {"excluded": len(hits), **{rule: counts.get(rule, 0) for rule in RULES}}
