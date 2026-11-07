"""Acquire recent Commons photos and review/freeze them; evaluate with eval.dev_eval."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from eval.common import duplicate, signature, write_new
from eval.expand_linkoping import BBOX, DEFAULT_ROOT
from eval.metrics import distance_m
from swespy.model.megaloc import file_hash

ROOT = Path("data/public_eval/recent_linkoping_streets_v1")
INDEX = Path("data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/gallery-index.json")
API = "https://commons.wikimedia.org/w/api.php"
SINCE, UNTIL = date(2023, 11, 6), date(2026, 11, 6)
UA = ("HeimdallSpyResearchBot/0.1 (https://github.com/brickphone/heimdall-spy; "
      "local photo evaluation) httpx/" + httpx.__version__)
SCENE_EXCLUSIONS = (
    "interior", "interiör", "portrait", "presskonferens", "repetition", "repetion",
    "aerial", "flygfoto", "floor plan", "historical map", "cosplay", "cosplayer",
    "närcon", "ncs24", "flygdagarna", "melodifestivalen", "rehears", "concert",
    "konsert", "läktarorgel", "kororgel", "church organ", "painting", "olja på duk",
)


def plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def taken_date(value: str) -> date | None:
    """Require a full capture day; never substitute an upload or modification date."""
    match = re.fullmatch(r"(\d{4})[-:](\d{2})[-:](\d{2})(?:[ T].*)?", plain(value))
    try:
        return date(*map(int, match.groups())) if match else None
    except ValueError:
        return None


def candidate(page: dict[str, Any], since: date, until: date) -> dict[str, Any] | None:
    """Filter dates, license, bounds and obvious non-photo/indoor metadata before inference."""
    try:
        info = page["imageinfo"][0]
        ext = {k: str(v["value"]) for k, v in info["extmetadata"].items()}
        captured = taken_date(ext.get("DateTimeOriginal", ""))
        if captured is None or not since <= captured <= until:
            return None
        if info["mime"] != "image/jpeg" or not 0.3 <= info["width"]/info["height"] <= 3:
            return None
        license_url = ext.get("LicenseUrl", "").replace("http://", "https://")
        if license_url.startswith("//"):
            license_url = "https:" + license_url
        if not license_url.startswith(("https://creativecommons.org/licenses/by/",
                                       "https://creativecommons.org/licenses/by-sa/",
                                       "https://creativecommons.org/publicdomain/")):
            return None
        creator = plain(ext.get("Artist", ""))
        if not creator:
            return None
        # Explicit GPS fields represent the camera location. Do not use an object's
        # coordinates or geocode a landmark name as the camera answer key.
        lat, lng = float(ext["GPSLatitude"]), float(ext["GPSLongitude"])
        west, south, east, north = BBOX
        if not (math.isfinite(lat) and math.isfinite(lng) and
                west <= lng <= east and south <= lat <= north):
            return None
        description = plain(ext.get("ImageDescription", ""))
        words = (page["title"] + " " + description).casefold()
        if any(word in words for word in SCENE_EXCLUSIONS):
            return None
        return {
            "image_id": f"commons-{page['pageid']}", "title": page["title"],
            "creator": creator, "lat": lat, "lng": lng,
            "capture_date": captured.isoformat(), "capture_date_source": "DateTimeOriginal",
            "captured_at": int(datetime.combine(captured, datetime.min.time(), UTC)
                               .timestamp()*1000),
            "coordinate_source": "Commons GPSLatitude/GPSLongitude; not independently verified",
            "source_url": info["descriptionurl"], "provider": "Wikimedia Commons",
            "license_url": license_url, "license_name": plain(ext.get("LicenseShortName", "")),
            "description": description, "download_url": info.get("thumburl", info["url"]),
        }
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None


def status(root: Path, **data: Any) -> None:
    path = root/"acquisition-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_at": datetime.now(UTC).isoformat(), **data},
                                    indent=2))
    temporary.replace(path)


async def request(client: httpx.AsyncClient, params: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(4):
        started = time.monotonic()
        response = await client.get(API, params={"action": "query", "format": "json",
                                                "maxlag": 5, **params})
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            await asyncio.sleep(2**attempt)
            continue
        response.raise_for_status()  # In particular, do not retry or evade HTTP 403.
        payload = response.json()
        if "error" in payload:
            raise ValueError(f"Commons API error: {payload['error'].get('code')}")
        if time.monotonic() - started > 1:
            # Wikimedia asks clients to pause after expensive Action API requests.
            await asyncio.sleep(5)
        return payload
    raise RuntimeError("Commons request retries exhausted")


async def acquire(root: Path, since: date, until: date, target: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"provider": "Wikimedia Commons only; no paid service", "since": since.isoformat(),
                "until": until.isoformat(),
                "date_field": "DateTimeOriginal; upload dates forbidden",
                "bbox": list(BBOX), "target": target, "metadata_limit": 2500,
                "download_limit": 200, "selection": "SHA256 ID order; <=15/creator; "
                ">=25m between selected query camera labels; metadata filtering before inference",
                "gallery_manifest_sha256": file_hash(DEFAULT_ROOT/"manifest.json")}
    protocol["scene_exclusions"] = list(SCENE_EXCLUSIONS)
    if (root/"protocol.json").exists():
        if json.loads((root/"protocol.json").read_text()) != protocol:
            raise ValueError("Existing public-photo acquisition protocol differs; use a new root")
    else:
        write_new(root/"protocol.json", protocol)
    if (root/"selection.json").exists():
        print("Selection exists; inspect review.html and prepare review.json before freezing")
        return
    status(root, state="acquiring", photos_frozen=False, user_agent=UA)
    cache = root/"metadata"
    cache.mkdir(exist_ok=True)
    rejected: Counter[str] = Counter()
    async with httpx.AsyncClient(headers={"User-Agent": UA}, timeout=45,
                                 follow_redirects=True) as client:
        pages: dict[str, Any] = {}
        centers = [(58.41, 15.62), (58.37, 15.55), (58.45, 15.55),
                   (58.37, 15.70), (58.45, 15.70)]
        try:
            for i, (lat, lng) in enumerate(centers):
                path = cache/f"geosearch-{i}.json"
                if path.exists():
                    found = json.loads(path.read_text())
                else:
                    found = await request(client, {"list": "geosearch", "gscoord": f"{lat}|{lng}",
                        "gsradius": 10000, "gsnamespace": 6, "gslimit": 500})
                    write_new(path, found)
                for item in found.get("query", {}).get("geosearch", []):
                    pages[str(item["pageid"])] = item
                await asyncio.sleep(0.25)
            ids = sorted(pages)
            entries = []
            for start in range(0, len(ids), 50):
                batch = ids[start:start+50]
                path = cache/f"info-{hashlib.sha256('|'.join(batch).encode()).hexdigest()}.json"
                if path.exists():
                    data = json.loads(path.read_text())
                else:
                    data = await request(client, {"pageids": "|".join(batch), "prop": "imageinfo",
                        "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 1280})
                    write_new(path, data)
                for page in data.get("query", {}).get("pages", {}).values():
                    row = candidate(page, since, until)
                    if row:
                        entries.append(row)
                    else:
                        rejected["metadata_date_license_gps_or_scene"] += 1
                await asyncio.sleep(0.25)
            entries.sort(key=lambda r: hashlib.sha256(r["image_id"].encode()).hexdigest())
            refs = json.loads((DEFAULT_ROOT/"manifest.json").read_text())["references"]
            selected: list[dict[str, Any]] = []
            counts: Counter[str] = Counter()
            downloads = 0
            folder = root/"photos"
            folder.mkdir(exist_ok=True)
            for row in entries:
                if len(selected) >= target or downloads >= 200:
                    break
                if counts[row["creator"]] >= 15 or any(
                    distance_m(row["lat"], row["lng"], q["lat"], q["lng"]) < 25 for q in selected
                ):
                    rejected["photographer_cap_or_camera_spacing"] += 1
                    continue
                path = folder/f"{row['image_id']}.jpg"
                if not path.exists():
                    response = await client.get(row["download_url"])
                    response.raise_for_status()
                    raw = root/"download.tmp"
                    raw.write_bytes(response.content)
                    try:
                        with Image.open(raw) as image:
                            pixels = ImageOps.exif_transpose(image).convert("RGB")
                            pixels.save(path, format="JPEG", quality=95)  # No EXIF/XMP labels.
                    finally:
                        raw.unlink(missing_ok=True)
                downloads += 1
                sig = signature(path)
                if duplicate(sig, refs+selected):
                    rejected["gallery_or_query_duplicate_screen"] += 1
                    continue
                selected.append({**{k: v for k, v in row.items() if k != "download_url"},
                                 **sig, "path": str(path.relative_to(root))})
                counts[row["creator"]] += 1
                print(f"Accepted recent public candidates: {len(selected)}/{target}", flush=True)
                await asyncio.sleep(0.25)
        except httpx.HTTPStatusError as exc:
            status(root, state="blocked", http_status=exc.response.status_code,
                   reason=exc.response.text[:250], photos_frozen=False)
            raise
    if not selected:
        status(root, state="no_eligible_photos", metadata_count=len(pages),
               rejections=dict(rejected), photos_frozen=False)
        raise ValueError("No eligible recent photos; "
                         "dates/GPS/license requirements were not relaxed")
    write_new(root/"selection.json", {"queries": selected, "rejections": dict(rejected),
                                     "metadata_count": len(pages), "review_required": True})
    cards = "".join(f"<article><img src='{html.escape(q['path'], quote=True)}'>"
                    f"<p>{html.escape(q['image_id'])} · {q['capture_date']}<br>"
                    f"{html.escape(q['title'])}<br>{html.escape(q['creator'])}</p></article>"
                    for q in selected)
    (root/"review.html").write_text("<!doctype html><meta charset='utf-8'>"
        "<title>Recent Linköping photo review</title><style>body{font:16px system-ui}"
        "article{display:inline-block;width:300px;vertical-align:top;margin:10px}"
        "img{width:300px;height:220px;object-fit:contain}</style>"
        "<h1>Review before running predictions</h1>"+cards)
    status(root, state="awaiting_image_review", selected_count=len(selected), photos_frozen=False)


def freeze(root: Path) -> None:
    """Require explicit image review; publish an immutable test set before any inference."""
    protocol = json.loads((root/"protocol.json").read_text())
    if file_hash(DEFAULT_ROOT/"manifest.json") != protocol["gallery_manifest_sha256"]:
        raise ValueError("Reference gallery changed after public-photo selection")
    selection = json.loads((root/"selection.json").read_text())
    review = json.loads((root/"review.json").read_text())
    if not review.get("reviewer") or not review.get("reviewed_at"):
        raise ValueError("Record the reviewer and review time")
    chosen = set(review["include_ids"])
    known = {q["image_id"] for q in selection["queries"]}
    if not chosen or not chosen <= known or len(chosen) != len(review["include_ids"]):
        raise ValueError("Reviewed IDs must be unique, nonempty and from the acquired selection")
    queries = [q for q in selection["queries"] if q["image_id"] in chosen]
    for q in queries:
        captured = taken_date(q["capture_date"])
        if captured is None or not date.fromisoformat(protocol["since"]) <= captured <= \
                date.fromisoformat(protocol["until"]) or file_hash(root/q["path"]) != q["sha256"]:
            raise ValueError("Reviewed photo date or bytes no longer agree")
    write_new(root/"manifest.json", {"kind": "recent_public_linkoping_heldout", **protocol,
        "query_count": len(queries), "queries": queries,
        "review_sha256": file_hash(root/"review.json"),
        "limitations": "Commons landmarks/photographers may be overrepresented; GPS unverified; "
                        "pretraining overlap is unknown"})
    status(root, state="frozen", query_count=len(queries), photos_frozen=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Evaluation moved to eval.dev_eval, which uses the shared API inference path.
    parser.add_argument("action", choices=["acquire", "freeze"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--since", type=date.fromisoformat, default=SINCE)
    parser.add_argument("--until", type=date.fromisoformat, default=UNTIL)
    parser.add_argument("--target", type=int, default=75)
    args = parser.parse_args()
    if args.since > args.until or args.until > datetime.now(UTC).date() or \
            not 1 <= args.target <= 100:
        parser.error("Use a valid past date interval and target between 1 and 100")
    try:
        if args.action == "acquire":
            asyncio.run(acquire(args.root, args.since, args.until, args.target))
        else:
            freeze(args.root)
    except httpx.HTTPStatusError as exc:
        print(f"Commons HTTP {exc.response.status_code}: acquisition incomplete; "
              "no test set frozen or evaluation performed", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
