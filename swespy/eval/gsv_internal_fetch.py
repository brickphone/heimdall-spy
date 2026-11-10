"""Street View internal-backend pilot, Part 2 (runs on the pod): fetch, render, encode.

Recorded decision: using the unofficial Google Street View backend via the open-source
``streetlevel`` library is approved for a pilot. Risk and responsibility accepted by
Mattis Hüttenrauch on 2026-11-10.

Rules enforced here:

- Images exist in memory only. Tiles are decoded from bytes, views are rendered from a
  numpy band, and only float16 descriptors plus metadata are written.
- One request in flight, and at least 1 s from the end of one request to the start of
  the next (``RateLimiter``). Searches and tiles share the limiter.
- Circuit breaker (``Breaker``): the whole pilot stops on HTTP 403 or 429, a consent or
  CAPTCHA page, or 3 consecutive errors of any kind. There are no retries, no proxies
  and no header changes: searches use the library's own call, and tiles use the
  library's URL builder and the library's default tile headers.
- Official Google captures only (``is_official``); no historical panoramas are fetched.
  Their count is only read from the search metadata.
- Fetching and encoding run concurrently: the encoder thread never delays the request
  schedule.

Usage on the pod:
  python -m eval.gsv_internal_fetch --points data/gsv_internal/pilot_points.csv \\
      --deadline <unix time>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import socket
import sys
import threading
import time
import types
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from swespy import gsv

ROOT = Path("data/gsv_internal")
INDEX_NAME = "gsv_internal_pilot"
SOURCE = "gsv_internal"
SEARCH_RADIUS_M = 25
MIN_SPACING_S = 1.0
MAX_CONSECUTIVE_ERRORS = 3
VIEW_SIZE, VIEW_FOV, VIEW_PITCH = 640, 90.0, 0.0
MIN_VIEW_WIDTH = 600  # Pixels per 90° of panorama width at the chosen zoom.
OFFICIAL_SOURCES = {"launch", "scout"}  # Car and trekker; not Business View tripods.
# The library's own default headers for tile requests (streetlevel 0.12.13,
# streetview.get_panorama). Copied verbatim, not altered.
LIBRARY_TILE_HEADERS = {
    "Host": "streetviewpixels-pa.googleapis.com",
    "Origin": "https://www.google.com",
    "Referer": "https://www.google.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 11.0; Win64; x64; rv:151.0) Gecko/20100101 "
                  "Firefox/151.0",
}
BLOCK_MARKERS = re.compile(
    r"consent\.google|/sorry/|captcha|unusual traffic|before you continue", re.IGNORECASE)
PULL_LOG_FIELDS = ("utc", "bbox", "request_type", "request_count", "success_count",
                   "cumulative_count", "machine", "stop_reason")


# ---------------------------------------------------------------- rate limit and breaker


class PilotStopped(RuntimeError):
    """Stop the entire pilot. ``reason`` is recorded in the pull log."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class RateLimiter:
    """One request in flight; >= ``spacing`` s from the end of one to the next start."""

    def __init__(self, spacing: float = MIN_SPACING_S,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.spacing = spacing
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.last_end = -math.inf
        self.in_flight = 0
        self.max_in_flight = 0
        self.starts: list[float] = []
        self.ends: list[float] = []

    def __enter__(self) -> RateLimiter:
        self.lock.acquire()
        wait = self.last_end + self.spacing - self.clock()
        if wait > 0:
            self.sleep(wait)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.starts.append(self.clock())
        return self

    def __exit__(self, *exc: object) -> None:
        self.in_flight -= 1
        self.last_end = self.clock()
        self.ends.append(self.last_end)
        self.lock.release()


@dataclass
class Breaker:
    """Classifies every response; raises PilotStopped on any sign of blocking."""

    consecutive_errors: int = 0
    counts: Counter = field(default_factory=Counter)

    def check(self, status: int, content_type: str, body: bytes, final_url: str,
              expect: str) -> bool:
        """``expect`` is "json" or "image". Raises on a block; False on a plain error."""
        head = body[:4096].decode("utf-8", "ignore")
        if status == 403:
            raise PilotStopped("http_403")
        if status == 429:
            raise PilotStopped("http_429")
        if BLOCK_MARKERS.search(final_url) or (
                "html" in content_type.lower() and BLOCK_MARKERS.search(head)):
            raise PilotStopped("consent_or_captcha", content_type)
        ok = status == 200 and (
            (expect == "image" and content_type.lower().startswith("image/"))
            or (expect == "json" and "html" not in content_type.lower()))
        if ok:
            self.consecutive_errors = 0
            return True
        kind = f"http_{status}" if status != 200 else f"unexpected_{expect}_type"
        self.error(kind)
        return False

    def error(self, kind: str) -> None:
        self.counts[kind] += 1
        self.consecutive_errors += 1
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            raise PilotStopped("consecutive_errors", kind)

    def success(self) -> None:
        self.consecutive_errors = 0


class GuardedSession:
    """A requests.Session that every request goes through: limiter, breaker, counting.

    Passed to streetlevel as its ``session``. It adds no headers of its own.
    """

    def __init__(self, limiter: RateLimiter, breaker: Breaker, deadline: float,
                 session: Any = None):
        import requests

        self.session = session or requests.Session()
        self.limiter = limiter
        self.breaker = breaker
        self.deadline = deadline
        self.counts: Counter = Counter()
        self.ok: Counter = Counter()
        self.kind = "search"

    def _send(self, method: str, url: str, expect: str, **kw: Any) -> Any:
        import requests

        if time.time() >= self.deadline:
            raise PilotStopped("deadline")
        self.counts[self.kind] += 1
        with self.limiter:
            try:
                r = self.session.request(method, url, timeout=30, **kw)
            except requests.RequestException as exc:
                self.breaker.error(type(exc).__name__)
                raise FetchError(type(exc).__name__) from None
        chain = " ".join([h.url for h in r.history] + [r.url])
        if not self.breaker.check(r.status_code, r.headers.get("content-type", ""),
                                  r.content, chain, expect):
            raise FetchError(f"http_{r.status_code}" if r.status_code != 200 else "bad_type")
        self.ok[self.kind] += 1
        return r

    # The two methods streetlevel uses.
    def post(self, url: str, data: Any = None, headers: dict | None = None, **kw: Any) -> Any:
        return self._send("POST", url, "json", data=data, headers=headers, **kw)

    def get(self, url: str, headers: dict | None = None, **kw: Any) -> Any:
        return self._send("GET", url, "json", headers=headers, **kw)

    def get_tile(self, url: str) -> bytes:
        return self._send("GET", url, "image", headers=dict(LIBRARY_TILE_HEADERS)).content


class FetchError(RuntimeError):
    """A failed request that is not a block; the pano or point is skipped."""


# ---------------------------------------------------------------- library adapter


def import_streetlevel() -> types.ModuleType:
    """Import streetlevel.streetview. The library imports pyexiv2 only to write EXIF
    into files it saves; we never save files, so a stub is used if the native exiv2
    library is missing (as on macOS without Homebrew's inih)."""
    try:
        from streetlevel.streetview import streetview
    except (OSError, ImportError) as exc:
        if "exiv2" not in str(exc).lower() and "INIReader" not in str(exc):
            raise
        for name in [m for m in sys.modules if m.startswith("streetlevel")]:
            del sys.modules[name]
        stub = types.ModuleType("pyexiv2")
        stub.__getattr__ = lambda name: type(name, (), {})  # Names used in annotations.
        sys.modules["pyexiv2"] = stub
        from streetlevel.streetview import streetview
    return streetview


@dataclass
class Pano:
    id: str
    lat: float
    lng: float
    heading_deg: float  # Compass direction of the panorama's centre column.
    date: str | None
    copyright: str | None
    source: str | None
    third_party: bool
    historical: int
    sizes: list[tuple[int, int]]
    tile: tuple[int, int]
    raw: Any = None  # The library object, needed to build tile URLs.


def is_official(p: Pano) -> bool:
    return (not p.third_party and gsv.is_google_captured(p.copyright)
            and (p.source or "") in OFFICIAL_SOURCES)


class Library:
    """The only place that calls streetlevel."""

    def __init__(self, session: GuardedSession):
        self.sv = import_streetlevel()
        self.session = session

    def search(self, lat: float, lng: float) -> Pano | None:
        self.session.kind = "search"
        try:
            raw = self.sv.find_panorama(lat, lng, radius=SEARCH_RADIUS_M,
                                        search_third_party=False, session=self.session)
        except (FetchError, PilotStopped):
            raise
        except Exception as exc:  # noqa: BLE001 - unparseable answer counts as an error.
            self.session.breaker.error(f"parse_{type(exc).__name__}")
            raise FetchError("parse") from None
        if raw is None:
            return None
        date = raw.date
        date_text = (f"{date.year:04d}-{date.month:02d}" if hasattr(date, "year") else None)
        return Pano(
            id=raw.id, lat=raw.lat, lng=raw.lon,
            heading_deg=math.degrees(raw.heading or 0.0) % 360.0, date=date_text,
            copyright=raw.copyright_message, source=raw.source,
            third_party=bool(raw.is_third_party), historical=len(raw.historical or []),
            sizes=[(s.x, s.y) for s in (raw.image_sizes or [])],
            tile=(raw.tile_size.x, raw.tile_size.y) if raw.tile_size else (512, 512), raw=raw)

    def tile_urls(self, pano: Pano, zoom: int) -> dict[tuple[int, int], str]:
        return {(t.x, t.y): t.url for t in self.sv._generate_tile_list(pano.raw, zoom)}


# ---------------------------------------------------------------- tiles and rendering


def choose_zoom(sizes: list[tuple[int, int]]) -> int | None:
    """Smallest zoom whose panorama gives >= MIN_VIEW_WIDTH px per 90°."""
    for z, (w, _) in enumerate(sizes):
        if w / 4 >= MIN_VIEW_WIDTH:
            return z
    return None


def view_maps(pano_height: int, width: int, pano_heading: float, heading: float,
              fov: float = VIEW_FOV, size: int = VIEW_SIZE,
              pitch: float = VIEW_PITCH) -> tuple[np.ndarray, np.ndarray]:
    """Source (column, row) in the full panorama for every pixel of a pinhole view."""
    f = (size / 2) / math.tan(math.radians(fov) / 2)
    u, v = np.meshgrid(np.arange(size) + 0.5 - size / 2, np.arange(size) + 0.5 - size / 2)
    x, y, z = u / f, -v / f, np.ones_like(u)
    p = math.radians(pitch)  # Tilt the camera about its x axis.
    y, z = y * math.cos(p) + z * math.sin(p), -y * math.sin(p) + z * math.cos(p)
    yaw = np.degrees(np.arctan2(x, z)) + heading
    elev = np.degrees(np.arctan2(y, np.hypot(x, z)))
    col = (((yaw - pano_heading) / 360.0 + 0.5) % 1.0) * width - 0.5
    row = (0.5 - elev / 180.0) * pano_height - 0.5
    return col, row


def needed_rows(height: int, tile_h: int, fov: float = VIEW_FOV,
                pitch: float = VIEW_PITCH) -> list[int]:
    """Tile rows holding every pixel row a level view samples (bilinear: floor, floor+1)."""
    _, row = view_maps(height, 4 * height, 0.0, 0.0, fov, VIEW_SIZE, pitch)
    lo = max(0, math.floor(row.min() + 1e-6))
    hi = min(height - 1, math.ceil(row.max() - 1e-6))
    return list(range(lo // tile_h, hi // tile_h + 1))


def assemble_band(tiles: dict[tuple[int, int], bytes], width: int, height: int,
                  tile_w: int, tile_h: int, rows: list[int]) -> np.ndarray:
    """Decode tiles in memory into an RGB band of the full-width panorama."""
    from PIL import Image

    band = np.zeros(((rows[-1] - rows[0] + 1) * tile_h, width, 3), dtype=np.uint8)
    for (x, y), data in tiles.items():
        with Image.open(BytesIO(data)) as im:
            a = np.asarray(im.convert("RGB"))
        r0, c0 = (y - rows[0]) * tile_h, x * tile_w
        h, w = min(a.shape[0], band.shape[0] - r0), min(a.shape[1], width - c0)
        if h > 0 and w > 0:
            band[r0:r0 + h, c0:c0 + w] = a[:h, :w]
    return band


def render_view(band: np.ndarray, band_top: int, pano_height: int, pano_heading: float,
                heading: float, fov: float = VIEW_FOV, size: int = VIEW_SIZE,
                pitch: float = VIEW_PITCH) -> np.ndarray:
    """Pinhole view from an equirectangular band; heading is a compass bearing.

    The band is padded by one column on each side (wrapping around 360°) and rows are
    clamped to the band, so interpolation never reads outside the fetched tiles.
    """
    import cv2

    width = band.shape[1]
    col, row = view_maps(pano_height, width, pano_heading, heading, fov, size, pitch)
    padded = np.concatenate([band[:, -1:], band, band[:, :1]], axis=1)
    row = np.clip(row - band_top, 0, band.shape[0] - 1)
    return cv2.remap(padded, (col + 1).astype(np.float32), row.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def view_headings(way_bearing: float | None) -> list[float]:
    base = way_bearing if way_bearing is not None else 0.0
    return [round((base + o) % 360.0, 1) for o in (0, 90, 180, 270)]


# ---------------------------------------------------------------- outputs


class PullLog:
    """Append-only, one row per batch. No image data, URLs or keys."""

    def __init__(self, path: Path, machine: str):
        self.path = path
        self.machine = machine
        self.totals: Counter = Counter()

    def append(self, request_type: str, count: int, ok: int,
               bbox: tuple[float, float, float, float] | None,
               stop_reason: str | None = None) -> dict[str, Any]:
        self.totals[request_type] += count
        row = {"utc": datetime.now(UTC).isoformat(),
               "bbox": [round(v, 6) for v in bbox] if bbox else None,
               "request_type": request_type, "request_count": count, "success_count": ok,
               "cumulative_count": self.totals[request_type], "machine": self.machine,
               "stop_reason": stop_reason}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row


class IndexWriter:
    """float16 descriptors (append-only .npy chunks) + one JSON line per view."""

    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.views = folder / "views.jsonl"
        self.n = sum(1 for _ in self.views.open()) if self.views.exists() else 0

    def add(self, descriptors: np.ndarray, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        np.save(self.folder / f"descriptors-{self.n:06d}.npy", descriptors.astype(np.float16))
        with self.views.open("a") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps({**row, "row": self.n + i}) + "\n")
        self.n += len(rows)


# ---------------------------------------------------------------- pipeline


def quality_rule(view: np.ndarray) -> str | None:
    from PIL import Image

    from eval import quality

    return quality.image_rule(quality.measures(Image.fromarray(view)))


def encoder_loop(jobs: queue.Queue, embed: Callable[[list[Any]], np.ndarray],
                 writer: IndexWriter, stats: Counter, errors: list[str]) -> None:
    from PIL import Image

    while True:
        job = jobs.get()
        if job is None:
            return
        try:
            pano, point, band, top, height = job
            keep, rows = [], []
            for h in view_headings(point.get("bearing")):
                view = render_view(band, top, height, pano.heading_deg, h)
                rule = quality_rule(view)
                stats[f"view_{rule or 'kept'}"] += 1
                if rule is None:
                    keep.append(Image.fromarray(view))
                    rows.append({"pano_id": pano.id, "lat": pano.lat, "lng": pano.lng,
                                 "heading": h, "capture_date": pano.date,
                                 "way_type": point["highway"], "priority": point["priority"],
                                 "point_id": point["point_id"], "source": SOURCE,
                                 "attribution": gsv.ATTRIBUTION})
            del band
            if keep:
                writer.add(embed(keep), rows)
            del keep
        except Exception as exc:  # noqa: BLE001 - recorded; fetching continues.
            errors.append(f"{type(exc).__name__}: {exc}"[:300])
        finally:
            jobs.task_done()


def run(points: list[dict[str, Any]], out: Path, library: Library, session: GuardedSession,
        embed: Callable[[list[Any]], np.ndarray], machine: str,
        log_every: int = 10) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    folder = out / INDEX_NAME
    writer = IndexWriter(folder)
    pull = PullLog(out / "pull_log.jsonl", machine)
    results_path = folder / "points.jsonl"
    done = ({json.loads(line)["point_id"] for line in results_path.open()}
            if results_path.exists() else set())
    seen_panos = {json.loads(line)["pano_id"] for line in results_path.open()
                  if json.loads(line).get("pano_id")} if results_path.exists() else set()
    stats: Counter = Counter()
    errors: list[str] = []
    jobs: queue.Queue = queue.Queue(maxsize=200)
    encoder = threading.Thread(target=encoder_loop, args=(jobs, embed, writer, stats, errors),
                               daemon=True)
    encoder.start()
    stop_reason = "completed"
    batch: list[dict[str, Any]] = []
    started = time.time()
    zooms: Counter = Counter()
    sizes: Counter = Counter()
    tiles_per_pano: list[int] = []
    max_queue = 0

    def flush(reason: str | None = None) -> None:
        if not batch and reason is None:
            return
        lats = [p["lat"] for p in batch] or [0.0]
        lngs = [p["lng"] for p in batch] or [0.0]
        bbox = (min(lngs), min(lats), max(lngs), max(lats)) if batch else None
        for kind in ("search", "tile"):
            n = session.counts[kind] - flush.sent[kind]
            ok = session.ok[kind] - flush.ok[kind]
            if n or reason:
                pull.append(kind, n, ok, bbox, reason)
            flush.sent[kind], flush.ok[kind] = session.counts[kind], session.ok[kind]
        batch.clear()

    flush.sent, flush.ok = Counter(), Counter()  # type: ignore[attr-defined]

    def record(point: dict[str, Any], **data: Any) -> None:
        with results_path.open("a") as f:
            f.write(json.dumps({"point_id": point["point_id"], "priority": point["priority"],
                                "way_type": point["highway"], **data}) + "\n")

    try:
        for point in points:
            if point["point_id"] in done:
                continue
            batch.append(point)
            try:
                pano = library.search(point["lat"], point["lng"])
            except FetchError as exc:
                record(point, outcome=f"search_error_{exc}")
                continue
            if pano is None:
                record(point, outcome="no_pano")
            elif not is_official(pano):
                record(point, outcome="not_official", source=pano.source)
            elif pano.id in seen_panos:
                record(point, outcome="duplicate_pano", pano_id=pano.id)
            else:
                zoom = choose_zoom(pano.sizes)
                if zoom is None:
                    record(point, outcome="too_small", pano_id=pano.id, sizes=pano.sizes)
                else:
                    width, height = pano.sizes[zoom]
                    rows = needed_rows(height, pano.tile[1])
                    urls = {k: u for k, u in library.tile_urls(pano, zoom).items()
                            if k[1] in rows}
                    session.kind = "tile"
                    tiles: dict[tuple[int, int], bytes] = {}
                    try:
                        for key in sorted(urls):
                            tiles[key] = session.get_tile(urls[key])
                    except FetchError as exc:
                        record(point, outcome=f"tile_error_{exc}", pano_id=pano.id)
                        seen_panos.add(pano.id)
                        continue
                    band = assemble_band(tiles, width, height, *pano.tile, rows)
                    del tiles
                    jobs.put((pano, point, band, rows[0] * pano.tile[1], height))
                    max_queue = max(max_queue, jobs.qsize())
                    seen_panos.add(pano.id)
                    zooms[zoom] += 1
                    sizes[f"{width}x{height}"] += 1
                    tiles_per_pano.append(len(urls))
                    record(point, outcome="fetched", pano_id=pano.id, zoom=zoom,
                           size=[width, height], tiles=len(urls), date=pano.date,
                           historical=pano.historical, source=pano.source)
            if len(batch) >= log_every:
                flush()
    except PilotStopped as stop:
        stop_reason = stop.reason if not stop.detail else f"{stop.reason} ({stop.detail})"
    finally:
        flush(stop_reason)
        jobs.put(None)
        encoder.join()
    elapsed = time.time() - started
    summary = {
        "stop_reason": stop_reason, "stopped_at": datetime.now(UTC).isoformat(),
        "elapsed_s": round(elapsed, 1), "requests": dict(session.counts),
        "successful": dict(session.ok), "errors": dict(session.breaker.counts),
        "max_in_flight": session.limiter.max_in_flight,
        "min_spacing_s": round(min((s - e for s, e in zip(session.limiter.starts[1:],
                                                           session.limiter.ends,
                                                           strict=False)),
                                   default=float("nan")), 3),
        "zooms": dict(zooms), "pano_sizes": dict(sizes),
        "tiles_per_pano": dict(Counter(tiles_per_pano)),
        "views": dict(stats), "views_stored": writer.n, "encoder_errors": errors[:20],
        "max_encoder_queue": max_queue, "machine": machine,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, default=ROOT / "pilot_points.csv")
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--deadline", type=float, required=True, help="Unix time to stop")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from eval.gsv_internal import read_points
    from swespy.model.megaloc import MegaLocEmbedder

    embedder = MegaLocEmbedder(Path("data/megaloc"), args.device)  # fp32 compute.
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # Leave a core for fetching.
    session = GuardedSession(RateLimiter(), Breaker(), args.deadline)
    machine = f"runpod:{os.environ.get('RUNPOD_POD_ID', socket.gethostname())}"
    summary = run(read_points(args.points), args.out, Library(session), session,
                  embedder.batch_images, machine)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
