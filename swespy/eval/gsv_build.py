"""Street View gap fill via the official Google APIs (Map Tiles API + Street View Static).

Owner decisions (2026-11-10): official APIs only; zoom 3 tiles; caps of 14,500 tile
requests per Pacific day, 220,000 tile requests in total and 9,500 Static requests per
billing month (enforced by swespy.gsv.Budget, crash-safe). Uncharged requests (session,
panoIds, metadata) are logged but not counted. A single manual run: nothing here
schedules anything; running ``build`` again resumes from the checkpoint.

All Street View rules apply: descriptors only (images live in memory and are discarded),
"Google Street View, © Google", every pull logged, never training or pair mining, never
displayed. The Mapillary/Commons/KartaView gallery is not touched; this writes its own
index in data/gsv/gsv_index/.

Stages:
  smoke     session + panoIds for 5 gap points + tiles for one view -> render -> encode.
  discover  panoIds (100 locations per request) and metadata for every gap point except
            forest tracks, in the owner's order; official Google captures only.
  build     assign panoramas to tiles / Static (never both), then fetch, render, quality
            filter, encode (MegaLoc fp32) and store float16 descriptors, both routes in
            parallel, until the caps.
  status    caps used and remaining, index size.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from eval.gsv_internal_fetch import render_view, view_maps
from eval.gsv_plan import NON_ROAD, metadata_queue
from swespy import gsv

ROOT = Path("data/gsv")
INDEX = ROOT / "gsv_index"
PLAN = ROOT / "build_plan.json"
JOURNAL = ROOT / "billable_journal.jsonl"
PULL_LOG = ROOT / "pull_log.jsonl"
TILE_API = "https://tile.googleapis.com/v1"
STATIC_API = "https://maps.googleapis.com/maps/api/streetview"
ZOOM = 3
MAX_ZOOM = 5  # Map Tiles API: zoom 5 is the full-size panorama.
RADIUS_M = 25
VIEW = 640
REQUESTS_PER_SECOND = 80.0  # Below Google's 6,000 per minute for each request family.
MAX_CONSECUTIVE_ERRORS = 10
STATIC_EVAL_RESERVE = 400  # Static requests kept for live verification in the evaluation.
# The owner's order. Forest tracks are skipped everywhere; ways the owner did not list
# come last, so the listed order is followed strictly.
ORDER = (
    (2, {"residential"}),
    (1, {"footway"}),
    (2, {"service", "unclassified"}),
    (1, {"path"}),
    (3, {"residential", "service", "unclassified", "living_street", "tertiary",
         "tertiary_link", "secondary", "secondary_link", "primary", "primary_link", "road"}),
)
SKIP_WAYS = {"track"}


class QuotaExceeded(RuntimeError):
    """Google reported a quota error before our cap."""


class ApiError(RuntimeError):
    """Repeated or fatal API errors. Never carries a URL or key."""


# ---------------------------------------------------------------- ordering and views


def order_group(priority: int, highway: str) -> int:
    for i, (prio, ways) in enumerate(ORDER):
        if priority == prio and highway in ways:
            return i
    return len(ORDER)


def ordered_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue_ = [p for p in metadata_queue(points) if p["highway"] not in SKIP_WAYS]
    return sorted(queue_, key=lambda p: order_group(p["priority"], p["highway"]))


def headings_for(highway: str, bearing: float | None) -> list[float]:
    """4 headings on footways/paths (non-road ways), 2 sideways on roads."""
    base = bearing if bearing is not None else 0.0
    offsets = (0, 90, 180, 270) if highway in NON_ROAD else (90, 270)
    return [round((base + o) % 360.0, 1) for o in offsets]


def zoom_size(image_w: int, image_h: int, zoom: int = ZOOM) -> tuple[int, int]:
    scale = 2 ** (MAX_ZOOM - zoom)
    return math.ceil(image_w / scale), math.ceil(image_h / scale)


def tiles_needed(pano: dict[str, Any], headings: list[float]) -> list[tuple[int, int]]:
    """Exact tiles (x, y) at ZOOM that the views sample (bilinear neighbours included)."""
    w, h = zoom_size(pano["image_w"], pano["image_h"])
    tw, th = pano["tile_w"], pano["tile_h"]
    need: set[tuple[int, int]] = set()
    for heading in headings:
        col, row = view_maps(h, w, pano["heading"], heading)
        c0 = np.floor(col).astype(int) % w
        r0 = np.clip(np.floor(row).astype(int), 0, h - 1)
        for cc in (c0, (c0 + 1) % w):
            for rr in (r0, np.clip(r0 + 1, 0, h - 1)):
                need |= set(zip((cc // tw).ravel().tolist(), (rr // th).ravel().tolist(),
                                strict=True))
    return sorted(need)


# ---------------------------------------------------------------- API client


class Limiter:
    """Simple request pacing shared by all threads."""

    def __init__(self, per_second: float = REQUESTS_PER_SECOND):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        if start > now:
            time.sleep(start - now)


@dataclass
class Counts:
    sent: Counter = field(default_factory=Counter)
    ok: Counter = field(default_factory=Counter)


class Api:
    """The only place that talks to Google. Billable calls reserve budget first."""

    def __init__(self, key: str, budget: gsv.Budget, client: Any = None,
                 limiter: Limiter | None = None, sleep: Any = time.sleep):
        import httpx

        self.key = key
        self.budget = budget
        self.client = client or httpx.Client(timeout=30)
        self.limiter = limiter or Limiter()
        self.sleep = sleep
        self.session: str | None = None
        self.counts = Counts()
        self.lock = threading.Lock()
        self.consecutive_errors = 0

    def _call(self, kind: str, method: str, url: str, params: dict[str, Any],
              body: Any = None, expect: str = "json") -> Any:
        import httpx

        for attempt in range(4):
            if kind in ("tile", "image") and attempt > 0 and not self.budget.take(kind, 1):
                raise ApiError(f"{kind} cap reached during retry")
            self.limiter.wait()
            with self.lock:
                self.counts.sent[kind] += 1
            try:
                r = self.client.request(method, url, params={**params, "key": self.key},
                                        json=body)
                status, text = r.status_code, r.text if expect == "json" else ""
            except httpx.HTTPError as exc:
                status, text, r = -1, type(exc).__name__, None
            if status == 200 and r is not None:
                ctype = r.headers.get("content-type", "")
                if expect == "image" and not ctype.startswith("image/"):
                    status = -2
                else:
                    with self.lock:
                        self.counts.ok[kind] += 1
                        self.consecutive_errors = 0
                    return r.json() if expect == "json" else r.content
            if expect != "json" and r is not None:
                text = r.text[:2000] if "json" in r.headers.get("content-type", "") else ""
            low = text.lower()
            if status in (429, 403) and ("quota" in low or "resource_exhausted" in low
                                         or "rate limit" in low and status == 403):
                raise QuotaExceeded(f"{kind}: HTTP {status} quota")
            if status in (400, 401, 403, 404):
                with self.lock:
                    self.consecutive_errors += 1
                    errors = self.consecutive_errors
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    raise ApiError(f"{errors} consecutive errors (last: {kind} HTTP {status})")
                raise ApiError(f"{kind}: HTTP {status}") if status in (401, 403) else \
                    LookupError(f"{kind}: HTTP {status}")
            with self.lock:
                self.consecutive_errors += 1
                errors = self.consecutive_errors
            if errors >= MAX_CONSECUTIVE_ERRORS:
                raise ApiError(f"{errors} consecutive errors (last: {kind} status {status})")
            self.sleep(min(30.0, 2.0 ** attempt))  # Back off on 429/5xx/network errors.
        raise LookupError(f"{kind}: failed after retries")

    def create_session(self) -> str:
        data = self._call("session", "POST", f"{TILE_API}/createSession", {},
                          body={"mapType": "streetview", "language": "en-US", "region": "SE"})
        self.session = data["session"]
        return self.session

    def pano_ids(self, locations: list[tuple[float, float]]) -> list[str]:
        data = self._call("panoids", "POST", f"{TILE_API}/streetview/panoIds",
                          {"session": self.session},
                          body={"locations": [{"lat": a, "lng": b} for a, b in locations],
                                "radius": RADIUS_M})
        return data.get("panoIds", [])

    def metadata(self, pano_id: str) -> dict[str, Any]:
        return self._call("metadata", "GET", f"{TILE_API}/streetview/metadata",
                          {"session": self.session, "panoId": pano_id})

    def tile(self, pano_id: str, x: int, y: int) -> bytes:
        """Caller must have reserved the tile in the budget."""
        return self._call("tile", "GET", f"{TILE_API}/streetview/tiles/{ZOOM}/{x}/{y}",
                          {"session": self.session, "panoId": pano_id}, expect="image")

    def static(self, pano_id: str, heading: float) -> bytes:
        """Caller must have reserved the image in the budget."""
        return self._call("image", "GET", STATIC_API,
                          {"pano": pano_id, "heading": heading, "fov": 90, "pitch": 0,
                           "size": f"{VIEW}x{VIEW}", "return_error_code": "true"},
                          expect="image")


# ---------------------------------------------------------------- logging


class LogBuffer:
    """Accumulates request counts and writes pull-log rows (no keys, URLs or images)."""

    def __init__(self, log: gsv.PullLog, api: Api):
        self.log = log
        self.api = api
        self.written = Counts()
        self.lock = threading.Lock()

    def flush(self, bbox: tuple[float, float, float, float] | None = None,
              priority: int | None = None) -> None:
        with self.lock:
            for kind in gsv.REQUEST_TYPES:
                sent = self.api.counts.sent[kind] - self.written.sent[kind]
                ok = self.api.counts.ok[kind] - self.written.ok[kind]
                if sent:
                    self.log.append(request_type=kind, request_count=sent, success_count=ok,
                                    bbox=bbox, priority=priority)
                    self.written.sent[kind] += sent
                    self.written.ok[kind] += ok


def bbox_of(items: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    if not items:
        return None
    lats = [p["lat"] for p in items]
    lngs = [p["lng"] for p in items]
    return min(lngs), min(lats), max(lngs), max(lats)


# ---------------------------------------------------------------- discover


def load_plan() -> dict[str, Any]:
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    return {"points_done": 0, "panos": [], "metadata_done": []}


def save_plan(plan: dict[str, Any]) -> None:
    tmp = PLAN.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan))
    tmp.replace(PLAN)


def official(meta: dict[str, Any]) -> bool:
    return (meta.get("imageryType", "outdoor") == "outdoor"
            and gsv.is_google_captured(meta.get("copyright")))


def discover(api: Api, logs: LogBuffer, points: list[dict[str, Any]],
             max_points: int | None = None, workers: int = 48) -> dict[str, Any]:
    plan = load_plan()
    seen = {p["pano_id"] for p in plan["panos"]} | set(plan.get("rejected", []))
    plan.setdefault("rejected", [])
    target = len(points) if max_points is None else min(len(points), max_points)
    start = plan["points_done"]
    with ThreadPoolExecutor(workers) as pool:
        for b in range(start, target, 100 * workers):
            chunks = [points[i:i + 100] for i in range(b, min(b + 100 * workers, target), 100)]
            def ids_of(c: list[dict[str, Any]]) -> list[str]:
                try:
                    return api.pano_ids([(p["lat"], p["lng"]) for p in c])
                except LookupError:
                    return []

            answers = list(pool.map(ids_of, chunks))
            fresh: list[tuple[str, dict[str, Any]]] = []
            for chunk, ids in zip(chunks, answers, strict=True):
                for point, pano_id in zip(chunk, ids, strict=False):
                    if pano_id and pano_id not in seen:
                        seen.add(pano_id)
                        fresh.append((pano_id, point))

            def meta_of(item: tuple[str, dict[str, Any]]) -> tuple[str, dict, dict] | None:
                try:
                    return item[0], item[1], api.metadata(item[0])
                except LookupError:
                    return None

            for res in pool.map(meta_of, fresh):
                if res is None:
                    continue
                pano_id, point, meta = res
                if not official(meta):
                    plan["rejected"].append(pano_id)
                    continue
                pano = {"pano_id": pano_id, "lat": meta["lat"], "lng": meta["lng"],
                        "heading": float(meta.get("heading", 0.0)), "date": meta.get("date"),
                        "image_w": int(meta["imageWidth"]), "image_h": int(meta["imageHeight"]),
                        "tile_w": int(meta.get("tileWidth", 512)),
                        "tile_h": int(meta.get("tileHeight", 512)),
                        "priority": point["priority"], "way_type": point["highway"],
                        "way_id": point.get("way_id"), "bearing": point.get("bearing"),
                        "group": order_group(point["priority"], point["highway"]),
                        "route": None}
                pano["headings"] = headings_for(pano["way_type"], pano["bearing"])
                plan["panos"].append(pano)
            plan["points_done"] = min(b + 100 * workers, target)
            save_plan(plan)
            logs.flush(bbox_of([p for c in chunks for p in c]))
            print(json.dumps({"discover": plan["points_done"], "of": target,
                              "panos": len(plan["panos"]),
                              "rejected": len(plan["rejected"])}), flush=True)
    plan["panos"].sort(key=lambda p: p["group"])  # Stable: keeps point order in a group.
    save_plan(plan)
    return plan


# ---------------------------------------------------------------- assignment


def assign(plan: dict[str, Any], tile_room: int, static_room: int,
           done: set[tuple[str, float]]) -> dict[str, int]:
    """Give unassigned panoramas a route, in order: tiles first, then Static.

    A panorama keeps its route forever once assigned (saved in the plan), so the two
    routes never fetch the same panorama, across runs too.
    """
    counts = Counter()
    tiles_open = True
    for p in plan["panos"]:
        if all((p["pano_id"], h) in done for h in p["headings"]):
            continue
        if "tiles" not in p:
            p["tiles"] = len(tiles_needed(p, p["headings"]))
        if p["route"] == "tiles":
            tile_room -= p["tiles"]
        elif p["route"] == "static":
            static_room -= len(p["headings"])
        elif tiles_open and p["tiles"] <= tile_room:
            p["route"] = "tiles"
            tile_room -= p["tiles"]
            counts["tiles"] += 1
        else:
            tiles_open = False  # Strict order: no skipping ahead to smaller panoramas.
            if len(p["headings"]) <= static_room:
                p["route"] = "static"
                static_room -= len(p["headings"])
                counts["static"] += 1
            else:
                break
    return dict(counts)


# ---------------------------------------------------------------- index


class Index:
    """float16 descriptors + views.jsonl + done.jsonl (checkpoint by pano ID + heading)."""

    def __init__(self, folder: Path = INDEX, flush_every: int = 256):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.views_path = folder / "views.jsonl"
        self.done_path = folder / "done.jsonl"
        self.n = sum(1 for _ in self.views_path.open()) if self.views_path.exists() else 0
        self.flush_every = flush_every
        self.buf_desc: list[np.ndarray] = []
        self.buf_rows: list[dict[str, Any]] = []
        self.buf_done: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def done(self) -> set[tuple[str, float]]:
        if not self.done_path.exists():
            return set()
        return {(d["pano_id"], d["heading"]) for d in
                (json.loads(x) for x in self.done_path.read_text().splitlines() if x)}

    def add(self, desc: np.ndarray | None, rows: list[dict[str, Any]],
            done: list[dict[str, Any]]) -> None:
        with self.lock:
            if desc is not None and len(rows):
                self.buf_desc.append(desc.astype(np.float16))
                self.buf_rows.extend(rows)
            self.buf_done.extend(done)
            if len(self.buf_rows) >= self.flush_every:
                self._flush()

    def flush(self) -> None:
        with self.lock:
            self._flush()

    def _flush(self) -> None:
        if self.buf_rows:
            np.save(self.folder / f"descriptors-{self.n:07d}.npy",
                    np.concatenate(self.buf_desc))
            with self.views_path.open("a") as f:
                for i, row in enumerate(self.buf_rows):
                    f.write(json.dumps({**row, "row": self.n + i}) + "\n")
            self.n += len(self.buf_rows)
        if self.buf_done:  # Checkpoint only after the descriptors are on disk.
            with self.done_path.open("a") as f:
                for d in self.buf_done:
                    f.write(json.dumps(d) + "\n")
        self.buf_desc, self.buf_rows, self.buf_done = [], [], []

    def load(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        rows = [json.loads(x) for x in self.views_path.read_text().splitlines() if x] \
            if self.views_path.exists() else []
        files = sorted(self.folder.glob("descriptors-*.npy"))
        desc = np.concatenate([np.load(p) for p in files]) if files else np.zeros((0, 8448))
        return rows, desc.astype(np.float32)


# ---------------------------------------------------------------- build


def fetch_tiles_views(api: Api, pano: dict[str, Any], headings: list[float],
                      pool: ThreadPoolExecutor) -> list[tuple[float, np.ndarray]]:
    from PIL import Image

    keys = tiles_needed(pano, headings)
    data = dict(zip(keys, pool.map(lambda k: api.tile(pano["pano_id"], *k), keys),
                    strict=True))
    w, h = zoom_size(pano["image_w"], pano["image_h"])
    tw, th = pano["tile_w"], pano["tile_h"]
    rows = sorted({y for _, y in keys})
    band = np.zeros(((rows[-1] - rows[0] + 1) * th, w, 3), dtype=np.uint8)
    for (x, y), raw in data.items():
        with Image.open(BytesIO(raw)) as im:
            a = np.asarray(im.convert("RGB"))
        r0, c0 = (y - rows[0]) * th, x * tw
        hh, ww = min(a.shape[0], band.shape[0] - r0), min(a.shape[1], w - c0)
        if hh > 0 and ww > 0:
            band[r0:r0 + hh, c0:c0 + ww] = a[:hh, :ww]
    del data
    return [(hd, render_view(band, rows[0] * th, h, pano["heading"], hd)) for hd in headings]


def fetch_static_views(api: Api, pano: dict[str, Any],
                       headings: list[float]) -> list[tuple[float, np.ndarray]]:
    from PIL import Image

    out = []
    for hd in headings:
        with Image.open(BytesIO(api.static(pano["pano_id"], hd))) as im:
            out.append((hd, np.asarray(im.convert("RGB"))))
    return out


def quality_rule(view: np.ndarray) -> str | None:
    from PIL import Image

    from eval import quality

    return quality.image_rule(quality.measures(Image.fromarray(view)))


def build(api: Api, logs: LogBuffer, embed: Any, index: Index, budget: gsv.Budget,
          routes: tuple[str, ...] = ("tiles", "static"), static_reserve: int = 0,
          log_every: int = 25) -> dict[str, Any]:
    from PIL import Image

    plan = load_plan()
    done = index.done()
    assigned = assign(plan, budget.remaining("tile") if "tiles" in routes else 0,
                      max(0, budget.remaining("image") - static_reserve)
                      if "static" in routes else 0, done)
    save_plan(plan)
    started = time.time()
    jobs: queue.Queue = queue.Queue(maxsize=64)
    stats: Counter = Counter()
    stop: dict[str, str] = {}
    errors: list[str] = []

    def encoder() -> None:
        while True:
            job = jobs.get()
            if job is None:
                return
            pano, route, views = job
            try:
                keep, rows, marks = [], [], []
                for hd, view in views:
                    rule = quality_rule(view)
                    stats[f"{route}_view_{rule or 'kept'}"] += 1
                    marks.append({"pano_id": pano["pano_id"], "heading": hd, "route": route,
                                  "kept": rule is None})
                    if rule is None:
                        keep.append(Image.fromarray(view))
                        rows.append({"pano_id": pano["pano_id"], "lat": pano["lat"],
                                     "lng": pano["lng"], "heading": hd,
                                     "capture_date": pano["date"], "way_type": pano["way_type"],
                                     "way_id": pano.get("way_id"), "priority": pano["priority"],
                                     "source": gsv.SOURCE, "route": route,
                                     "attribution": gsv.ATTRIBUTION})
                del views
                index.add(embed(keep) if keep else None, rows, marks)
                stats[f"{route}_panos"] += 1
            except Exception as exc:  # noqa: BLE001 - recorded; the run continues.
                errors.append(f"{type(exc).__name__}: {exc}"[:200])
            finally:
                jobs.task_done()

    def worker(route: str) -> None:
        try:
            work(route)
        except Exception as exc:  # noqa: BLE001 - recorded as the route's stop reason.
            stop[route] = f"crash: {type(exc).__name__}: {exc}"[:300]

    def work(route: str) -> None:
        todo = [p for p in plan["panos"] if p["route"] == route]
        batch: list[dict[str, Any]] = []
        with ThreadPoolExecutor(8) as pool:
            for pano in todo:
                if route in stop:
                    return
                headings = [h for h in pano["headings"] if (pano["pano_id"], h) not in done]
                if not headings:
                    continue
                n = len(tiles_needed(pano, headings)) if route == "tiles" else len(headings)
                if not budget.take("tile" if route == "tiles" else "image", n):
                    stop.setdefault(route, "cap")
                    return
                try:
                    views = (fetch_tiles_views(api, pano, headings, pool) if route == "tiles"
                             else fetch_static_views(api, pano, headings))
                except LookupError as exc:
                    stats[f"{route}_pano_errors"] += 1
                    errors.append(f"{route} {exc}"[:200])
                    continue
                except (QuotaExceeded, ApiError) as exc:
                    stop[route] = f"{type(exc).__name__}: {exc}"
                    return
                jobs.put((pano, route, views))
                batch.append(pano)
                if len(batch) >= log_every:
                    logs.flush(bbox_of(batch), min(p["priority"] for p in batch))
                    print(json.dumps({"route": route, **budget.summary(),
                                      "panos": dict(stats), "queue": jobs.qsize(),
                                      "elapsed_s": round(time.time() - started)}), flush=True)
                    batch = []
        if batch:
            logs.flush(bbox_of(batch), min(p["priority"] for p in batch))
        stop.setdefault(route, "completed")

    enc = threading.Thread(target=encoder, daemon=True)
    enc.start()
    threads = [threading.Thread(target=worker, args=(r,)) for r in routes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    jobs.put(None)
    enc.join()
    index.flush()
    logs.flush()
    summary = {"stopped": stop, "assigned_now": assigned, "stats": dict(stats),
               "errors": errors[:20], "elapsed_s": round(time.time() - started),
               "budget": budget.summary(), "index_views": index.n,
               "requests_sent": dict(api.counts.sent), "requests_ok": dict(api.counts.ok)}
    (ROOT / f"build_summary_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json").write_text(
        json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------- smoke


def smoke(api: Api, logs: LogBuffer, embed: Any, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Session + panoIds for 5 gap points + tiles for one view -> render -> encode."""
    from PIL import Image

    api.create_session()
    five = points[:5]
    ids = api.pano_ids([(p["lat"], p["lng"]) for p in five])
    found = [(i, p) for i, p in zip(ids, five, strict=False) if i]
    result: dict[str, Any] = {"points": 5, "panos_found": len(found)}
    for pano_id, point in found:
        meta = api.metadata(pano_id)
        if not official(meta):
            continue
        pano = {"pano_id": pano_id, "heading": float(meta.get("heading", 0.0)),
                "image_w": int(meta["imageWidth"]), "image_h": int(meta["imageHeight"]),
                "tile_w": int(meta.get("tileWidth", 512)),
                "tile_h": int(meta.get("tileHeight", 512))}
        heading = headings_for(point["highway"], point.get("bearing"))[0]
        n = len(tiles_needed(pano, [heading]))
        if not api.budget.take("tile", n):
            raise ApiError("tile cap reached in smoke test")
        with ThreadPoolExecutor(8) as pool:
            views = fetch_tiles_views(api, pano, [heading], pool)
        vec = embed([Image.fromarray(views[0][1])])
        result.update({"pano": pano_id, "date": meta.get("date"),
                       "copyright": meta.get("copyright"),
                       "pano_size": [pano["image_w"], pano["image_h"]],
                       "zoom3_size": list(zoom_size(pano["image_w"], pano["image_h"])),
                       "tiles_for_one_view": n, "descriptor_norm": float(np.linalg.norm(vec)),
                       "quality_rule": quality_rule(views[0][1])})
        del views
        break
    logs.flush(bbox_of(five))
    result["requests_sent"] = dict(api.counts.sent)
    result["requests_ok"] = dict(api.counts.ok)
    return result


def scan_images(roots: list[Path], since: float) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif", ".bmp", ".tif", ".tiff"}
    magic = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8")
    found = []
    for root in roots:
        for p in root.rglob("*"):
            try:
                if not p.is_file() or p.stat().st_mtime < since:
                    continue
                if p.suffix.lower() in exts:
                    found.append(str(p))
                    continue
                with p.open("rb") as f:
                    if f.read(4).startswith(magic):
                        found.append(str(p))
            except OSError:
                continue
    return found


def streets_report(top: int = 10) -> list[dict[str, Any]]:
    """Streets with the most new Street View coverage (unique panoramas), for field tests.

    Names come from OpenStreetMap (one Overpass request by way ID; ODbL).
    """
    import httpx

    rows, _ = Index().load()
    by_way: dict[int, set[str]] = {}
    where: dict[int, tuple[float, float]] = {}
    for r in rows:
        if r.get("way_id") is None:
            continue
        by_way.setdefault(int(r["way_id"]), set()).add(r["pano_id"])
        where.setdefault(int(r["way_id"]), (r["lat"], r["lng"]))
    if not by_way:
        return []
    ids = sorted(by_way, key=lambda w: -len(by_way[w]))[:400]
    query = f"[out:json][timeout:60];way(id:{','.join(map(str, ids))});out tags;"
    r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": query},
                   headers={"User-Agent": "heimdall-spy/0.1 (research)"}, timeout=90)
    r.raise_for_status()
    names = {e["id"]: e.get("tags", {}).get("name") for e in r.json()["elements"]}
    streets: dict[str, dict[str, Any]] = {}
    for w in ids:
        name = names.get(w) or f"(unnamed way {w})"
        s_ = streets.setdefault(name, {"street": name, "panos": set(), "example": where[w],
                                       "views": 0})
        s_["panos"] |= by_way[w]
        s_["views"] += sum(1 for r in rows if r.get("way_id") == w)
    ranked = sorted((s for s in streets.values() if not s["street"].startswith("(unnamed")),
                    key=lambda s: -len(s["panos"]))[:top]
    return [{"street": s["street"], "new_panoramas": len(s["panos"]), "new_views": s["views"],
             "example_latlng": [round(s["example"][0], 6), round(s["example"][1], 6)]}
            for s in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["smoke", "discover", "build", "status", "streets"])
    parser.add_argument("--routes", default="tiles,static")
    parser.add_argument("--static-reserve", type=int, default=STATIC_EVAL_RESERVE)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    budget = gsv.Budget(JOURNAL)
    if args.stage == "streets":
        print(json.dumps(streets_report(), indent=1, ensure_ascii=False))
        return
    if args.stage == "status":
        rows = Index().n
        print(json.dumps({**budget.summary(), "index_views": rows}, indent=1))
        return
    started = time.time()
    api = Api(gsv.api_key(), budget)
    logs = LogBuffer(gsv.PullLog(PULL_LOG), api)
    points = ordered_points(json.loads((ROOT / "sample_points.json").read_text())["points"])
    embed = None
    if args.stage in ("smoke", "build"):
        from swespy.model.megaloc import MegaLocEmbedder

        embed = MegaLocEmbedder(Path("data/megaloc"), args.device).batch_images  # fp32
    try:
        if args.stage == "smoke":
            out = smoke(api, logs, embed, points)
        elif args.stage == "discover":
            api.create_session()
            plan = discover(api, logs, points)
            out = {"panos": len(plan["panos"]), "rejected": len(plan["rejected"]),
                   "by_group": dict(Counter(p["group"] for p in plan["panos"]))}
        else:
            api.create_session()
            out = build(api, logs, embed, Index(), budget,
                        tuple(r.strip() for r in args.routes.split(",")),
                        static_reserve=args.static_reserve)
    finally:
        logs.flush()
    out["image_files_written"] = scan_images([ROOT, Path("data/gsv_internal")], started)
    out["budget"] = budget.summary()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
