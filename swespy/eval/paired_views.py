"""Prepare recent cross-capture pairs and a separate photographer-held-out photo pilot."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from eval.common import duplicate, signature, thumbnail, write_new
from eval.dense_linkoping import normalized
from eval.expand_linkoping import BBOX, CENTRAL
from eval.metrics import distance_m
from eval.public_photos import ROOT as PUBLIC
from eval.public_photos import SINCE, UA, UNTIL, candidate
from swespy.model.megaloc import file_hash

ROOT = Path("data/paired_views/recent_pilot_v2")
SOURCES = Path("data/paired_views/feasibility_v1")
TRAIN_CREATORS = {"la2", "quickness805", "ugz"}


async def discover() -> None:
    """Reproduce bounded free-source feasibility queries, preserving metadata checkpoints."""
    from loguru import logger

    from eval.dense_linkoping import FIELDS
    from eval.public_photos import request
    from swespy.config import get_settings
    from swespy.data.mapillary import MapillaryClient

    logger.disable("swespy.data.mapillary")
    SOURCES.mkdir(parents=True, exist_ok=True)
    boxes = {
        "recent-central": (15.615, 58.405, 15.635, 58.418),
        "central_west": (15.595, 58.400, 15.620, 58.420),
        "central_east": (15.620, 58.400, 15.645, 58.420),
        "campus": (15.565, 58.390, 15.600, 58.410),
        "skaggetorp": (15.560, 58.420, 15.600, 58.440),
        "station": (15.615, 58.412, 15.645, 58.432),
    }
    async with MapillaryClient(get_settings()) as api:
        for name, box in boxes.items():
            path = SOURCES / f"mapillary-{name}.json"
            if not path.exists():
                data = await api._request(
                    "/images",
                    {
                        "bbox": ",".join(map(str, box)),
                        "fields": FIELDS,
                        "limit": 200,
                        "start_captured_at": "2023-11-06T00:00:00Z",
                        "end_captured_at": "2026-11-07T00:00:00Z",
                    },
                )
                write_new(path, data)
    async with httpx.AsyncClient(headers={"User-Agent": UA}, timeout=45) as client:
        ids: set[str] = set()
        for year in (2024, 2025, 2026):
            category = SOURCES / f"commons-category-{year}.json"
            if not category.exists():
                data = await request(
                    client,
                    {
                        "list": "categorymembers",
                        "cmnamespace": 6,
                        "cmtitle": f"Category:{year} in Linköping Municipality",
                        "cmlimit": 500,
                    },
                )
                write_new(category, data)
            ids.update(
                str(r["pageid"])
                for r in json.loads(category.read_text())
                .get("query", {})
                .get("categorymembers", [])
            )
            search = SOURCES / f"commons-search-{year}.json"
            if not search.exists():
                query = f"Linköping {year} -intitle:cosplay -intitle:Flygdagarna "
                query += "-intitle:Melodifestivalen -intitle:NärCon -intitle:landsmöte"
                data = await request(
                    client, {"list": "search", "srsearch": query, "srnamespace": 6, "srlimit": 100}
                )
                write_new(search, data)
            ids.update(
                str(r["pageid"])
                for r in json.loads(search.read_text()).get("query", {}).get("search", [])
            )
        old = {
            str(r["pageid"])
            for p in (PUBLIC / "metadata").glob("info-*.json")
            for r in json.loads(p.read_text()).get("query", {}).get("pages", {}).values()
        }
        ordered = sorted(ids - old)
        for start in range(0, len(ordered), 50):
            batch = ordered[start : start + 50]
            path = (
                SOURCES
                / f"commons-info-{hashlib.sha256('|'.join(batch).encode()).hexdigest()}.json"
            )
            if not path.exists():
                data = await request(
                    client,
                    {
                        "pageids": "|".join(batch),
                        "prop": "imageinfo",
                        "iiprop": "url|mime|size|extmetadata",
                        "iiurlwidth": 1280,
                    },
                )
                write_new(path, data)
    print("Free-source metadata checkpoints ready:", SOURCES, flush=True)


def creator_key(value: str) -> str:
    return value.casefold().strip()


def cell(row: dict[str, Any], metres: int = 250) -> str:
    import utm

    x, y, _, _ = utm.from_latlon(row["lat"], row["lng"])
    return f"{int(x // metres)}:{int(y // metres)}"


def proposals(items: list[dict[str, Any]], cap: int = 36) -> list[dict[str, Any]]:
    """Geography/heading only propose pairs; visual review must confirm a shared scene."""
    items = sorted(items, key=lambda r: r["uid"])
    options = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if a["heading"] is None or b["heading"] is None:
                continue
            heading = abs((a["heading"] - b["heading"] + 180) % 360 - 180)
            distance = distance_m(a["lat"], a["lng"], b["lat"], b["lng"])
            if 5 <= distance <= 40 and 20 <= heading <= 70:
                cross_day = (
                    a["sequence"] != b["sequence"]
                    and abs(a["captured_at"] - b["captured_at"]) >= 86400000
                )
                key = hashlib.sha256((a["uid"] + ":" + b["uid"]).encode()).hexdigest()
                options.append((str(int(not cross_day)) + key, a, b, distance, heading))
    buckets: dict[str, list[Any]] = defaultdict(list)
    for option in sorted(options, key=lambda r: r[0]):
        buckets[cell(option[1])].append(option)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for _turn in range(2):
        for place in sorted(buckets):
            for _, a, b, distance, heading in buckets[place]:
                if a["uid"] in used or b["uid"] in used:
                    continue
                selected.append(
                    {
                        "id": f"pair-{len(selected) + 1:03d}",
                        "a": a["uid"],
                        "b": b["uid"],
                        "place_group": place,
                        "distance_m": distance,
                        "heading_difference": heading,
                        "capture_gap_days": abs(a["captured_at"] - b["captured_at"]) / 86400000,
                        "pair_type": "cross_day"
                        if a["sequence"] != b["sequence"]
                        and abs(a["captured_at"] - b["captured_at"]) >= 86400000
                        else "within_capture_viewpoint",
                    }
                )
                used.update((a["uid"], b["uid"]))
                break
            if len(selected) == cap:
                return selected
    return selected


def balanced(items: list[dict[str, Any]], cap: int, per_creator: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=lambda r: hashlib.sha256(r["uid"].encode()).hexdigest()):
        buckets[cell(item, 1000)].append(item)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    used_cells: set[str] = set()
    for turn in range(max((len(v) for v in buckets.values()), default=0)):
        for key in sorted(buckets):
            if turn >= len(buckets[key]):
                continue
            item = buckets[key][turn]
            if counts[creator_key(item["creator"])] >= per_creator or cell(item, 100) in used_cells:
                continue
            selected.append(item)
            counts[creator_key(item["creator"])] += 1
            used_cells.add(cell(item, 100))
            if len(selected) == cap:
                return selected
    return selected


async def acquire() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "proposal.json"
    if frozen.exists():
        saved = json.loads(frozen.read_text())
        for row in saved["images"]:
            if file_hash(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("Saved pilot image changed")
        print("Validated existing pilot proposal; review before freezing")
        return
    protected = json.loads((PUBLIC / "manifest.json").read_text())["queries"]
    central = json.loads((CENTRAL / "manifest.json").read_text())
    old = Path("data/mapillary_eval/20261106T095121.923253Z/manifest.json")
    protected += json.loads(old.read_text())["queries"] + central["queries"]
    sequences = set(central["protected_previous_query_sequences"]) | set(
        central["reserved_query_sequences"]
    )
    rows = {
        str(r["id"]): r
        for f in SOURCES.glob("mapillary-*.json")
        for r in json.loads(f.read_text()).get("data", [])
    }
    map_items = []
    for raw in rows.values():
        item = normalized(raw, BBOX)
        if not item or item["sequence"] in sequences:
            continue
        day = datetime.fromtimestamp(item["captured_at"] / 1000, UTC).date()
        if not SINCE <= day <= UNTIL:
            continue
        map_items.append(
            {
                **item,
                "uid": "mapillary-" + item["image_id"],
                "provider": "Mapillary",
                "capture_date": day.isoformat(),
                "download_url": item["thumb_url"],
            }
        )
    training = [r for r in map_items if creator_key(r["creator"]) in TRAIN_CREATORS]
    pairs = proposals(training)
    paired_ids = {p[key] for p in pairs for key in ("a", "b")}
    pair_images = [r for r in training if r["uid"] in paired_ids]
    map_validation = balanced(
        [r for r in map_items if creator_key(r["creator"]) not in TRAIN_CREATORS], 24, 12
    )
    # Fresh ordinary photos: no previously downloaded proposal or test image, and
    # no photographer shared with the proposed Mapillary training set.
    seen_ids = {
        p.stem
        for parent in [PUBLIC, Path("data/public_eval/recent_linkoping_v1")]
        for p in (parent / "photos").glob("*.jpg")
    }
    raw_commons = {
        str(r["pageid"]): r
        for f in list((PUBLIC / "metadata").glob("info-*.json"))
        + list(SOURCES.glob("commons-info-*.json"))
        for r in json.loads(f.read_text()).get("query", {}).get("pages", {}).values()
    }
    commons = []
    for page in raw_commons.values():
        item = candidate(page, SINCE, UNTIL)
        if (
            item
            and item["image_id"] not in seen_ids
            and creator_key(item["creator"]) not in TRAIN_CREATORS
        ):
            commons.append({**item, "uid": item["image_id"]})
    public = balanced(commons, 32, 6)
    selected = pair_images + map_validation + public
    protocol = {
        "since": SINCE.isoformat(),
        "until": UNTIL.isoformat(),
        "train_creators": sorted(TRAIN_CREATORS),
        "pairs": pairs,
        "pair_selection": "Viewpoint pairs5–40m/20–70degrees; prefer cross-day/sequences; "
        "retain within-capture pairs as a separate training tier, not independent evaluation; "
        "<=2 non-reused-image pairs per250m group; no model predictions",
        "fresh_holdout_selection": "Unseen image IDs; photographers disjoint from pair training; "
        "1km round robin;<=1/100m cell;<=6/photographer",
        "original66_sha256": file_hash(PUBLIC / "manifest.json"),
        "source_metadata_sha256": {str(p): file_hash(p) for p in sorted(SOURCES.glob("*.json"))},
        "selected_ids": [r["uid"] for r in selected],
    }
    if (ROOT / "protocol.json").exists():
        if json.loads((ROOT / "protocol.json").read_text()) != protocol:
            raise ValueError("Pilot proposal protocol differs")
    else:
        write_new(ROOT / "protocol.json", protocol)
    (ROOT / "photos").mkdir(exist_ok=True)
    records: dict[str, Any] = {}
    rejected: Counter[str] = Counter()
    async with httpx.AsyncClient(headers={"User-Agent": UA}, timeout=45) as client:
        for n, item in enumerate(selected, 1):
            path = ROOT / "photos" / f"{item['uid']}.jpg"
            if not path.exists():
                response = await client.get(item["download_url"])
                response.raise_for_status()
                from io import BytesIO

                with Image.open(BytesIO(response.content)) as image:
                    pixels = ImageOps.exif_transpose(image).convert("RGB")
                    pixels.save(path, format="JPEG", quality=95)
            sig = signature(path)
            if duplicate(sig, protected):
                rejected["similar_to_protected_query"] += 1
            else:
                records[item["uid"]] = {
                    **{k: v for k, v in item.items() if k not in ("download_url", "thumb_url")},
                    **sig,
                    "path": str(path.relative_to(ROOT)),
                }
            if n % 20 == 0 or n == len(selected):
                print(f"Pilot photos {n}/{len(selected)}; accepted {len(records)}", flush=True)
            await asyncio.sleep(0.3)
    valid_pairs = [
        p
        for p in pairs
        if p["a"] in records
        and p["b"] in records
        and not duplicate(records[p["a"]], [records[p["b"]]])
    ]
    train_ids = {p[k] for p in valid_pairs for k in ("a", "b")}
    holdouts: list[dict[str, Any]] = []
    for row in map_validation + public:
        if row["uid"] not in records:
            continue
        if duplicate(
            records[row["uid"]],
            [records[i] for i in train_ids] + [records[r["uid"]] for r in holdouts],
        ):
            rejected["holdout_train_or_holdout_near_duplicate"] += 1
            continue
        holdouts.append(row)
    write_new(
        frozen,
        {
            "kind": "recent_cross_view_pilot_proposal",
            "review_required": True,
            "protocol_sha256": file_hash(ROOT / "protocol.json"),
            "images": list(records.values()),
            "pair_candidates": valid_pairs,
            "fresh_commons_ids": [
                r["uid"] for r in holdouts if r["provider"] == "Wikimedia Commons"
            ],
            "mapillary_validation_ids": [
                r["uid"] for r in holdouts if r["provider"] == "Mapillary"
            ],
            "rejections": dict(rejected),
        },
    )
    print(
        "PAIR_PROPOSAL",
        frozen,
        "pairs",
        len(valid_pairs),
        "fresh Commons",
        sum(r["provider"] == "Wikimedia Commons" for r in holdouts),
        flush=True,
    )


def freeze() -> None:
    """Publish only independently reviewed positives and outdoor holdout photos."""
    proposal = json.loads((ROOT / "proposal.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if proposal["protocol_sha256"] != file_hash(ROOT / "protocol.json"):
        raise ValueError("Pair proposal and acquisition protocol differ")
    if file_hash(PUBLIC / "manifest.json") != protocol["original66_sha256"]:
        raise ValueError("Original held-out66 manifest changed")
    review = json.loads((ROOT / "review.json").read_text())
    if not review.get("reviewer") or not review.get("reviewed_at"):
        raise ValueError("Record actual visual review before freezing")
    images = {r["uid"]: r for r in proposal["images"]}
    known_pairs = {p["id"]: p for p in proposal["pair_candidates"]}
    accepted = review["positive_pair_ids"]
    if not accepted or len(set(accepted)) != len(accepted) or not set(accepted) <= set(known_pairs):
        raise ValueError("Reviewed positive pair IDs must be nonempty, unique and proposed")
    pairs = [known_pairs[p] for p in accepted]
    groups = sorted({p["place_group"] for p in pairs})
    # Keep every image in a spatial group in one fold. This is pilot validation,
    # not a claim of a large independent benchmark.
    validation_groups = set(
        sorted(groups, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[::4]
    )
    for p in pairs:
        p["split"] = "validation" if p["place_group"] in validation_groups else "train"
    val_ids = {p[k] for p in pairs if p["split"] == "validation" for k in ("a", "b")}
    buffered_out = [
        p["id"]
        for p in pairs
        if p["split"] == "train"
        and any(
            distance_m(images[p[k]]["lat"], images[p[k]]["lng"], images[v]["lat"], images[v]["lng"])
            < 100
            for k in ("a", "b")
            for v in val_ids
        )
    ]
    pairs = [p for p in pairs if p["id"] not in buffered_out]
    if not any(p["split"] == "train" for p in pairs):
        raise ValueError("Need multiple reviewed place groups for a training split")
    fresh = review["fresh_commons_ids"]
    if not set(fresh) <= set(proposal["fresh_commons_ids"]) or len(set(fresh)) != len(fresh):
        raise ValueError("Unknown or repeated fresh Commons photo")
    map_ids = review["mapillary_validation_ids"]
    if not set(map_ids) <= set(proposal["mapillary_validation_ids"]) or len(set(map_ids)) != len(
        map_ids
    ):
        raise ValueError("Unknown validation photo")
    train_ids = {p[k] for p in pairs for k in ("a", "b")}
    if train_ids & set(fresh + map_ids):
        raise ValueError("Training and holdout images overlap")
    if {creator_key(images[i]["creator"]) for i in train_ids} & {
        creator_key(images[i]["creator"]) for i in fresh + map_ids
    }:
        raise ValueError("Training and holdout photographers overlap")
    used = train_ids | set(fresh + map_ids)
    gallery = json.loads(Path("data/linkoping_dense/city_v1/manifest.json").read_text())[
        "references"
    ]
    if set(map_ids) & {"mapillary-" + r["image_id"] for r in gallery}:
        raise ValueError("Mapillary holdout image is already in the old reference index")
    for uid in fresh:
        if duplicate(images[uid], gallery):
            raise ValueError("Fresh Commons holdout resembles an existing reference image")
    for uid in used:
        if file_hash(ROOT / images[uid]["path"]) != images[uid]["sha256"]:
            raise ValueError("Reviewed image changed")
        if not SINCE <= date.fromisoformat(images[uid]["capture_date"]) <= UNTIL:
            raise ValueError("Reviewed photo is outside the required capture-date interval")
    manifest = {
        "kind": "reviewed_recent_cross_view_pilot",
        "pairs": pairs,
        "images": [images[i] for i in sorted(used)],
        "fresh_commons_ids": fresh,
        "mapillary_validation_ids": map_ids,
        "review_sha256": file_hash(ROOT / "review.json"),
        "train_validation_spatial_buffer_m": 100,
        "buffered_out_pair_ids": buffered_out,
        "protocol_sha256": file_hash(ROOT / "protocol.json"),
        "limitations": "Small reviewed pilot. GPS/heading do not prove shared content. "
        "Same-creator cross-day pairs are not independent photographers. "
        "No training result or accuracy improvement claimed.",
    }
    write_new(ROOT / "manifest.json", manifest)
    render()
    print("FROZEN_PAIRED_PILOT", ROOT / "manifest.json", flush=True)


def render() -> None:
    """Human-readable evidence with source attribution; no predictions or accuracy claims."""
    saved = json.loads((ROOT / "manifest.json").read_text())
    images = {r["uid"]: r for r in saved["images"]}
    pairs = saved["pairs"]

    def card(uid: str) -> str:
        row = images[uid]
        return (
            f"<article><img src='{thumbnail(ROOT / row['path'])}' alt='{html.escape(uid)}'>"
            f"<p>{html.escape(uid)}<br>{row['capture_date']} · "
            f"{html.escape(row['creator'])}</p><a href='"
            f"{html.escape(row['source_url'], quote=True)}'>Source</a> · "
            f"<a href='{html.escape(row['license_url'], quote=True)}'>License</a></article>"
        )

    pair_html = "".join(
        f"<h3>{p['id']} · {p['split']} · {p['pair_type']}</h3><section>"
        + card(p["a"])
        + card(p["b"])
        + "</section>"
        for p in pairs
    )
    fresh = saved["fresh_commons_ids"]
    cells = len({cell(images[uid], 1000) for uid in fresh})
    creators = len({creator_key(images[uid]["creator"]) for uid in fresh})
    cross_days = sum(p["pair_type"] == "cross_day" for p in pairs)
    pair_creators = len({creator_key(images[p[k]]["creator"]) for p in pairs for k in ("a", "b")})
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reviewed recent Linköping paired-view pilot</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;
color:#17253c;background:#f8f9fb}}p{{line-height:1.6}}section{{display:flex;flex-wrap:wrap;gap:16px}}
article{{background:white;padding:12px;width:320px}}img{{width:320px;height:190px;
object-fit:contain}}a{{color:#165ca3}}</style>
<h1>Recent Linköping paired-view pilot</h1>
<p>{len(pairs)} visually reviewed pairs: {sum(p["split"] == "train" for p in pairs)} training,
{sum(p["split"] == "validation" for p in pairs)} spatial validation. {cross_days} cross-day pairs;
the rest are same-capture viewpoint changes. Pair photographer keys: {pair_creators}.
Same-session pairs are training examples, not independent tests.</p>
<p>{len(fresh)} new Commons test photos from {creators} photographers across {cells} occupied
1km cells. Test photographer keys are disjoint from training. Dates and image
hashes are frozen; query metadata is stripped. Original66 photos remain untouched.</p>
<p>This is a small pilot, not enough to claim robust fine-tuning or general Sweden
accuracy. No new model training or predictions on this holdout have been run.
GPS/heading labels are unverified; quotas do not create a random city sample.</p>
<p><a href='manifest.json'>Frozen manifest</a> · <a href='review.json'>Review decisions</a>
 · <a href='protocol.json'>Acquisition protocol</a></p>
<h2>Verified positive pairs</h2>{pair_html}
<h2>New ordinary-photo holdout</h2><section>{"".join(card(uid) for uid in fresh)}</section>
<h2>Separate Mapillary capture validation</h2><p>These vehicle-mounted motorway views
are secondary capture validation, not ordinary-phone-photo test results.</p>
<section>{"".join(card(uid) for uid in saved["mapillary_validation_ids"])}</section></html>"""
    path = ROOT / "report.html"
    if not path.exists():
        path.write_text(page)
    print("PAIRED_PILOT_REPORT", path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["discover", "acquire", "freeze", "report"])
    args = parser.parse_args()
    if args.action == "discover":
        asyncio.run(discover())
    elif args.action == "acquire":
        asyncio.run(acquire())
    elif args.action == "freeze":
        freeze()
    else:
        render()
