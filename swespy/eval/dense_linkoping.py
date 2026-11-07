"""Acquire a bounded dense central-Linkoping gallery and a separate sequence holdout.

No model inference, training or index writes. Format labels for the official VPR
evaluator and keep all imagery/provenance under ignored data/linkoping_dense/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from eval.common import duplicate, record, signature, write_new
from eval.metrics import distance_m
from swespy.config import get_settings
from swespy.data.mapillary import MapillaryClient, valid_image
from swespy.model.megaloc import file_hash

BBOX = (15.618, 58.406, 15.631, 58.413)
FIELDS = "id,sequence,captured_at,geometry,creator,compass_angle,is_pano,thumb_1024_url"
OLD_MANIFEST = Path("data/mapillary_eval/20261106T095121.923253Z/manifest.json")


def normalized(
    raw: dict[str, Any], bbox: tuple[float, float, float, float] = BBOX,
) -> dict[str, Any] | None:
    """Require nonpanoramic imagery and attribution; retain camera heading when available."""
    import utm

    if raw.get("is_pano") is not False:
        return None
    item = record({**raw, "thumb_2048_url": raw.get("thumb_1024_url")})
    if item is None or not item["thumb_url"]:
        return None
    west, south, east, north = bbox
    if not west <= item["lng"] <= east or not south <= item["lat"] <= north:
        return None
    angle = raw.get("compass_angle")
    heading = (float(angle) % 360
               if isinstance(angle, int | float) and math.isfinite(angle) else None)
    easting, northing, zone, letter = utm.from_latlon(item["lat"], item["lng"])
    return {**item, "heading": heading, "easting": easting, "northing": northing,
            "utm_zone": zone, "utm_letter": letter}


def split_metadata(
    items: list[dict[str, Any]], old_reference_sequences: set[str],
    old_query_sequences: set[str], cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Reserve capture sequences before viewing images/predictions; spatially balance references."""
    new_sequences = {r["sequence"] for r in items} - old_reference_sequences - old_query_sequences
    ordered_sequences = sorted(new_sequences, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    reserve_count = min(len(ordered_sequences), max(12, len(new_sequences) // 4))
    reserved = set(ordered_sequences[:reserve_count])
    if not reserved:
        raise ValueError("No new capture sequences available for an independent query set")
    queries = [r for r in items if r["sequence"] in reserved]
    bins: dict[tuple[int, int, int], dict[str, Any]] = {}
    for item in sorted(items, key=lambda r: (-r["captured_at"], r["image_id"])):
        if item["sequence"] in reserved | old_query_sequences:
            continue
        heading_bin = int(item["heading"] // 45) if item["heading"] is not None else -1
        key = (int(item["easting"] // 10), int(item["northing"] // 10), heading_bin)
        bins.setdefault(key, item)
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for (x, y, _), item in sorted(bins.items()):
        cells[x, y].append(item)
    rng = random.Random(42)
    keys = sorted(cells)
    rng.shuffle(keys)
    for values in cells.values():
        rng.shuffle(values)
    gallery = []
    for round_index in range(8):
        for cell in keys:
            if round_index < len(cells[cell]):
                gallery.append(cells[cell][round_index])
                if len(gallery) == cap:
                    break
        if len(gallery) == cap:
            break
    queries.sort(key=lambda r: hashlib.sha256(r["image_id"].encode()).hexdigest())
    return gallery, queries, reserved


async def metadata(api: MapillaryClient, root: Path) -> list[dict[str, Any]]:
    path = root / "metadata.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["bbox"] != list(BBOX) or saved["budget"] != 6000:
            raise ValueError("Cached metadata scope differs")
        return saved["items"]
    west, south, east, north = BBOX
    seen = 0
    items: dict[str, dict[str, Any]] = {}
    for x in range(3):
        for y in range(3):
            box = (west + (east-west)*x/3, south + (north-south)*y/3,
                   west + (east-west)*(x+1)/3, south + (north-south)*(y+1)/3)
            count = 0
            cursor = None
            while count < 667 and seen < 6000:
                limit = min(200, 667-count, 6000-seen)
                params: dict[str, Any] = {
                    "bbox": ",".join(map(str, box)), "fields": FIELDS, "limit": limit,
                }
                if cursor:
                    params["after"] = cursor
                payload = await api._request("/images", params)
                page = payload.get("data", [])[:limit]
                seen += len(page)
                count += len(page)
                for raw in page:
                    item = normalized(raw)
                    if item:
                        items[item["image_id"]] = item
                cursor = payload.get("paging", {}).get("cursors", {}).get("after")
                if not page or not cursor:
                    break
            print(f"Dense grid {x*3+y+1}/9: {seen} records examined, "
                  f"{len(items)} valid nonpanoramic candidates", flush=True)
    write_new(path, {"bbox": BBOX, "budget": 6000, "examined": seen, "items": list(items.values())})
    return list(items.values())


def formatted_link(root: Path, item: dict[str, Any], role: str) -> str:
    directory = root / "images/test" / role
    directory.mkdir(parents=True, exist_ok=True)
    name = (f"@{item['easting']:.3f}@{item['northing']:.3f}@{item['utm_zone']}@"
            f"{item['utm_letter']}@{item['lat']:.7f}@{item['lng']:.7f}@{item['image_id']}@.jpg")
    path = directory / name
    original = root / "originals" / f"{item['image_id']}.jpg"
    if path.is_symlink():
        if path.resolve() != original.resolve():
            raise ValueError("Formatted image link points to unexpected source")
    elif path.exists():
        if not os.path.samefile(path, original):
            raise ValueError("Refusing to replace an existing formatted image")
    else:
        # Hard links retain coordinate-bearing filenames when a dataset is moved.
        os.link(original, path)
    return str(path.relative_to(root))


async def prepare(root: Path, gallery_cap: int, query_target: int) -> Path:
    settings = get_settings()
    if not settings.mapillary_access_token.strip():
        raise ValueError("Private Mapillary token required")
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "manifest.json"
    if frozen.exists():
        saved = json.loads(frozen.read_text())
        for item in saved["references"] + saved["queries"]:
            if file_hash(root / item["path"]) != item["sha256"]:
                raise ValueError("Frozen dense image changed")
        print("Reusing frozen dense dataset:", frozen, flush=True)
        return frozen
    old = json.loads(OLD_MANIFEST.read_text())
    old_query_sequences = {q["sequence"] for q in old["queries"]}
    reference_sequences = set(old["reference_sequences"])
    old_ids = {p.stem for p in Path("data/raw").glob("*.jpg")}
    old_ids |= {q["image_id"] for q in old["queries"]}
    # The retired CLIP index can no longer be re-fingerprinted; record its frozen hash.
    before = old["index_sha256"]
    async with MapillaryClient(settings) as api:
        items = await metadata(api, root)
        gallery, candidates, reserved = split_metadata(
            items, reference_sequences, old_query_sequences, gallery_cap
        )
        if not gallery:
            raise ValueError("No independent gallery images available")
        print(f"Selected {len(gallery)} gallery images; "
              f"reserved {len(reserved)} query sequences", flush=True)
        originals = root / "originals"
        originals.mkdir(exist_ok=True)
        semaphore = asyncio.Semaphore(6)
        completed = 0

        async def download(item: dict[str, Any]) -> Path:
            path = originals / f"{item['image_id']}.jpg"
            if path.exists() and valid_image(path):
                return path
            local = Path("data/raw") / path.name
            if local.is_file() and valid_image(local):
                shutil.copyfile(local, path)
            else:
                await api.download_image(item["thumb_url"], path)
            return path

        async def worker(item: dict[str, Any]) -> None:
            nonlocal completed
            async with semaphore:
                await download(item)
                completed += 1
                if completed % 100 == 0 or completed == len(gallery):
                    print(f"Gallery downloaded {completed}/{len(gallery)}", flush=True)

        await asyncio.gather(*(worker(item) for item in gallery))
        gallery_signatures = []
        references = []
        for item in gallery:
            sig = await asyncio.to_thread(signature, originals / f"{item['image_id']}.jpg")
            gallery_signatures.append(sig)
            references.append({**{k: v for k, v in item.items() if k != "thumb_url"}, **sig,
                               "path": formatted_link(root, item, "database")})
        queries: list[dict[str, Any]] = []
        query_signatures: list[dict[str, Any]] = []
        per_sequence: Counter[str] = Counter()
        rejects: Counter[str] = Counter()
        downloaded = 0
        for item in candidates:
            if len(queries) >= query_target or downloaded >= query_target * 3:
                break
            if item["image_id"] in old_ids or per_sequence[item["sequence"]] >= 3:
                rejects["previous_image_or_sequence_cap"] += 1
                continue
            if any(distance_m(item["lat"], item["lng"], q["lat"], q["lng"]) < 40
                   for q in queries):
                rejects["query_spacing"] += 1
                continue
            path = await download(item)
            downloaded += 1
            sig = await asyncio.to_thread(signature, path)
            if duplicate(sig, gallery_signatures + query_signatures):
                rejects["near_duplicate"] += 1
                path.unlink()
                continue
            queries.append({**{k: v for k, v in item.items() if k != "thumb_url"}, **sig,
                            "path": formatted_link(root, item, "queries")})
            query_signatures.append(sig)
            per_sequence[item["sequence"]] += 1
            print(f"Dense queries accepted {len(queries)}/{query_target}", flush=True)
        if not queries:
            raise ValueError("No separate-sequence query images passed the safeguards")
        coords = np.array([[r["easting"], r["northing"]] for r in references])
        nearest = [float(np.linalg.norm(coords - [q["easting"], q["northing"]], axis=1).min())
                   for q in queries]
        counts = Counter(int(r["heading"] // 45) if r["heading"] is not None else -1
                         for r in references)
        manifest = {
            "kind": "dense_linkoping_sequence_holdout", "bbox": BBOX,
            "metadata_sha256": file_hash(root / "metadata.json"),
            "original_index_sha256": before, "gallery_cap": gallery_cap,
            "requested_queries": query_target, "reference_count": len(references),
            "query_count": len(queries), "query_sequence_count": len(per_sequence),
            "reserved_query_sequences": sorted(reserved),
            "protected_previous_query_sequences": sorted(old_query_sequences),
            "selection": "10m XY cells x 45-degree heading bins; spatial round robin; seed 42",
            "query_selection": "sequence holdout; >=40m spacing; <=3/sequence; "
                               "no prediction filter",
            "duplicate_screen": "SHA256; pHash <=6 or dHash <=5 against gallery/queries",
            "gallery_heading_bins": dict(counts), "rejections": dict(rejects),
            "coverage": {str(radius): sum(d <= radius for d in nearest)
                         for radius in (25, 100, 500)},
            "references": references, "queries": queries,
            "limitations": ["Same-provider street-only experiment; GPS not manually verified",
                            "Capture headings may contain errors",
                            "Pretraining overlap not audited; development experiment"],
        }
        if {r["sequence"] for r in references} & set(per_sequence):
            raise RuntimeError("Reference/query sequence leakage")
        if {r["sequence"] for r in references} & old_query_sequences:
            raise RuntimeError("New gallery would contaminate the previous query set")
        write_new(frozen, manifest)
        print("DENSE_MANIFEST", frozen, flush=True)
        print("Coverage counts:", manifest["coverage"], "of", len(queries), flush=True)
        return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/linkoping_dense/central_v1"))
    parser.add_argument("--gallery-cap", type=int, default=1200)
    parser.add_argument("--queries", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.gallery_cap <= 1500 or not 1 <= args.queries <= 50:
        parser.error("Gallery cap must be 1–1500 and queries 1–50")
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    asyncio.run(prepare(args.root, args.gallery_cap, args.queries))


if __name__ == "__main__":
    main()
