"""Apply the gallery quality rules to an existing gallery and write a review sheet.

    python -m eval.quality_review \
        --out ../docs/experiments/quality_filter_review.html

Images are first resized to the gallery v2 storage size (640 px), exactly as the
download stage stores them. The sheet shows up to 30 random rejections per rule
(seed 0), each attributed (contributor, licence, source link) as CC BY-SA requires.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from eval import quality
from eval.common import signature

STORE_MAX_SIDE = 640
GALLERY = Path("data/linkoping_dense/city_v1")


def stored_version(path: Path) -> bytes:
    """The JPEG bytes gallery v2 would store for this image (640 px, quality 88)."""
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    image.thumbnail((STORE_MAX_SIDE, STORE_MAX_SIDE))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=88)
    return out.getvalue()


def review(gallery: Path, scratch: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    refs = json.loads((gallery / "manifest.json").read_text())["references"]
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in refs:
        data = stored_version(gallery / r["path"])
        tmp = scratch / "q.jpg"
        tmp.write_bytes(data)
        with Image.open(tmp) as im:
            m = quality.measures(im)
        rows.append({**r, "source": "mapillary", **m, **signature(tmp), "stored": data})
    decisions: Counter[str] = Counter()
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (-r["captured_at"], r["image_id"])):
        rule = quality.image_rule(r)
        if rule is None:
            size = quality.DUPLICATE_RADIUS_M
            cy = int(r["lat"] * 111_320 // size)
            cx = int(r["lng"] * 111_320 * math.cos(math.radians(r["lat"])) // size)
            near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for k in cells[(cx + dx, cy + dy)]]
            if quality.is_duplicate(r, near):
                rule = "near_duplicate"
                r["duplicate_of"] = next(
                    k["image_id"] for k in near if quality.is_duplicate(r, [k]))
            else:
                cells[(cx, cy)].append(r)
        r["rule"] = rule or "kept"
        decisions[r["rule"]] += 1
    return rows, decisions


def thumb(data: bytes, size: tuple[int, int] = (300, 220)) -> str:
    with Image.open(io.BytesIO(data)) as im:
        im.thumbnail(size)
        out = io.BytesIO()
        im.convert("RGB").save(out, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def sheet(rows: list[dict[str, Any]], decisions: Counter[str], out: Path, per_rule: int) -> None:
    by_id = {r["image_id"]: r for r in rows}
    sections = []
    for rule in ("blurry", "sky_or_ground_only", "near_duplicate"):
        rejected = [r for r in rows if r["rule"] == rule]
        chosen = random.Random(0).sample(rejected, min(per_rule, len(rejected)))
        cards = []
        for r in chosen:
            extra = ""
            if rule == "near_duplicate":
                kept = by_id[r["duplicate_of"]]
                extra = (f"<img src='{thumb(kept['stored'], (150, 110))}' alt='kept'>"
                         f"<br><small>kept: {html.escape(kept['image_id'])}</small>")
            cards.append(
                f"<figure><img src='{thumb(r['stored'])}' alt='{html.escape(r['image_id'])}'>"
                f"{extra}<figcaption>blur {r['blur_effect']:.3f} · edges "
                f"{r['edge_share']:.3f}<br><a href='{html.escape(r['source_url'])}'>Mapillary "
                f"{html.escape(r['image_id'])}</a> by {html.escape(r['creator'])}, "
                f"<a href='{html.escape(r['license_url'])}'>CC BY-SA 4.0</a>, resized"
                "</figcaption></figure>")
        sections.append(f"<h2>{rule} — {len(chosen)} of {len(rejected)} rejections</h2>"
                        + "".join(cards))
    thresholds = (f"blur_effect ≥ {quality.BLUR_EFFECT_THRESHOLD} (Crete et al. 2007); "
                  f"edge_share < {quality.EDGE_SHARE_THRESHOLD}; near-duplicate pHash ≤ 6 or "
                  f"dHash ≤ 5 within {quality.DUPLICATE_RADIUS_M:.0f} m")
    counts = ", ".join(f"{k}: {v}" for k, v in sorted(decisions.items()))
    out.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Quality filter review</title><style>body{font:14px system-ui;margin:16px;"
        "background:#fafafa;color:#222}figure{display:inline-block;width:300px;margin:8px;"
        "vertical-align:top;background:#fff;padding:6px;border:1px solid #ddd}"
        "figure img{max-width:100%}figcaption{font-size:12px}</style>"
        f"<h1>Quality filter review — {len(rows):,} existing references at 640 px</h1>"
        f"<p>{html.escape(counts)}</p><p>Rules: {html.escape(thresholds)}. Random sample, "
        "seed 0. Images © their Mapillary contributors, CC BY-SA 4.0, resized.</p>"
        + "".join(sections) + "</html>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery", type=Path, default=GALLERY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-rule", type=int, default=30)
    parser.add_argument("--scratch", type=Path, default=Path("data/gallery_v2/scratch"))
    args = parser.parse_args()
    rows, decisions = review(args.gallery, args.scratch)
    sheet(rows, decisions, args.out, args.per_rule)
    summary = {"references": len(rows), "decisions": dict(decisions),
               "thresholds": {"blur_effect": quality.BLUR_EFFECT_THRESHOLD,
                              "edge_share": quality.EDGE_SHARE_THRESHOLD,
                              "duplicate_radius_m": quality.DUPLICATE_RADIUS_M},
               "sheet_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
