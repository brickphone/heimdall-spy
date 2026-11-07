"""Freeze a stratified failure sample and render query/nearby/retrieved evidence."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from eval.mapillary_check import thumbnail
from eval.megaloc_compare import file_hash
from eval.metrics import distance_m
from eval.public_photos import INDEX, ROOT
from eval.run_eval import write_new

REPORT = ROOT / "runs/20261106T135549400456Z/results.json"
OUT = Path("data/failure_audit/recent66_v1")


def prepare() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads(REPORT.read_text())
    index = json.loads(INDEX.read_text())
    refs = index["references_in_index_order"]
    ref_root = Path(index["dataset"])
    by_id = {r["image_id"]: r for r in refs}
    buckets = {"near100_wrong": [], "near500_wrong": [], "no500_reference": []}
    for q in result["queries"]:
        if q["error_m"] <= 500:
            continue
        key = (
            "near100_wrong"
            if q["nearest_reference_m"] <= 100
            else "near500_wrong"
            if q["nearest_reference_m"] <= 500
            else "no500_reference"
        )
        buckets[key].append(q)
    chosen = []
    for name, rows in buckets.items():
        rows.sort(key=lambda q: hashlib.sha256(("audit42:" + q["image_id"]).encode()).hexdigest())
        chosen += [{"stratum": name, "query": q} for q in rows[:5]]
    path = OUT / "sample.json"
    sample = {
        "report_sha256": file_hash(REPORT),
        "index_manifest_sha256": file_hash(INDEX),
        "selection": "First5 per geographic-failure stratum by SHA256(audit42:image_id)",
        "stratum_sizes": {k: len(v) for k, v in buckets.items()},
        "cases": chosen,
    }
    if path.exists():
        if json.loads(path.read_text()) != sample:
            raise ValueError("Frozen audit sample differs")
    else:
        write_new(path, sample)
    evidence = []
    for item in chosen:
        q = item["query"]
        distances = np.array([distance_m(q["lat"], q["lng"], r["lat"], r["lng"]) for r in refs])
        nearest = np.argsort(distances)[:2]
        predicted = q["prediction"]["top_matches"]
        best = min(predicted, key=lambda r: distance_m(q["lat"], q["lng"], r["lat"], r["lng"]))
        examples = [
            ("Query", q, ROOT / q["path"]),
            (
                f"Nearest {distances[nearest[0]]:.0f}m",
                refs[nearest[0]],
                ref_root / refs[nearest[0]]["path"],
            ),
            (
                f"Second {distances[nearest[1]]:.0f}m",
                refs[nearest[1]],
                ref_root / refs[nearest[1]]["path"],
            ),
            (
                f"Top1 {q['error_m']:.0f}m",
                by_id[predicted[0]["image_id"]],
                ref_root / by_id[predicted[0]["image_id"]]["path"],
            ),
            (
                f"Best top5 {q['best_candidate_error_m']:.0f}m",
                by_id[best["image_id"]],
                ref_root / by_id[best["image_id"]]["path"],
            ),
        ]
        for _, row, image in examples:
            if file_hash(image) != row["sha256"]:
                raise ValueError("Audit image bytes changed")
        evidence.append((item, examples))
    pages = []
    for start in range(0, len(evidence), 3):
        fig, axes = plt.subplots(3, 5, figsize=(20, 10))
        for ax in axes.ravel():
            ax.axis("off")
        for i, (_item, examples) in enumerate(evidence[start : start + 3]):
            for j, (label, row, image) in enumerate(examples):
                with Image.open(image) as im:
                    im = im.convert("RGB")
                    im.thumbnail((450, 300))
                    axes[i, j].imshow(im)
                axes[i, j].set_title(f"{start + i + 1}. {label}\n{row['image_id']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"sheet-{start // 3 + 1}.jpg", dpi=110)
        plt.close(fig)
    for i, (item, examples) in enumerate(evidence, 1):
        q = item["query"]
        cards = "".join(
            f"<div><strong>{html.escape(label)}</strong><br>"
            f"<img src='{thumbnail(image)}'><br>"
            f"<a href='{html.escape(row['source_url'], quote=True)}'>"
            f"{html.escape(row['creator'])}</a> · "
            f"<a href='{html.escape(row['license_url'], quote=True)}'>License</a></div>"
            for label, row, image in examples
        )
        pages.append(
            f"<h2>{i}. {html.escape(q['title'])}</h2>"
            f"<p>{item['stratum']} · Camera label {q['lat']:.6f}, {q['lng']:.6f}. "
            "GPS not independently verified.</p><section>" + cards + "</section>"
        )
    (OUT / "evidence.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>Linköping failure audit</title><style>body{font:15px system-ui;margin:24px}"
        "section{display:flex;gap:12px}section div{width:20%}img{width:100%;height:190px;"
        "object-fit:contain}h2{margin-top:36px}</style><h1>Frozen15-case failure audit</h1>"
        "<p>Query, nearest2 geographic references, top1 retrieval and best top5 retrieval. "
        "Selection fixed before visual inspection; all66 original test photos remain held out.</p>"
        + "".join(pages)
    )
    print("AUDIT", OUT / "evidence.html", flush=True)


if __name__ == "__main__":
    prepare()
