"""Gallery alone vs gallery + Street View (Part 4). Dev, field and street-image sets.

Both configurations use the production rules: exact MegaLoc retrieval, the top 5
candidates by cosine are geometrically verified (ALIKED + LightGlue), then
``rerank.finalize`` (score, location, confidence). Configuration B merges the gallery's
and the Street View index's candidates by cosine before taking the top 5.

- Gallery candidates on the dev set reuse the frozen dev inliers
  (data/gallery_v2/rerank/dev_pairs_k50.json), so configuration A reproduces the frozen
  dev results; other sets verify gallery candidates on the fly.
- Street View candidates are verified against the same view fetched live from the
  Static API (pano ID + heading), in memory, discarded after matching, never feature-
  cached; each fetch reserves one Static request in the shared budget. If the budget is
  exhausted the candidate counts as unverified (0 inliers) and is reported.
- The test set is not used.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from eval.geo_eval import haversine_m, wilson_interval
from eval.gsv_build import INDEX, JOURNAL, PULL_LOG, Api, Index, LogBuffer
from eval.gsv_plan import GALLERY_INDEX, exact_top
from swespy import gsv
from swespy.config import LINKOPING_AREA, in_area
from swespy.model import rerank as rr

DEV_ROOT = Path("data/public_eval/recent_linkoping_streets_v1")
DEV_PAIRS = Path("data/gallery_v2/rerank/dev_pairs_k50.json")
STREET_ROOT = Path("data/linkoping_dense/city_eval_v1")
FIELD_ROOT = Path("data/field_photos")
OUT = Path("data/gsv/eval")
DEPTH = 10  # Candidates per index before merging; only the merged top 5 are verified.
NEAR_M = 100.0
CORRECT_M = 500.0


def load_sets() -> dict[str, list[dict[str, Any]]]:
    dev = json.loads((DEV_ROOT / "manifest.json").read_text())["queries"]
    street = json.loads((STREET_ROOT / "manifest.json").read_text())["queries"]
    field = []
    if FIELD_ROOT.exists():
        for rec in sorted(FIELD_ROOT.glob("*/record.json")):
            r = json.loads(rec.read_text())
            field.append({"image_id": rec.parent.name, "path": str(rec.parent / "photo.jpg"),
                          "lat": r["true_location"]["lat"], "lng": r["true_location"]["lng"]})
    return {
        "dev": [{**q, "path": str(DEV_ROOT / q["path"])} for q in dev],
        "street": [{**q, "path": str(STREET_ROOT / q["path"])} for q in street],
        "field": field,
    }


def open_image(path: str) -> Any:
    from swespy.model.imageio import decode_upload

    return decode_upload(Path(path).read_bytes())


def query_vectors(name: str, queries: list[dict[str, Any]], embedder: Any) -> np.ndarray:
    if name == "dev":
        cache = next(d for d in Path("data/megaloc/descriptors").iterdir()
                     if all((d / f"{q['sha256']}.npy").exists() for q in queries))
        return np.stack([np.load(cache / f"{q['sha256']}.npy") for q in queries]).astype(
            np.float32)
    if not queries:
        return np.zeros((0, 8448), dtype=np.float32)
    out = []
    for i in range(0, len(queries), 8):
        out.append(embedder.batch_images([open_image(q["path"]) for q in queries[i:i + 8]]))
    return np.concatenate(out)


class Verifier:
    """Inliers for gallery (path, cached on disk) and Street View (live, memory) refs."""

    def __init__(self, api: Api, budget: gsv.Budget, device: str = "mps"):
        self.v = rr.GeometricVerifier(device, cache_dir=Path("data/gallery_v2/rerank/features"))
        self.api = api
        self.budget = budget
        self.unverified = 0
        self.static_used = 0

    def gallery(self, query_image: Any, ref: dict[str, Any]) -> int:
        res = self.v.verify(query_image, [(ref["sha256"], GALLERY_INDEX.parent / ref["path"])])
        return int(res[0]["inliers"])

    def gsv(self, query_image: Any, view: dict[str, Any]) -> int:
        from PIL import Image

        if not self.budget.take("image", 1):
            self.unverified += 1
            return 0
        self.static_used += 1
        try:
            raw = self.api.static(view["pano_id"], view["heading"])
        except Exception:  # noqa: BLE001 - counted as unverified; key never in messages.
            self.unverified += 1
            return 0
        with Image.open(BytesIO(raw)) as im:
            ref = im.convert("RGB").copy()
        del raw
        res = self.v.verify(query_image, [("", ref)])  # No key: features never cached.
        del ref
        return int(res[0]["inliers"])


def evaluate() -> dict[str, Any]:
    from swespy.model.megaloc import MegaLocEmbedder

    sets = load_sets()
    index = json.loads(GALLERY_INDEX.read_text())
    refs = index["references_in_index_order"]
    keep = [i for i, r in enumerate(refs) if in_area(r["lng"], r["lat"], LINKOPING_AREA)]
    folder = Path(index["descriptor_dir"])
    views, gdesc = Index(INDEX).load()
    gdesc /= np.maximum(np.linalg.norm(gdesc, axis=1, keepdims=True), 1e-12)
    vlat = np.array([v["lat"] for v in views])
    vlng = np.array([v["lng"] for v in views])
    cached = json.loads(DEV_PAIRS.read_text())["queries"]
    embedder = MegaLocEmbedder(Path("data/megaloc"), "mps")
    budget = gsv.Budget(JOURNAL)
    api = Api(gsv.api_key(), budget)
    logs = LogBuffer(gsv.PullLog(PULL_LOG), api)
    verifier = Verifier(api, budget)
    report: dict[str, Any] = {"gsv_views": len(views)}
    try:
        for name, queries in sets.items():
            rows = []
            if not queries:
                report[name] = {"n": 0}
                continue
            q = query_vectors(name, queries, embedder)
            g_s, g_i = exact_top(q, refs, keep, folder, max(DEPTH, 50))
            for n, query in enumerate(queries):
                image = open_image(query["path"])
                pairs = {c["image_id"]: c for c in cached.get(query["image_id"], [])} \
                    if name == "dev" else {}

                gal = []
                for k in range(DEPTH):
                    r = refs[int(g_i[n, k])]
                    c = pairs.get(r["image_id"])
                    gal.append({"image_id": r["image_id"], "lat": r["lat"], "lng": r["lng"],
                                "similarity": float(g_s[n, k]), "source": r.get("source"),
                                "inliers": c["inliers"] if c is not None else None, "_ref": r})
                for c in gal[:5]:  # Config A: gallery top 5.
                    if c["inliers"] is None:
                        c["inliers"] = verifier.gallery(image, c["_ref"])
                a = rr.finalize([{k: v for k, v in c.items() if k != "_ref"} for c in gal[:5]],
                                rr.DEFAULT_WEIGHT, rr.DEFAULT_MIN_INLIERS, top_k=5)
                sims = gdesc @ q[n] if len(views) else np.zeros(0)
                order = np.argsort(-sims)[:DEPTH]
                sv = [{"image_id": f"gsv:{views[j]['pano_id']}:{views[j]['heading']}",
                       "lat": views[j]["lat"], "lng": views[j]["lng"],
                       "similarity": float(sims[j]), "source": "gsv",
                       "route": views[j]["route"], "inliers": None, "_view": views[j]}
                      for j in order]
                merged = sorted(gal + sv, key=lambda c: -c["similarity"])[:5]
                for c in merged:
                    if c["inliers"] is None:
                        c["inliers"] = (verifier.gsv(image, c["_view"]) if c["source"] == "gsv"
                                        else verifier.gallery(image, c["_ref"]))
                bres = rr.finalize([{k: v for k, v in c.items() if not k.startswith("_")}
                                    for c in merged],
                                   rr.DEFAULT_WEIGHT, rr.DEFAULT_MIN_INLIERS, top_k=5)
                del image
                near = float(haversine_m(query["lat"], query["lng"], vlat, vlng).min()) \
                    if len(views) else math.inf
                row = {"image_id": query["image_id"], "nearest_gsv_m": round(near, 1)}
                for label, res in (("A", a), ("B", bres)):
                    err = float(haversine_m(query["lat"], query["lng"],
                                            np.array([res["predicted_lat"]]),
                                            np.array([res["predicted_lng"]]))[0])
                    top = res["ranked"][0]
                    row[label] = {"error_m": round(err, 1),
                                  "state": res["confidence"]["state"],
                                  "top_source": top.get("source"),
                                  "top_route": top.get("route"),
                                  "gsv_in_top5": sum(c.get("source") == "gsv"
                                                     for c in res["top"])}
                rows.append(row)
            report[name] = summarize(rows)
            (OUT / f"{name}_rows.json").parent.mkdir(parents=True, exist_ok=True)
            (OUT / f"{name}_rows.json").write_text(json.dumps(rows, indent=1))
    finally:
        logs.flush()
    report["static_used_for_verification"] = verifier.static_used
    report["unverified_gsv_candidates"] = verifier.unverified
    report["budget"] = budget.summary()
    (OUT / "summary.json").write_text(json.dumps(report, indent=1))
    return report


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(k, n) if n else (float("nan"), float("nan"))
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None,
            "ci95": [round(lo, 3), round(hi, 3)]}


def block(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(rows)
    errs = [r[label]["error_m"] for r in rows]
    ok = [r for r in rows if r[label]["state"] == "ok"]
    low = [r for r in rows if r[label]["state"] != "ok"]
    correct = [r for r in rows if r[label]["error_m"] <= CORRECT_M]
    return {
        "within_100m": rate(sum(e <= 100 for e in errs), n),
        "within_500m": rate(sum(e <= 500 for e in errs), n),
        "within_1km": rate(sum(e <= 1000 for e in errs), n),
        "median_error_m": round(float(np.median(errs)), 1) if errs else None,
        "ok": {"n": len(ok), **{f"correct_{k}": v for k, v in
                                rate(sum(r[label]["error_m"] <= CORRECT_M for r in ok),
                                     len(ok)).items() if k in ("k", "rate", "ci95")}},
        "low": {"n": len(low), **{f"correct_{k}": v for k, v in
                                  rate(sum(r[label]["error_m"] <= CORRECT_M for r in low),
                                       len(low)).items() if k in ("k", "rate", "ci95")}},
        "confident_but_wrong": sum(1 for r in ok if r[label]["error_m"] > CORRECT_M),
        "correct_top_source": dict(Counter(
            (r[label]["top_source"] or "?") + (f"/{r[label]['top_route']}"
                                               if r[label]["top_route"] else "")
            for r in correct)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    near = [r for r in rows if r["nearest_gsv_m"] <= NEAR_M]
    far = [r for r in rows if r["nearest_gsv_m"] > NEAR_M]
    out = {"n": len(rows)}
    for group, rs in (("all", rows), ("near_new_gsv", near), ("others", far)):
        out[group] = {"n": len(rs), "gallery": block(rs, "A"), "gallery_plus_gsv": block(rs, "B"),
                      "changed_500m": {
                          "fixed": sum(r["A"]["error_m"] > 500 >= r["B"]["error_m"] for r in rs),
                          "broken": sum(r["A"]["error_m"] <= 500 < r["B"]["error_m"]
                                        for r in rs)}} if rs else {"n": 0}
    return out


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=1))
