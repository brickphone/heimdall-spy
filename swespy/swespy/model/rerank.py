"""Local-feature geometric verification for re-ranking MegaLoc candidates.

ALIKED keypoints (BSD-3-Clause) + LightGlue matcher (Apache-2.0, ALIKED weights), then
a fundamental-matrix RANSAC (OpenCV USAC-MAGSAC) on the matches. The number of RANSAC
inliers measures whether the query and a reference show the same 3D scene. SuperPoint
weights (non-commercial) are never loaded. See docs/LICENSES.md.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

LIGHTGLUE_COMMIT = "5a650c444718caffc2669dc0094cde908f42dd73"
IMAGE_SIDE = 640  # Same as stored gallery images.
MAX_KEYPOINTS = 1024
RANSAC_THRESHOLD_PX = 2.0
MIN_MATCHES_FOR_RANSAC = 8
# SHA-256 of the downloaded weights, recorded on first verified download (2026-11-08).
WEIGHT_SHA256: dict[str, str] = {
    "aliked-n16.pth": "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2",
    "aliked_lightglue_v0-1_arxiv.pth":
        "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503",
}


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class GeometricVerifier:
    """Counts RANSAC-consistent LightGlue matches between a query and references."""

    def __init__(self, device: str = "cpu", cache_dir: Path | None = None,
                 max_keypoints: int = MAX_KEYPOINTS, match_device: str | None = None) -> None:
        """``device`` runs ALIKED; ``match_device`` runs LightGlue (defaults to ``device``).

        Measured on the M1: ALIKED 0.33 s/img on MPS vs 1.29 s on CPU; LightGlue+RANSAC
        1.54 s/pair on MPS vs 0.31 s on CPU, so extract on MPS and match on CPU there.
        """
        import torch
        from lightglue import ALIKED, LightGlue

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        if match_device in (None, "auto"):
            match_device = "cpu" if device == "mps" else device
        self.device, self.match_device = device, match_device
        self.extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(match_device)
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "extractor": "ALIKED aliked-n16", "matcher": "LightGlue aliked",
            "lightglue_commit": LIGHTGLUE_COMMIT, "max_keypoints": max_keypoints,
            "extract_device": device, "match_device": match_device,
            "image_side": IMAGE_SIDE, "ransac": f"USAC_MAGSAC fundamental {RANSAC_THRESHOLD_PX}px",
            "weights_sha256": self._weight_hashes(),
        }

    @staticmethod
    def _weight_hashes() -> dict[str, str]:
        import torch

        hub = Path(torch.hub.get_dir()) / "checkpoints"
        found = {p.name: _file_sha256(p) for p in sorted(hub.glob("*.pth"))
                 if "aliked" in p.name}
        for name, digest in WEIGHT_SHA256.items():
            if found.get(name) not in (None, digest):
                raise ValueError(f"Weights {name} differ from the pinned checksum")
        return found

    def _tensor(self, image: Any) -> Any:
        import torch

        image = image.convert("RGB")
        image.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).to(self.device)

    def features(self, image: Any, key: str | None = None) -> dict[str, Any]:
        """ALIKED features for one PIL image; cached on disk under ``key`` if given."""
        import torch

        path = self.cache_dir / f"{key}.pt" if self.cache_dir and key else None
        if path is not None and path.exists():
            saved = torch.load(path, map_location=self.device, weights_only=True)
            return {k: v.to(self.device) for k, v in saved.items()}
        with torch.inference_mode():
            feats = self.extractor.extract(self._tensor(image))
        feats = {k: v for k, v in feats.items() if isinstance(v, torch.Tensor)}
        if path is not None:
            tmp = path.with_suffix(".tmp")
            torch.save({k: v.cpu() for k, v in feats.items()}, tmp)
            tmp.replace(path)
        return feats

    def inliers(self, query: dict[str, Any], reference: dict[str, Any]) -> dict[str, int]:
        """LightGlue matches and fundamental-matrix RANSAC inliers between two feature sets."""
        import cv2
        import torch

        query = {k: v.to(self.match_device) for k, v in query.items()}
        reference = {k: v.to(self.match_device) for k, v in reference.items()}
        with torch.inference_mode():
            out = self.matcher({"image0": query, "image1": reference})
        matches = out["matches"][0].cpu().numpy()
        n = int(len(matches))
        if n < MIN_MATCHES_FOR_RANSAC:
            return {"matches": n, "inliers": 0}
        p0 = query["keypoints"][0].cpu().numpy()[matches[:, 0]]
        p1 = reference["keypoints"][0].cpu().numpy()[matches[:, 1]]
        try:
            _, mask = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, RANSAC_THRESHOLD_PX,
                                             0.999, 10_000)
        except cv2.error:
            # MAGSAC asserts on degenerate match sets (no fundamental matrix): no geometry.
            return {"matches": n, "inliers": 0}
        return {"matches": n, "inliers": int(mask.sum()) if mask is not None else 0}

    def verify_iter(self, query_image: Any, references: list[tuple[str, Any]]) -> Any:
        """Yield inliers for each (cache key, PIL image or path) reference, in order.

        The query's features are extracted once, so results equal ``verify``'s.
        """
        from PIL import Image

        q = self.features(query_image)
        for key, ref in references:
            if isinstance(ref, str | Path):
                cached = self.cache_dir and (self.cache_dir / f"{key}.pt").exists()
                if cached:
                    feats = self.features(None, key)
                else:
                    with Image.open(ref) as im:
                        feats = self.features(im.copy(), key)
            else:
                feats = self.features(ref, key)
            yield self.inliers(q, feats)

    def verify(self, query_image: Any, references: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        """Inliers for each (cache key, PIL image or path) reference, plus timings."""
        started = time.perf_counter()
        results = list(self.verify_iter(query_image, references))
        if results:
            results[0]["seconds_total"] = time.perf_counter() - started
        return results


# ---------------------------------------------------------------- scoring (shared by API/eval)

# Chosen on the 66-photo dev set only (eval/rerank_dev.py tune); see
# docs/experiments/07_reranking.md. Test set not used.
DEFAULT_K = 5  # Part 1 sweep: same dev accuracy/confidence split as K=50, 4.0 s/query
DEFAULT_WEIGHT = 0.005  # Dev-tuned (rule: max dev within 100 m); grid in the report.
CLUSTER_RADIUS_M = 100.0  # Fixed a priori, not tuned.
DISAGREE_M = 500.0  # Fixed a priori, not tuned.
DEFAULT_MIN_INLIERS = 25  # Dev-calibrated: confident subset 93% within 500 m (27/66).


def combined_scores(similarities: list[float], inliers: list[int], weight: float) -> list[float]:
    """Retrieval cosine plus ``weight * log(1 + RANSAC inliers)``."""
    return [s + weight * float(np.log1p(n)) for s, n in zip(similarities, inliers, strict=True)]


def _dist_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    from math import asin, cos, radians, sin, sqrt

    p1, p2 = radians(a["lat"]), radians(b["lat"])
    h = sin((p2 - p1) / 2) ** 2 + cos(p1) * cos(p2) * sin(radians(b["lng"] - a["lng"]) / 2) ** 2
    return 2 * 6_371_000 * asin(sqrt(min(1.0, h)))


def predict_location(candidates: list[dict[str, Any]], top_n: int = 5,
                     radius_m: float = CLUSTER_RADIUS_M) -> tuple[float, float, int]:
    """Score-weighted mean of the top re-ranked candidates within ``radius_m`` of the first.

    Candidates further than ``radius_m`` from the top match are ignored as outliers.
    Weights are scores shifted to be positive. Returns (lat, lng, candidates used).
    """
    top = candidates[0]
    kept = [c for c in candidates[:top_n] if _dist_m(top, c) <= radius_m]
    floor = min(c["score"] for c in kept)
    weights = [c["score"] - floor + 1e-3 for c in kept]
    total = sum(weights)
    lat = sum(w * c["lat"] for w, c in zip(weights, kept, strict=True)) / total
    lng = sum(w * c["lng"] for w, c in zip(weights, kept, strict=True)) / total
    return lat, lng, len(kept)


def confidence_state(candidates: list[dict[str, Any]], min_inliers: int,
                     disagree_m: float = DISAGREE_M) -> dict[str, Any]:
    """'low' if the top match has < ``min_inliers`` inliers, or if another geometrically
    verified candidate (>= ``min_inliers``) among the top 5 lies more than ``disagree_m``
    from the top match (two verified scenes in different places); otherwise 'ok'.
    Unverified runners-up (few inliers) do not count as disagreement."""
    reasons = []
    top = candidates[0]
    if top["inliers"] < min_inliers:
        reasons.append(f"top match has {top['inliers']} < {min_inliers} inliers")
    rivals = [c for c in candidates[1:5]
              if c["inliers"] >= max(min_inliers, 1) and _dist_m(top, c) > disagree_m]
    if rivals:
        reasons.append(f"another verified candidate is more than {disagree_m:.0f} m away")
    return {"state": "low" if reasons else "ok", "reasons": reasons}


def rerank(candidates: list[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    """Order candidates (dicts with similarity, inliers, lat, lng) by combined score."""
    scores = combined_scores([c["similarity"] for c in candidates],
                             [c["inliers"] for c in candidates], weight)
    ranked = [{**c, "score": s} for c, s in zip(candidates, scores, strict=True)]
    return sorted(ranked, key=lambda c: -c["score"])


def finalize(candidates: list[dict[str, Any]], weight: float, min_inliers: int,
             top_k: int = 5) -> dict[str, Any]:
    """The single post-verification step used by the API and the evaluation."""
    ranked = rerank(candidates, weight)
    lat, lng, used = predict_location(ranked)
    return {"ranked": ranked, "top": ranked[:top_k], "predicted_lat": lat,
            "predicted_lng": lng, "cluster_size": used,
            "confidence": confidence_state(ranked, min_inliers)}
