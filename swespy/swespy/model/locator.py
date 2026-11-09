"""The single inference path: MegaLoc descriptor -> gallery search -> ranked locations.

The API and the evaluation scripts both call ``GeoLocator.locate_vector``; only how the
descriptor is obtained differs (in-memory upload versus a hash-keyed cache of the same
pixels).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from swespy.config import LINKOPING_AREA, in_area
from swespy.model.megaloc import file_hash, same_descriptor_space

REFERENCE_FIELDS = ("image_id", "lat", "lng", "creator", "source_url", "license_url")


class Gallery:
    """An exact (flat L2) MegaLoc reference index restricted to the project polygon."""

    def __init__(
        self,
        manifest_path: Path,
        polygon: tuple[tuple[float, float], ...] = LINKOPING_AREA,
    ) -> None:
        # Match the verified benchmark's import order for the two OpenMP runtimes.
        import torch

        torch.set_num_threads(1)
        import faiss

        saved = json.loads(manifest_path.read_text())
        if saved.get("sealed_guard") == "pending":
            raise ValueError("Gallery has not passed the sealed test-set guard; run "
                             "eval.gallery_build --finalize-sealed first")
        refs = saved["references_in_index_order"]
        keep = [i for i, r in enumerate(refs) if in_area(r["lng"], r["lat"], polygon)]
        if not keep:
            raise ValueError("No gallery references lie inside the project area")
        self.index = faiss.IndexFlatL2(saved["dimension"])
        if saved.get("format") == "descriptor_cache_v1":
            # Gallery v2: descriptors live once, in the hash-keyed MegaLoc cache.
            digest = hashlib.sha256("\n".join(r["sha256"] for r in refs).encode()).hexdigest()
            if digest != saved["index_sha256"]:
                raise ValueError("Gallery reference list differs from its manifest checksum")
            folder = Path(saved["descriptor_dir"])
            vectors = np.stack([np.load(folder / f"{refs[i]['sha256']}.npy", allow_pickle=False)
                                for i in keep]).astype(np.float32)
            if vectors.shape[1] != saved["dimension"]:
                raise ValueError("Descriptor dimension differs from the manifest")
            self.index.add(vectors)
        else:
            index_path = manifest_path.parent / "gallery.faiss"
            if file_hash(index_path) != saved["index_sha256"]:
                raise ValueError("Search index checksum differs from its manifest")
            full = faiss.read_index(str(index_path))
            if full.ntotal != len(refs) or full.d != saved["dimension"]:
                raise ValueError("Search index and reference mapping differ")
            self.index.add(full.reconstruct_n(0, full.ntotal)[keep])
        self.refs: list[dict[str, Any]] = [refs[i] for i in keep]
        self.excluded_outside_area = len(refs) - len(keep)
        self.identity: dict[str, Any] = saved["model_identity"]
        self.root = manifest_path.parent  # Reference image paths are relative to this.
        self.index_sha256: str = saved["index_sha256"]

    def search(self, vectors: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (reference indices, cosine similarities) for each query row."""
        import faiss

        faiss.omp_set_num_threads(1)
        vectors = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        distances, indices = self.index.search(vectors, min(depth, len(self.refs)))
        # Squared L2 between unit vectors is 2 - 2 cos.
        return indices, 1 - distances / 2


class GeoLocator:
    """Rank gallery references for a query and report the top candidates."""

    def __init__(self, embedder: Any, gallery: Gallery, top_k: int = 5, depth: int = 50,
                 verifier: Any = None, weight: float = 0.0, min_inliers: int = 0,
                 rerank_k: int | None = None) -> None:
        """``verifier`` (swespy.model.rerank.GeometricVerifier) enables re-ranking of the
        top ``depth`` candidates; ``weight``/``min_inliers`` come from dev-set tuning."""
        if not same_descriptor_space(embedder.identity, gallery.identity):
            raise ValueError("Query encoder and gallery were built with different MegaLoc assets")
        if not 1 <= top_k <= depth:
            raise ValueError("top_k must be between 1 and the retrieval depth")
        self.embedder, self.gallery, self.top_k, self.depth = embedder, gallery, top_k, depth
        self.verifier, self.weight, self.min_inliers = verifier, weight, min_inliers
        self.rerank_k = min(rerank_k or depth, depth)

    def locate_vector(self, vector: np.ndarray) -> dict[str, Any]:
        indices, similarities = self.gallery.search(vector, self.depth)
        order, sims = indices[0].tolist(), similarities[0].tolist()
        refs = self.gallery.refs
        top = refs[order[0]]
        return {
            "predicted_lat": top["lat"],
            "predicted_lng": top["lng"],
            "prediction_method": "top-ranked reference",
            "confidence": float(np.clip(sims[0], 0, 1)),
            "confidence_status": "top-1 cosine similarity; not a calibrated probability",
            "candidates": [
                {"rank": rank, "similarity": s, **{k: refs[i].get(k) for k in REFERENCE_FIELDS}}
                for rank, (i, s) in enumerate(zip(order[: self.top_k], sims, strict=False), 1)
            ],
            # Full retrieval depth, for evaluation diagnostics only.
            "ranked_indices": order,
            "ranked_similarities": sims,
        }

    def locate_bytes(self, contents: bytes) -> dict[str, Any]:
        """Locate an uploaded image held only in memory (re-ranked if a verifier is set)."""
        result = self.locate_vector(self.embedder.embed_bytes(contents))
        if self.verifier is None:
            return result
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(contents)) as im:
            image = im.convert("RGB")
        return self.rerank_result(result, image)

    def rerank_result(self, result: dict[str, Any], image: Any) -> dict[str, Any]:
        """Geometric verification of the top ``depth`` candidates, then shared finalize."""
        from swespy.model import rerank as rr

        refs = self.gallery.refs
        k = self.rerank_k
        order, sims = result["ranked_indices"][:k], result["ranked_similarities"][:k]
        checks = self.verifier.verify(
            image, [(refs[i]["sha256"], self.gallery.root / refs[i]["path"]) for i in order])
        candidates = [{"index": i, "similarity": s, "inliers": c["inliers"],
                       **{k: refs[i].get(k) for k in REFERENCE_FIELDS}}
                      for i, s, c in zip(order, sims, checks, strict=True)]
        f = rr.finalize(candidates, self.weight, self.min_inliers, self.top_k)
        return {
            "predicted_lat": f["predicted_lat"],
            "predicted_lng": f["predicted_lng"],
            "prediction_method": "geometric re-ranking (ALIKED+LightGlue, RANSAC inliers); "
                                 "score-weighted mean of top candidates within 100 m",
            "confidence": float(np.clip(f["top"][0]["score"], 0, 1)),
            "confidence_status": "re-ranked score; see confidence_state",
            "confidence_state": f["confidence"]["state"],
            "confidence_reasons": f["confidence"]["reasons"],
            "candidates": [{"rank": r, **c} for r, c in enumerate(f["top"], 1)],
            # Verified candidates in re-ranked order, then the unverified rest of the depth.
            "ranked_indices": [c["index"] for c in f["ranked"]] + result["ranked_indices"][k:],
            "ranked_similarities": [c["similarity"] for c in f["ranked"]]
            + result["ranked_similarities"][k:],
        }
