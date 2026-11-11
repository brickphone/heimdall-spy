"""The single inference path: MegaLoc descriptor -> gallery search -> ranked locations.

The API and the evaluation scripts both call ``GeoLocator.locate_vector``; only how the
descriptor is obtained differs (in-memory upload versus a hash-keyed cache of the same
pixels).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from swespy.config import LINKOPING_AREA, in_area
from swespy.model.megaloc import file_hash, same_descriptor_space

LOAD_CHUNK = 4096
REFERENCE_FIELDS = (
    "image_id", "lat", "lng", "creator", "source", "source_url", "license_url", "capture_year",
)


def vectors_file(manifest_path: Path) -> tuple[Path, Path]:
    """The memory-mapped descriptor matrix of a gallery and its checksum sidecar."""
    stem = manifest_path.with_suffix("")
    return stem.with_name(stem.name + ".vectors.npy"), stem.with_name(stem.name + ".vectors.json")


def build_vectors(folder: Path, refs: list[dict[str, Any]], keep: list[int], dim: int,
                  path: Path, sidecar: Path, key: str) -> None:
    """Write the kept references' descriptors, in index order, as one float32 matrix.

    Streams in chunks (never more than one chunk in memory) and is written to a
    temporary name first, so an interrupted build leaves no half-valid file.
    """
    tmp = path.with_name(path.name + ".tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(keep), dim))
    for start in range(0, len(keep), LOAD_CHUNK):
        for j, i in enumerate(keep[start:start + LOAD_CHUNK]):
            vector = np.load(folder / f"{refs[i]['sha256']}.npy", allow_pickle=False)
            if vector.shape != (dim,):
                raise ValueError("Descriptor dimension differs from the manifest")
            out[start + j] = vector
    out.flush()
    del out
    tmp.replace(path)
    sidecar.write_text(json.dumps({"key": key, "rows": len(keep), "dimension": dim}))


# Approximate search (experiment 15): IVF-PQ proposes candidates, which are re-scored
# exactly from the memory-mapped vectors. Build settings live in swespy/model/ann.py;
# nprobe/refine are the dev-chosen search settings.
IVF_NLIST, PQ_M, PQ_BITS = 1024, 96, 8


def ann_file(manifest_path: Path) -> tuple[Path, Path]:
    """The approximate index of a gallery and its checksum sidecar."""
    stem = manifest_path.with_suffix("")
    return stem.with_name(stem.name + ".ivfpq.faiss"), stem.with_name(stem.name + ".ivfpq.json")


def build_ann(vectors_path: Path, path: Path, sidecar: Path, key: str) -> None:
    """Build the IVF-PQ index in a separate process, without torch loaded."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "swespy.model.ann", str(vectors_path), str(path),
                    str(sidecar), key, str(IVF_NLIST), str(PQ_M), str(PQ_BITS)], check=True)


class Gallery:
    """An exact (flat L2) MegaLoc reference index restricted to the project polygon.

    Gallery v2 (descriptor cache) is searched from one memory-mapped float32 matrix
    (built once next to the manifest): its pages are file-backed and shared through the
    OS page cache, so several processes hold one copy and nothing of it is swapped.
    """

    def __init__(
        self,
        manifest_path: Path,
        polygon: tuple[tuple[float, float], ...] = LINKOPING_AREA,
        mmap: bool = True,
        search: str = "exact",
        nprobe: int = 32,
        refine: int = 10,
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
        if search not in ("exact", "ivfpq", "ivf"):
            raise ValueError("search must be exact, ivfpq or ivf")
        if search != "exact" and not (saved.get("format") == "descriptor_cache_v1" and mmap):
            raise ValueError("Approximate search needs the memory-mapped descriptor gallery")
        self.index: Any = None
        self.ann: Any = None
        self.lists: list[np.ndarray] = []
        self.coarse: Any = None
        self.search_mode, self.nprobe, self.refine = search, nprobe, refine
        self.vectors: np.ndarray | None = None
        if saved.get("format") == "descriptor_cache_v1" and mmap:
            digest = hashlib.sha256("\n".join(r["sha256"] for r in refs).encode()).hexdigest()
            if digest != saved["index_sha256"]:
                raise ValueError("Gallery reference list differs from its manifest checksum")
            # The matrix is valid only for this reference list and this polygon filter.
            key = hashlib.sha256(f"{digest}:{','.join(map(str, keep))}".encode()).hexdigest()
            path, sidecar = vectors_file(manifest_path)
            stale = not (path.exists() and sidecar.exists()
                         and json.loads(sidecar.read_text()).get("key") == key)
            if stale:
                build_vectors(Path(saved["descriptor_dir"]), refs, keep, saved["dimension"],
                              path, sidecar, key)
            self.vectors = np.load(path, mmap_mode="r")
            if self.vectors.shape != (len(keep), saved["dimension"]):
                raise ValueError("Memory-mapped descriptors differ from the manifest")
            if search in ("ivfpq", "ivf"):
                ann, ann_sidecar = ann_file(manifest_path)
                stale = not (ann.exists() and ann_sidecar.exists()
                             and json.loads(ann_sidecar.read_text()).get("key") == key)
                if stale:
                    build_ann(path, ann, ann_sidecar, key)
                self.ann = faiss.read_index(str(ann))
                if search == "ivf":
                    # The IVF-PQ index's coarse clusters, with every member scored exactly.
                    ivf = faiss.extract_index_ivf(self.ann)
                    lists = ivf.invlists
                    self.lists = [faiss.rev_swig_ptr(lists.get_ids(n), lists.list_size(n)).copy()
                                  for n in range(ivf.nlist)]
                    self.coarse = ivf.quantizer
        elif saved.get("format") == "descriptor_cache_v1":
            self.index = faiss.IndexFlatL2(saved["dimension"])
            # Gallery v2: descriptors live once, in the hash-keyed MegaLoc cache.
            digest = hashlib.sha256("\n".join(r["sha256"] for r in refs).encode()).hexdigest()
            if digest != saved["index_sha256"]:
                raise ValueError("Gallery reference list differs from its manifest checksum")
            folder = Path(saved["descriptor_dir"])
            # Stream descriptors into the index in chunks: peak memory is the index itself
            # (~3.1 GB for 91k refs) plus one small buffer, not several full copies.
            dim = saved["dimension"]
            buffer = np.empty((min(LOAD_CHUNK, len(keep)), dim), dtype=np.float32)
            for start in range(0, len(keep), LOAD_CHUNK):
                rows = keep[start:start + LOAD_CHUNK]
                for j, i in enumerate(rows):
                    vector = np.load(folder / f"{refs[i]['sha256']}.npy", allow_pickle=False)
                    if vector.shape != (dim,):
                        raise ValueError("Descriptor dimension differs from the manifest")
                    buffer[j] = vector
                self.index.add(buffer[: len(rows)])
        else:
            self.index = faiss.IndexFlatL2(saved["dimension"])
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
        self.manifest_name = manifest_path.name
        self.stage = saved.get("stage")
        self.created_at = saved.get("created_at")
        self.sealed_guard = saved.get("sealed_guard")
        self.by_image_id = {r.get("image_id"): i for i, r in enumerate(self.refs)}
        self.index_sha256: str = saved["index_sha256"]

    def _search_ivf(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Probe the ``nprobe`` nearest clusters and score all their members exactly."""
        _, probe = self.coarse.search(vectors, self.nprobe)
        return self._rescore(vectors, [np.concatenate([self.lists[c] for c in row if c >= 0])
                                       for row in probe], k)

    def _rescore(self, vectors: np.ndarray, candidates: list[np.ndarray], k: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Exact distances for each query's candidate rows; the best ``k`` per query."""
        import faiss

        indices = np.full((len(vectors), k), -1, dtype=np.int64)
        distances = np.full((len(vectors), k), np.inf, dtype=np.float32)
        for row, cand in enumerate(candidates):
            cand = np.sort(cand[cand >= 0])  # Sorted rows read the mapped file in order.
            if not len(cand):
                continue
            d, i = faiss.knn(vectors[row:row + 1], np.ascontiguousarray(self.vectors[cand]),
                             min(k, len(cand)))
            indices[row, : i.shape[1]] = cand[i[0]]
            distances[row, : d.shape[1]] = d[0]
        return indices, 1 - distances / 2

    def _search_ann(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """IVF-PQ candidates, re-scored exactly: returned similarities are exact."""

        self.ann.nprobe = self.nprobe
        _, proposed = self.ann.search(vectors, min(self.refine * k, len(self.refs)))
        return self._rescore(vectors, list(proposed), k)

    def reference(self, image_id: str) -> dict[str, Any] | None:
        i = self.by_image_id.get(image_id)
        return None if i is None else self.refs[i]

    def image_path(self, image_id: str) -> Path | None:
        """Stored reference image for ``image_id`` (looked up, never built from user input)."""
        ref = self.reference(image_id)
        if ref is None or "path" not in ref:
            return None
        path = self.root / ref["path"]
        return path if path.is_file() else None

    def search(self, vectors: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (reference indices, cosine similarities) for each query row."""
        import faiss

        faiss.omp_set_num_threads(1)
        vectors = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        k = min(depth, len(self.refs))
        if self.search_mode == "ivf":
            return self._search_ivf(vectors, k)
        if self.ann is not None:
            return self._search_ann(vectors, k)
        if self.vectors is not None:
            # Same exact squared-L2 search as IndexFlatL2, over the mapped matrix (no copy).
            distances, indices = faiss.knn(vectors, self.vectors, k)
        else:
            distances, indices = self.index.search(vectors, k)
        # Squared L2 between unit vectors is 2 - 2 cos.
        return indices, 1 - distances / 2


class Cancelled(Exception):
    """The caller cancelled a locating run (e.g. the client disconnected)."""


class GeoLocator:
    """Rank gallery references for a query and report the top candidates."""

    def __init__(self, embedder: Any, gallery: Gallery, top_k: int = 5, depth: int = 50,
                 verifier: Any = None, weight: float = 0.0, min_inliers: int = 0,
                 rerank_k: int | None = None, segmenter: Any = None,
                 variants_mode: str = "full") -> None:
        """``verifier`` (swespy.model.rerank.GeometricVerifier) enables re-ranking of the
        top ``depth`` candidates; ``weight``/``min_inliers`` come from dev-set tuning.
        ``segmenter`` (swespy.model.variants.PersonSegmenter) enables the automatic search
        variants; ``variants_mode`` is "whole", "masked" or "full" (see variants.py)."""
        if not same_descriptor_space(embedder.identity, gallery.identity):
            raise ValueError("Query encoder and gallery were built with different MegaLoc assets")
        if not 1 <= top_k <= depth:
            raise ValueError("top_k must be between 1 and the retrieval depth")
        self.embedder, self.gallery, self.top_k, self.depth = embedder, gallery, top_k, depth
        self.verifier, self.weight, self.min_inliers = verifier, weight, min_inliers
        self.rerank_k = min(rerank_k or depth, depth)
        if variants_mode not in ("whole", "masked", "full"):
            raise ValueError("variants_mode must be whole, masked or full")
        self.segmenter, self.variants_mode = segmenter, variants_mode

    def versions(self) -> dict[str, Any]:
        """What produced a result: model, gallery and re-ranking settings."""
        ident = self.embedder.identity
        out: dict[str, Any] = {
            "model": {"name": "MegaLoc", "source_revision": ident.get("source_revision"),
                      "weights_revision": ident.get("weights_revision"),
                      "weights_sha256": str(ident.get("weights_sha256", ""))[:16],
                      "device": ident.get("device")},
            "gallery": {"file": self.gallery.manifest_name, "stage": self.gallery.stage,
                        "references": len(self.gallery.refs),
                        "index_sha256": self.gallery.index_sha256[:16],
                        "created_at": self.gallery.created_at,
                        "sealed_guard": self.gallery.sealed_guard,
                        "search": getattr(self.gallery, "search_mode", "exact"),
                        **({"nprobe": self.gallery.nprobe, "refine": self.gallery.refine}
                           if getattr(self.gallery, "search_mode", "exact") != "exact" else {})},
            "retrieval_depth": self.depth,
            "rerank": {"enabled": self.verifier is not None},
            "variants": {"mode": self.variants_mode if self.segmenter else "whole",
                         **(self.segmenter.identity if self.segmenter else {})},
        }
        if self.verifier is not None:
            v = self.verifier.identity
            out["rerank"].update({
                "k": self.rerank_k, "weight": self.weight, "min_inliers": self.min_inliers,
                "extractor": v.get("extractor"), "matcher": v.get("matcher"),
                "max_keypoints": v.get("max_keypoints"),
                "lightglue_commit": v.get("lightglue_commit"), "ransac": v.get("ransac"),
            })
        return out

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

    def locate_bytes(self, contents: bytes,
                     region: dict[str, float] | None = None) -> dict[str, Any]:
        """Locate an uploaded image held only in memory (re-ranked if a verifier is set).

        ``region`` (normalised x, y, width, height) restricts both stages to that part.
        Runs the same steps as ``locate_events``, so /predict and /predict/stream agree.
        """
        result: dict[str, Any] = {}
        for event in self.locate_events(contents, region):
            if event["type"] == "_result":
                result = event["result"]
        return result

    def analyze_bytes(self, contents: bytes) -> dict[str, Any]:
        """Person share and suggested region for an upload, without retrieval."""
        from swespy.model import variants as va
        from swespy.model.imageio import decode_upload

        image = decode_upload(contents)
        if self.segmenter is None:
            return {"person_share": None, "suggested_region": None, "suggest": False,
                    "variants": [{"id": "whole", "region": dict(va.FULL)}]}
        return va.analyze(image, self.segmenter.mask(image))

    def locate_events(self, contents: bytes, region: dict[str, float] | None = None,
                      cancelled: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
        """The locating pipeline as progress events (see POST /predict/stream).

        Yields, in order: stage "retrieval"; "candidates" (the retrieval depth, locations
        and similarity only; ``id`` is the retrieval rank, 0-based); stage "verification";
        one "verified" event per re-ranked candidate as it completes; and finally an
        internal "_result" event carrying exactly what ``locate_bytes`` returns.
        ``cancelled`` is checked between steps; when it returns True, ``Cancelled`` is
        raised and no further work is done.

        With a segmenter, every search variant (variants.py) is retrieved in one batched
        forward pass and one index search; the lists are merged keeping each reference's
        best variant, and the merged top ``rerank_k`` are verified, each against the
        variant that retrieved it. The best verified score decides the winning variant.
        """
        from swespy.model import rerank as rr
        from swespy.model import variants as va
        from swespy.model.imageio import decode_upload

        def check() -> None:
            if cancelled():
                raise Cancelled

        timings: dict[str, float] = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round((now - clock) * 1000, 1)
            clock = now

        yield {"type": "stage", "stage": "retrieval"}
        # One decode for all stages: upright RGB pixels only, no EXIF/GPS metadata.
        whole = decode_upload(contents)
        mask = None
        if self.segmenter is not None and self.variants_mode != "whole":
            mask = self.segmenter.mask(whole)
            lap("segment_ms")
        check()
        variants = va.make_variants(whole, mask, region, self.variants_mode)
        lap("variants_ms")
        check()
        vectors = self.embedder.batch_images([v.image for v in variants])
        lap("embed_ms")
        indices, similarities = self.gallery.search(vectors, self.depth)
        merged = va.merge([(i.tolist(), s.tolist()) for i, s in zip(indices, similarities,
                                                                     strict=True)],
                          [v.priority for v in variants])[: self.depth]
        lap("search_ms")
        refs = self.gallery.refs
        top = refs[merged[0][0]]
        result: dict[str, Any] = {
            "predicted_lat": top["lat"],
            "predicted_lng": top["lng"],
            "prediction_method": "top-ranked reference",
            "confidence": float(np.clip(merged[0][1], 0, 1)),
            "confidence_status": "top-1 cosine similarity; not a calibrated probability",
            "candidates": [
                {"rank": rank, "similarity": s, "variant": variants[v].id,
                 **{k: refs[i].get(k) for k in REFERENCE_FIELDS}}
                for rank, (i, s, v) in enumerate(merged[: self.top_k], 1)
            ],
            "ranked_indices": [i for i, _, _ in merged],
            "ranked_similarities": [s for _, s, _ in merged],
            "ranked_variants": [v for _, _, v in merged],
        }
        yield {"type": "candidates", "items": [
            {"id": n, "lat": refs[i]["lat"], "lng": refs[i]["lng"], "similarity": s}
            for n, (i, s, _) in enumerate(merged)]}
        check()
        if self.verifier is not None:
            k = self.rerank_k
            yield {"type": "stage", "stage": "verification", "count": len(merged[:k])}
            checks = []
            pairs = [(v, refs[i]["sha256"], self.gallery.root / refs[i]["path"])
                     for i, _, v in merged[:k]]
            for n, (c, (_, s, _)) in enumerate(zip(self.verifier.verify_multi(
                    [v.image for v in variants], pairs), merged[:k], strict=True)):
                checks.append(c)
                score = rr.combined_scores([s], [c["inliers"]], self.weight)[0]
                yield {"type": "verified", "id": n, "inliers": c["inliers"], "score": score}
                check()
            lap("verify_ms")
            result = self.finalize_result(result, checks)
        yield {"type": "_result", "result": self.describe_search(result, variants, mask,
                                                                 similarities, timings)}

    def describe_search(self, result: dict[str, Any], variants: list[Any], mask: Any,
                        similarities: np.ndarray, timings: dict[str, float]) -> dict[str, Any]:
        """Add which variant won, the region searched, the person share and timings."""
        from swespy.model import variants as va

        winner = variants[result["ranked_variants"][0]]
        best = [float(s[0]) for s in similarities]
        # "Try a different region": the automatic crop (not the winner) whose own top
        # retrieval match is most similar; else the largest people-free rectangle.
        crops = sorted((n for n, v in enumerate(variants)
                        if v.id.startswith("crop:") and v is not winner), key=lambda n: -best[n])
        alternative = variants[crops[0]].region if crops else (
            va.largest_clean_rect(mask) if mask is not None else None)
        candidates = [{**c, "variant": variants[c["variant_index"]].id}
                      if "variant_index" in c else c for c in result["candidates"]]
        return {
            **result,
            "candidates": candidates,
            "searched_variant": winner.id,
            "searched_region": winner.region,
            "person_share": None if mask is None else va.person_share(mask),
            "alternative_region": alternative,
            "variants": [{"id": v.id, "region": v.region, "top_similarity": best[n]}
                         for n, v in enumerate(variants)],
            "timings_ms": timings,
        }

    def rerank_result(self, result: dict[str, Any], image: Any) -> dict[str, Any]:
        """Geometric verification of the top ``depth`` candidates, then shared finalize."""
        refs = self.gallery.refs
        order = result["ranked_indices"][: self.rerank_k]
        checks = self.verifier.verify(
            image, [(refs[i]["sha256"], self.gallery.root / refs[i]["path"]) for i in order])
        return self.finalize_result(result, checks)

    def finalize_result(self, result: dict[str, Any], checks: list[dict[str, Any]]
                        ) -> dict[str, Any]:
        """Re-ranked result from the verification of the first ``rerank_k`` candidates."""
        from swespy.model import rerank as rr

        refs = self.gallery.refs
        k = self.rerank_k
        order, sims = result["ranked_indices"][:k], result["ranked_similarities"][:k]
        variants = result.get("ranked_variants") or [0] * len(result["ranked_indices"])
        candidates = [{"index": i, "similarity": s, "inliers": c["inliers"], "variant_index": v,
                       **{k: refs[i].get(k) for k in REFERENCE_FIELDS}}
                      for i, s, c, v in zip(order, sims, checks, variants, strict=False)]
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
            "ranked_variants": [c["variant_index"] for c in f["ranked"]] + variants[k:],
        }
