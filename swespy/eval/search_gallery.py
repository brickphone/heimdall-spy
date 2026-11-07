"""Search an exported MegaLoc gallery with image pixels; no GPS metadata is read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from swespy.config import get_settings
from swespy.model.geo_head import GeoHead
from swespy.model.locator import REFERENCE_FIELDS, Gallery, GeoLocator
from swespy.model.megaloc import MegaLocEmbedder, file_hash


def search(
    image: Path,
    manifest_path: Path,
    device: str = "mps",
    top_k: int = 5,
    classifier: Path | None = None,
) -> dict[str, Any]:
    if not image.is_file():
        raise ValueError("Query image does not exist")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    gallery = Gallery(manifest_path)
    embedder = MegaLocEmbedder(Path(get_settings().megaloc_root), device)
    locator = GeoLocator(embedder, gallery, top_k=top_k, depth=50)
    vector = embedder.batch([image])
    result = locator.locate_vector(vector[0])
    extra: dict[str, Any] = {}
    candidates = result["candidates"]
    if classifier:
        # Historical classifier experiment; it re-orders the same 50 retrieved candidates.
        head = GeoHead.load(classifier, gallery.identity)
        indices = np.array([result["ranked_indices"]])
        order = head.rerank(vector, indices, np.array([result["ranked_similarities"]]),
                            gallery.refs)[0][:top_k]
        sims = dict(zip(result["ranked_indices"], result["ranked_similarities"], strict=True))
        candidates = [{"rank": rank, "similarity": sims[int(i)],
                       **{k: gallery.refs[int(i)].get(k) for k in REFERENCE_FIELDS}}
                      for rank, i in enumerate(order, 1)]
        extra = {
            "classifier_locations": head.locations(vector, top_k)[0],
            "fusion_weight": head.checkpoint["fusion_weight"],
            "checkpoint_sha256": file_hash(classifier),
        }
    return {
        "model": "MegaLoc",
        "reference_count": len(gallery.refs),
        "score_status": result["confidence_status"],
        **extra,
        "matches": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="Exported gallery-index.json")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--classifier", type=Path, help="Optional trained geographic head checkpoint"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            search(args.image, args.index, args.device, args.top_k, args.classifier), indent=2
        )
    )


if __name__ == "__main__":
    main()
