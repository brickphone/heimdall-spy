"""Search an exported MegaLoc gallery with image pixels; no GPS metadata is read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.megaloc_compare import MegaLocEmbedder, file_hash


def search(
    image: Path, manifest_path: Path, device: str = "mps", top_k: int = 5,
) -> dict[str, Any]:
    # Match the verified benchmark's import order for the two OpenMP runtimes.
    import torch

    torch.set_num_threads(1)
    import faiss

    if not image.is_file():
        raise ValueError("Query image does not exist")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    saved = json.loads(manifest_path.read_text())
    index_path = manifest_path.parent/"gallery.faiss"
    if file_hash(index_path) != saved["index_sha256"]:
        raise ValueError("Search index checksum differs from its manifest")
    index = faiss.read_index(str(index_path))
    refs = saved["references_in_index_order"]
    if index.ntotal != len(refs) or index.d != saved["dimension"]:
        raise ValueError("Search index and reference mapping differ")
    embedder = MegaLocEmbedder(Path("data/megaloc"), device)
    if any(saved["model_identity"].get(k) != value for k, value in embedder.identity.items()):
        raise ValueError("Query encoder identity differs from indexed model/device/software")
    vector = embedder.batch([image])
    faiss.omp_set_num_threads(1)
    distances, indices = index.search(vector, min(top_k, len(refs)))
    return {
        "model": "MegaLoc", "reference_count": len(refs),
        "score_status": "cosine similarity, not calibrated location confidence",
        "matches": [{"rank": rank, "similarity": float(1-distance/2),
                     **{key: refs[int(i)][key] for key in
                        ("image_id", "lat", "lng", "creator", "source_url", "license_url")}}
                    for rank, (distance, i) in enumerate(zip(distances[0], indices[0], strict=True),
                                                       start=1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="Exported gallery-index.json")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(search(args.image, args.index, args.device, args.top_k), indent=2))


if __name__ == "__main__":
    main()
