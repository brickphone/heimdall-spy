"""Build the approximate (IVF-PQ) gallery index in its own process.

FAISS's multi-threaded training crashes (SIGSEGV) when torch's OpenMP runtime is already
loaded in the process, so ``Gallery`` runs this module as a subprocess, which imports
only numpy and FAISS:

    python -m swespy.model.ann <vectors.npy> <index.faiss> <sidecar.json> <key>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Fixed build settings (docs/experiments/15_approximate_search.md).
IVF_NLIST = 1024
PQ_M = 96
PQ_BITS = 8
IVF_SEED = 1234
ADD_CHUNK = 4096


def build(vectors_path: Path, path: Path, sidecar: Path, key: str, nlist: int = IVF_NLIST,
          pq_m: int = PQ_M, pq_bits: int = PQ_BITS) -> None:
    import faiss

    vectors = np.load(vectors_path, mmap_mode="r")
    dim = vectors.shape[1]
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, nlist, pq_m, pq_bits)
    index.by_residual = False  # Residual training would copy the whole matrix in RAM.
    index.cp.seed = IVF_SEED
    index.pq.cp.seed = IVF_SEED
    index.train(vectors)
    for start in range(0, len(vectors), ADD_CHUNK):
        index.add(np.ascontiguousarray(vectors[start:start + ADD_CHUNK]))
    tmp = path.with_name(path.name + ".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(path)
    sidecar.write_text(json.dumps({"key": key, "nlist": nlist, "pq_m": pq_m,
                                   "pq_bits": pq_bits, "seed": IVF_SEED,
                                   "by_residual": False}))


if __name__ == "__main__":
    args = sys.argv[1:]
    sizes = [int(a) for a in args[4:7]] if len(args) >= 7 else []
    build(Path(args[0]), Path(args[1]), Path(args[2]), args[3], *sizes)
