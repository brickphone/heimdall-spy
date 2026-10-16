"""Check indexing safeguards without loading a real model or writing to Qdrant."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/index_vectors.py"


@pytest.mark.parametrize("populated,invalid", [(True, False), (False, True), (False, False)])
def test_index_guards_and_portable_paths(
    tmp_path: Path, monkeypatch, populated: bool, invalid: bool
) -> None:
    metadata = tmp_path / "data/metadata"
    raw = tmp_path / "data/raw"
    metadata.mkdir(parents=True)
    raw.mkdir()
    (raw / "123.jpg").write_bytes(b"synthetic placeholder for mocked embedder")
    pd.DataFrame(
        [
            {
                "image_id": "123",
                "lat": 58.41,
                "lng": 15.62,
                "s2_cell": "fixed",
                "local_path": "Z:/nonportable/123.jpg",
            }
        ]
    ).to_parquet(metadata / "images.parquet")
    uploaded: list[int] = []

    class Store:
        def __init__(self, settings):
            self.collection_name = "offline"
            self.client = SimpleNamespace(
                count=lambda **kwargs: SimpleNamespace(count=int(populated))
            )

        def create_collection(self) -> None:
            pass

        def upsert_embeddings(self, embeddings, payloads) -> None:
            uploaded.append(len(payloads))

    class Embedder:
        def __init__(self, settings):
            assert not populated, "Model should not be loaded for a populated index"

        def embed_batch(self, paths):
            assert paths == [raw / "123.jpg"]
            vectors = np.zeros((1, 768), dtype=np.float32)
            vectors[0, 0] = float("nan") if invalid else 1.0
            return vectors

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("swespy.vector_store.qdrant_client.VectorStore", Store)
    monkeypatch.setattr("swespy.model.embedder.CLIPEmbedder", Embedder)
    if populated or invalid:
        with pytest.raises(RuntimeError):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        assert not uploaded
    else:
        runpy.run_path(str(SCRIPT), run_name="__main__")
        assert uploaded == [1]
