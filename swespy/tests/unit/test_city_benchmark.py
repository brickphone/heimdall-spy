"""Expanded-gallery evaluation must retain coverage misses and reject query leakage."""

import json

import numpy as np
import pytest

from eval.city_benchmark import prepare, score_radii
from swespy.model.megaloc import file_hash


def test_city_metrics_include_uncovered_queries_at_each_radius():
    refs = np.array([[0.0, 0.0], [1000.0, 0.0]])
    queries = np.array([[80.0, 0.0], [2000.0, 0.0]])
    pred = np.tile(np.array([0, 1]), (2, 1))
    measured = score_radii(pred, refs, queries)
    assert measured["25"]["covered_query_count"] == 0
    assert measured["25"]["covered_query_recall_pct"]["1"] is None
    assert measured["100"]["covered_query_count"] == 1
    assert measured["100"]["all_query_recall_pct"]["1"] == 50
    assert measured["500"]["query_count"] == 2
    assert measured["500"]["all_query_recall_pct"]["10"] == 50


def test_dataset_preparation_refuses_a_query_sequence_in_references(tmp_path, monkeypatch):
    pytest.importorskip("utm", reason="Optional benchmark dependency")
    import eval.city_benchmark as module

    city, central, legacy = [tmp_path/name for name in ("city", "central", "legacy")]
    for folder in (city, central, legacy):
        folder.mkdir()

    def image(root, name, content):
        path = root/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        return {"image_id": content.decode(), "path": name, "sha256": file_hash(path),
                "sequence": "same-capture", "lat": 58.41, "lng": 15.62}

    ref = image(city, "reference.jpg", b"1")
    query = image(central, "query.jpg", b"2")
    old = image(legacy, "photos/3.jpg", b"3")
    old["filename"] = "3.jpg"
    for folder, data in ((city, {"references": [ref]}), (central, {"queries": [query]}),
                         (legacy, {"queries": [old]})):
        (folder/"manifest.json").write_text(json.dumps(data))
    monkeypatch.setattr(module, "DEFAULT_ROOT", city)
    monkeypatch.setattr(module, "CENTRAL", central)
    monkeypatch.setattr(module, "OLD_MANIFEST", legacy/"manifest.json")
    with pytest.raises(ValueError, match="capture sequences overlap"):
        prepare(tmp_path/"evaluation")
    assert not (tmp_path/"evaluation/manifest.json").exists()
