"""The report must use the same 25m planar labels as the official evaluator."""

from pathlib import Path

import numpy as np

from eval.benchmark_report import positives_for, xy


def test_positive_radius_is_25m_inclusive_and_exposes_uncovered_queries() -> None:
    database = np.array([[0.0, 0.0], [50.0, 0.0]])
    queries = np.array([[25.0, 0.0], [25.001, 0.0], [100.0, 100.0]])
    positives = positives_for(database, queries)
    assert positives[0].tolist() == [0, 1]
    assert positives[1].tolist() == [1]
    assert positives[2].tolist() == []


def test_coordinate_labels_preserve_negative_baidu_values() -> None:
    labels = xy([Path("@-0.162@-4.859@frame.jpg"), Path("@100.001@200.002@frame.jpg")])
    np.testing.assert_allclose(labels, [[-0.162, -4.859], [100.001, 200.002]])
