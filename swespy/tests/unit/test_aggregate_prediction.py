import numpy as np

from swespy.vector_store.qdrant_client import VectorStore


def test_aggregate_prediction_weighted_average() -> None:
    results = [
        {"lat": 59.0, "lng": 18.0, "similarity_score": 0.8, "s2_cell": "a"},
        {"lat": 60.0, "lng": 19.0, "similarity_score": 0.2, "s2_cell": "b"},
    ]

    aggregated = VectorStore.aggregate_prediction(results)

    expected_lat = float(np.dot([0.8, 0.2], [59.0, 60.0]) / (0.8 + 0.2))
    expected_lng = float(np.dot([0.8, 0.2], [18.0, 19.0]) / (0.8 + 0.2))
    assert np.isclose(aggregated["lat"], expected_lat)
    assert np.isclose(aggregated["lng"], expected_lng)
    assert np.isclose(aggregated["confidence"], 0.5)
