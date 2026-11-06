"""Recall coverage and interruption-cache safeguards without benchmark downloads."""

import numpy as np
import pytest
import torch

from eval.paper_benchmark import DIM, DescriptorCache, recall_metrics


def test_uncovered_queries_remain_in_recall_denominator() -> None:
    predictions = np.tile(np.arange(20), (3, 1))
    positives = [np.array([4]), np.array([8]), np.array([], dtype=int)]
    result = recall_metrics(predictions, positives)
    assert result["covered_query_count"] == 2
    assert result["coverage_pct"] == pytest.approx(200 / 3)
    assert result["all_query_recall_pct"]["1"] == 0
    assert result["all_query_recall_pct"]["5"] == pytest.approx(100 / 3)
    assert result["all_query_recall_pct"]["10"] == pytest.approx(200 / 3)
    assert result["covered_query_recall_pct"]["10"] == 100


def test_cache_resumes_without_encoding_and_rejects_corruption(tmp_path) -> None:
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            output = torch.zeros((len(images), DIM))
            output[:, 0] = 1
            return output

    encoder = Encoder()
    model = DescriptorCache(encoder, tmp_path, {"model": "unit-fixture"})
    images = torch.ones((2, 3, 2, 2))
    first = model(images)
    second = model(images)
    assert torch.equal(first, second)
    assert encoder.calls == 1
    path = next(model.root.glob("*.npy"))
    np.save(path, np.zeros(DIM, dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="descriptors"):
        model(images)
