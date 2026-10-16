"""Test evaluation contracts with synthetic fixtures, never fabricated baseline results."""

import json
from pathlib import Path

import pytest
from PIL import Image

from eval.dataset import DatasetError, Photo, load_dataset
from eval.metrics import distance_m, observation, summarize
from eval.run_eval import compare_baseline, compare_snapshot, write_new


def dataset(tmp_path: Path, row: str) -> tuple[Path, Path]:
    """Create a private-format CSV and synthetic image for loader tests."""
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (10, 10), "red").save(photos / "query.png")
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,lat,lng,viewpoint_type\n" + row + "\n")
    return labels, photos


def test_missing_labels_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Missing labels"):
        load_dataset(tmp_path / "labels.csv", tmp_path)


def test_valid_private_query(tmp_path: Path) -> None:
    labels, photos = dataset(tmp_path, "query.png,58.41,15.62,window")
    loaded = load_dataset(labels, photos)
    assert len(loaded) == 1
    assert loaded[0].viewpoint_type == "window"
    assert loaded[0].path == (photos / "query.png").resolve()


@pytest.mark.parametrize(
    "row",
    [
        "query.png,nan,15.62,street",
        "query.png,91,15.62,street",
        "query.png,58.41,inf,street",
        "query.png,58.41,181,street",
        "query.png,58.41,15.62,roof",
        "query.png,,15.62,street",
        "../query.png,58.41,15.62,street",
        "C:/query.png,58.41,15.62,street",
        "missing.png,58.41,15.62,street",
        "query.png,58.41,15.62,street,extra",
    ],
)
def test_invalid_labels_rejected(tmp_path: Path, row: str) -> None:
    labels, photos = dataset(tmp_path, row)
    with pytest.raises(DatasetError):
        load_dataset(labels, photos)


def test_empty_dataset_rejected(tmp_path: Path) -> None:
    labels, photos = dataset(tmp_path, "")
    with pytest.raises(DatasetError, match="no photos"):
        load_dataset(labels, photos)


def test_corrupt_photo_rejected(tmp_path: Path) -> None:
    labels, photos = dataset(tmp_path, "query.png,58.41,15.62,street")
    (photos / "query.png").write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="Corrupt"):
        load_dataset(labels, photos)


def test_reference_leakage_rejected(tmp_path: Path) -> None:
    labels, photos = dataset(tmp_path, "query.png,58.41,15.62,street")
    reference = tmp_path / "raw"
    reference.mkdir()
    (reference / "mapillary.jpg").write_bytes((photos / "query.png").read_bytes())
    with pytest.raises(DatasetError, match="duplicates local reference"):
        load_dataset(labels, photos, reference)


def test_duplicate_queries_rejected(tmp_path: Path) -> None:
    labels, photos = dataset(tmp_path, "query.png,58.41,15.62,street")
    with labels.open("a") as stream:
        stream.write("query.png,58.41,15.62,street\n")
    with pytest.raises(DatasetError, match="Duplicate filename"):
        load_dataset(labels, photos)


def test_top_k_measures_alternatives_instead_of_averaged_pin() -> None:
    photo = Photo("query.png", Path("query.png"), 58.41, 15.62, "window", "private")
    result = observation(
        photo,
        {
            "predicted_lat": 59.41,
            "predicted_lng": 15.62,
            "confidence": 0.8,
            "top_matches": [{"lat": 58.41, "lng": 15.62}],
        },
        100,
    )
    assert result["top_k_hit"] is True
    assert result["error_m"] > 100_000
    metrics = summarize([result])
    assert metrics["overall"]["top_k_hit_pct"] == 100
    assert metrics["by_viewpoint"]["window"]["pct_within_100m"] == 0
    assert metrics["by_viewpoint"]["street"]["median_error_m"] is None


def test_metrics_thresholds_and_median() -> None:
    rows = [
        {"viewpoint_type": "street", "error_m": error, "top_k_hit": False}
        for error in (100, 500, 2000, 3000)
    ]
    result = summarize(rows)["overall"]
    assert result["median_error_m"] == 1250
    assert [result[f"pct_within_{r}m"] for r in (100, 500, 2000)] == [25, 50, 75]
    assert 111_000 < distance_m(0, 0, 1, 0) < 112_000


def test_artifacts_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_new(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_snapshot_detects_prediction_and_identity_changes() -> None:
    snapshot = {
        "identity": {
            key: "fixed"
            for key in (
                "dataset_sha256",
                "index_sha256",
                "model",
                "model_revision",
                "processor_sha256",
                "top_k",
                "s2_level",
                "embedding_dim",
            )
        },
        "predictions": [
            {"predicted_lat": 58.41, "predicted_lng": 15.62, "confidence": 0.8, "top_matches": []}
        ],
    }
    actual = json.loads(json.dumps(snapshot))
    actual["predictions"][0]["processing_time_ms"] = 900
    compare_snapshot(snapshot, actual)
    actual["predictions"][0]["predicted_lat"] += 0.01
    with pytest.raises(ValueError, match="pin changed"):
        compare_snapshot(snapshot, actual)
    actual = json.loads(json.dumps(snapshot))
    actual["identity"]["index_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        compare_snapshot(snapshot, actual)


def test_baseline_detects_viewpoint_regression_hidden_by_overall_median() -> None:
    identity = {
        key: "fixed"
        for key in ("dataset_sha256", "index_sha256", "model_revision", "processor_sha256", "top_k")
    }
    old_rows = [
        {"viewpoint_type": "street", "error_m": 100, "top_k_hit": True},
        {"viewpoint_type": "window", "error_m": 200, "top_k_hit": False},
    ]
    new_rows = [
        {"viewpoint_type": "street", "error_m": 200, "top_k_hit": True},
        {"viewpoint_type": "window", "error_m": 100, "top_k_hit": False},
    ]
    old = {"identity": identity, "hit_radius_m": 100, "metrics": summarize(old_rows)}
    new = {"identity": identity, "hit_radius_m": 100, "metrics": summarize(new_rows)}
    result = compare_baseline(old, new)
    assert result["deltas"]["overall"]["median_error_m"] == 0
    assert result["regressions"] == ["street"]
