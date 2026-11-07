import pytest

from eval import geo_eval


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [(0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0),
     (81, 263, 0.2553, 0.3662), (1, 20, 0.0089, 0.2361)],
)
def test_wilson_interval_known_values(k, n, low, high):
    lo, hi = geo_eval.wilson_interval(k, n)
    assert lo == pytest.approx(low, abs=1e-4)
    assert hi == pytest.approx(high, abs=1e-4)


def test_wilson_rejects_invalid_counts():
    with pytest.raises(ValueError):
        geo_eval.wilson_interval(1, 0)
    with pytest.raises(ValueError):
        geo_eval.wilson_interval(3, 2)


@pytest.mark.parametrize(
    ("nearest", "ranked", "expected"),
    [
        (150.0, [20.0, 30.0], "no_coverage"),  # Nothing within 100 m in the gallery at all.
        (10.0, [40.0, 900.0], "success"),
        (10.0, [900.0] * 10 + [60.0], "ranking"),
        (10.0, [900.0] * 49 + [99.0], "ranking"),  # Rank 50 still counts.
        (10.0, [900.0] * 50 + [5.0], "retrieval"),  # Rank 51 is outside the depth.
        (100.0, [101.0, 3000.0], "retrieval"),
    ],
)
def test_failure_type_synthetic_cases(nearest, ranked, expected):
    assert geo_eval.failure_type(nearest, ranked) == expected


def test_diagnose_reports_top_k_membership():
    truth = (58.41, 15.62)
    far = (58.50, 15.62)  # About 10 km north.
    near = (58.4105, 15.62)  # About 56 m north.
    row = geo_eval.diagnose(*truth, predicted=far, ranked_coords=[far] * 7 + [near],
                            nearest_reference_m=56.0)
    assert row["failure_type"] == "ranking"
    assert row["rank_of_first_within_m"] == {"100": 8, "500": 8}
    assert row["true_in_top_k"]["100m"] == {"1": False, "5": False, "20": True, "50": True}
    assert row["error_m"] == pytest.approx(10_008, rel=0.01)


def test_baselines_and_distribution_flags():
    refs = [{"lat": 58.41, "lng": 15.62, "easting": 500_100, "northing": 6_000_100}] * 3 + [
        {"lat": 58.50, "lng": 15.70, "easting": 509_000, "northing": 6_010_000}]
    queries = [{"lat": 58.41, "lng": 15.62, "creator": "a", "image_id": str(i)}
               for i in range(3)] + [{"lat": 58.30, "lng": 15.50, "creator": "b",
                                      "image_id": "x"}]
    majority = geo_eval.majority_location(refs)
    assert (majority["lat"], majority["reference_count"]) == (58.41, 3)
    constant = geo_eval.constant_baseline(queries, majority["lat"], majority["lng"])
    assert constant["within"]["25m"]["count"] == 3
    random = geo_eval.random_reference_baseline(queries, refs)
    assert random["within"]["25m"]["count"] == pytest.approx(3 * 0.75)
    dist = geo_eval.distribution(queries)
    assert dist["photographer_count"] == 2
    assert dist["clustered"] and len(dist["flags"]) == 3


def test_photographer_balanced_averages_per_creator():
    result = geo_eval.photographer_balanced(["a"] * 9 + ["b"], [True] * 9 + [False])
    assert result["rate"] == pytest.approx(0.5)
    assert result["photographer_count"] == 2
    assert 0 <= result["ci95"][0] <= result["ci95"][1] <= 1


def test_target_requires_rate_and_non_overlapping_cis():
    def within(k, n):
        return {"within": {"500m": geo_eval.rate(k, n)}}

    assert geo_eval.target_check(within(40, 60), within(10, 60))["met"]
    overlapping = geo_eval.target_check(within(28, 66), within(15, 66))
    assert overlapping["a_rate_at_least_40pct"] and not overlapping["b_ci_above_majority"]
    assert not geo_eval.target_check(within(20, 66), within(0, 66))["met"]


def test_project_area_polygon():
    from swespy.config import in_area

    assert in_area(15.6214, 58.4108)  # Linköping centre.
    assert in_area(15.51, 58.41)  # Malmslätt.
    assert not in_area(15.74, 58.47)  # North-east corner of the old rectangle.
    assert not in_area(15.62, 58.30)
