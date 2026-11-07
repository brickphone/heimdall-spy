from eval import provenance

VALID = {"source": "mapillary", "source_id": "1", "license_url":
         "https://creativecommons.org/licenses/by-sa/4.0/", "captured_at": 1, "lat": 58.4,
         "lng": 15.6, "heading": None, "capture_year": 1970,
         "location_source": "mapillary_computed"}


def test_valid_row_and_missing_fields():
    assert provenance.problems(VALID) == []
    row = {k: v for k, v in VALID.items() if k not in ("captured_at", "heading")}
    assert set(provenance.problems(row)) == {"missing:captured_at", "missing:heading"}


def test_non_commercial_and_no_derivatives_licences_rejected():
    for url in ("https://creativecommons.org/licenses/by-nc/4.0/",
                "https://creativecommons.org/licenses/by-nd/4.0/",
                "https://creativecommons.org/licenses/by-nc-sa/2.0/"):
        assert "license_not_commercial" in provenance.problems({**VALID, "license_url": url})
    assert provenance.license_allowed("http://creativecommons.org/publicdomain/zero/1.0/")


def test_bad_heading_and_source():
    assert "bad_heading" in provenance.problems({**VALID, "heading": 360.0})
    assert "unknown_source" in provenance.problems({**VALID, "source": "flickr"})


def test_location_source_required():
    assert "unknown_location_source" in provenance.problems({**VALID, "location_source": "x"})
