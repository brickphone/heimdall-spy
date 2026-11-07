"""Self-collected walk ingestion: GPS rules, blurring before storage, review gate."""

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from eval import walks

T0 = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)


def gpx(points):
    pts = "".join(
        f"<trkpt lat='{lat}' lon='{lng}'><time>{(T0 + timedelta(seconds=t)).isoformat()}</time>"
        + (f"<extensions><accuracy>{acc}</accuracy></extensions>" if acc is not None else "")
        + "</trkpt>" for t, lat, lng, acc in points)
    return ("<?xml version='1.0'?><gpx xmlns='http://www.topografix.com/GPX/1/1'><trk><trkseg>"
            f"{pts}</trkseg></trk></gpx>")


def test_gpx_interpolation_gap_and_accuracy_rules():
    fixes = walks.parse_gpx(gpx([(0, 58.41, 15.62, 5), (2, 58.4102, 15.62, 5),
                                 (20, 58.4110, 15.62, 30)]))
    pos, why = walks.locate(fixes, T0.timestamp() + 1)
    assert why == "ok" and pos["lat"] == pytest.approx(58.4101)
    assert pos["heading"] == pytest.approx(0, abs=1)  # Walking north.
    assert walks.locate(fixes, T0.timestamp() + 11)[1] == "gps_gap_over_5s"
    assert walks.locate(fixes, T0.timestamp() + 19)[1] == "gps_accuracy_over_15m"
    hdop_only = walks.parse_gpx(gpx([(0, 58.41, 15.62, None)]).replace(
        "</time>", "</time><hdop>4</hdop>"))
    assert hdop_only[0].accuracy_m == 20.0
    unknown = walks.parse_gpx(gpx([(0, 58.41, 15.62, None), (1, 58.41, 15.62, None)]))
    assert walks.locate(unknown, T0.timestamp())[1] == "gps_accuracy_unknown"


def test_blur_changes_only_the_box():
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, (200, 200, 3), dtype=np.uint8))
    before = np.asarray(image).copy()
    after = np.asarray(walks.blur(image, [(50, 50, 100, 100)], scale=1.0))
    assert after[60:90, 60:90].std() < before[60:90, 60:90].std() / 2
    assert np.array_equal(after[150:, 150:], before[150:, 150:])


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def boxes(self, image):
        self.calls += 1
        return [(10.0, 10.0, 60.0, 60.0)]  # A "face" in every frame.


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_process_blurs_every_stored_frame_and_needs_approval(tmp_path):
    walk = tmp_path / "raw" / "walk-01"
    walk.mkdir(parents=True)
    video = walk / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=10:duration=6", "-metadata",
                    f"creation_time={T0.strftime('%Y-%m-%dT%H:%M:%S.000000Z')}",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    # Walk north 2 m/s for 6 s inside Linköping; good accuracy.
    (walk / "track.gpx").write_text(gpx([(t, 58.41 + t * 2 / 111_320, 15.62, 4)
                                         for t in range(0, 8)]))
    face, plate = FakeDetector(), FakeDetector()
    report = walks.process(walk, tmp_path / "out", face, plate, fps=2, every_m=3,
                           creator="me", allow_unknown_accuracy=False)
    rows = json.loads((tmp_path / "out" / "walk-01" / "pending.json").read_text())["references"]
    assert rows and report["privacy"]["frames_checked"] == len(rows) == face.calls
    assert all(r["blurred"] and r["location_source"] == "gpx_interpolated" for r in rows)
    assert report["privacy"]["face_boxes_blurred"] == len(rows)
    assert not (tmp_path / "out" / "walk-01" / "approved.json").exists()
    walks.approve(tmp_path / "out" / "walk-01", {rows[0]["source_id"]})
    approved = json.loads((tmp_path / "out" / "walk-01" / "approved.json").read_text())
    assert len(approved["references"]) == len(rows) - 1
    with Image.open(tmp_path / rows[0]["path"]) as stored:
        gray = np.asarray(stored.convert("L"), dtype=float)

    def sharpness(a):
        return np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()

    # The detected box is blurred in the stored file; an untouched region is not.
    assert sharpness(gray[15:55, 15:55]) < sharpness(gray[150:190, 150:190]) / 3
