"""Turn self-recorded walk videos + a GPS track into blurred, geotagged gallery references.

INPUT FORMAT (one folder per walk, e.g. data/gallery_v2/self_collected/raw/walk-01/):

    walk-01/
      track.gpx            GPS track recorded during the walk (required)
      IMG_0001.MOV, ...    One or more phone videos (.mov/.mp4), camera facing forward
      offsets.json         Optional: {"IMG_0001.MOV": -2.0} seconds added to the video
                           start time if the phone clock and GPS clock disagree

- GPX: every <trkpt> needs lat/lon and <time> (UTC, ISO 8601). Horizontal accuracy in
  metres is read from <extensions> (<accuracy>, <hacc> or <horizontalAccuracy>); if
  only <hdop> exists, accuracy is estimated as 5 m x HDOP. Points with no accuracy
  information are rejected unless --allow-unknown-accuracy is given.
  Suitable apps: GPSLogger (Android), Open GPX Tracker / GPS Tracks (iOS); log at 1 Hz.
- Video start time: the container's ``creation_time`` (UTC), as written by iOS and
  Android camera apps, plus any offset from offsets.json.

PROCESSING

1. ffmpeg decodes frames at --fps into memory (no unblurred frame is ever written).
2. Each frame time is matched to the GPX track: position is linearly interpolated;
   the frame is rejected if it is > 5 s from the nearest GPX point or the
   interpolated accuracy is worse than 15 m.
3. A frame is kept only if it is >= --every-m metres from the previously kept frame.
4. Faces and licence plates are detected with EgoBlur (Apache-2.0) and blurred
   **before** the frame is written. Without both model files the script refuses to run.
5. Blur / sky-ground rules from ``eval.quality`` drop unusable frames.
6. Output goes to ``pending/``, with ``review.html`` (contact sheet). Frames enter the
   gallery only after you list approved frame IDs in ``approved.json`` (reject private
   interiors, missed faces/plates, or bad frames there).

    python -m eval.walks process data/gallery_v2/self_collected/raw/walk-01 \
        --face-model ego_blur_face.jit --plate-model ego_blur_lp.jit
    python -m eval.walks approve data/gallery_v2/self_collected/walk-01 --all-except id1,id2

Heading = direction of travel from the GPX track (assumes a forward-facing camera).
"""

from __future__ import annotations

import argparse
import base64
import bisect
import io
import json
import math
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageFilter

from eval import quality
from eval.common import signature
from eval.geo_eval import haversine_m
from eval.provenance import SELF_COLLECTED_LICENSE
from swespy.config import in_area

MAX_GPS_GAP_S = 5.0
MAX_ACCURACY_M = 15.0
HDOP_TO_METRES = 5.0
FRAME_WIDTH = 1280


@dataclass(frozen=True)
class Fix:
    t: float  # Unix seconds.
    lat: float
    lng: float
    accuracy_m: float | None


def parse_gpx(text: str) -> list[Fix]:
    root = ET.fromstring(text)
    fixes = []
    for el in root.iter():
        if not el.tag.endswith("trkpt"):
            continue
        children = {c.tag.split("}")[-1]: c for c in el.iter()}
        if "time" not in children:
            continue
        stamp = datetime.fromisoformat((children["time"].text or "").strip().replace("Z", "+00:00"))
        accuracy: float | None = None
        for key in ("accuracy", "hacc", "horizontalAccuracy"):
            value = children[key].text if key in children else None
            if value:
                accuracy = float(value)
                break
        hdop = children["hdop"].text if "hdop" in children else None
        if accuracy is None and hdop:
            accuracy = HDOP_TO_METRES * float(hdop)
        fixes.append(
            Fix(
                stamp.timestamp(),
                float(el.get("lat", "nan")),
                float(el.get("lon", "nan")),
                accuracy,
            )
        )
    return sorted(fixes, key=lambda f: f.t)


def locate(
    fixes: list[Fix], t: float, allow_unknown_accuracy: bool = False
) -> tuple[dict[str, Any] | None, str]:
    """Interpolate a position at time ``t``; returns (position, reason)."""
    times = [f.t for f in fixes]
    i = bisect.bisect_left(times, t)
    neighbours = [fixes[j] for j in (i - 1, i) if 0 <= j < len(fixes)]
    if not neighbours or min(abs(f.t - t) for f in neighbours) > MAX_GPS_GAP_S:
        return None, "gps_gap_over_5s"
    if len(neighbours) == 2 and neighbours[1].t > neighbours[0].t:
        a, b = neighbours
        w = (t - a.t) / (b.t - a.t)
    else:
        a = b = min(neighbours, key=lambda f: abs(f.t - t))
        w = 0.0
    known = [f.accuracy_m for f in (a, b) if f.accuracy_m is not None]
    accuracy: float | None = None
    if len(known) < 2:
        if not allow_unknown_accuracy:
            return None, "gps_accuracy_unknown"
    else:
        accuracy = max(known)  # Conservative.
        if accuracy > MAX_ACCURACY_M:
            return None, "gps_accuracy_over_15m"
    lat, lng = a.lat + w * (b.lat - a.lat), a.lng + w * (b.lng - a.lng)
    return {"lat": lat, "lng": lng, "accuracy_m": accuracy, "heading": course(fixes, i)}, "ok"


def course(fixes: list[Fix], i: int) -> float | None:
    """Direction of travel over roughly +/- 3 s around index ``i``."""
    lo, hi = max(0, i - 3), min(len(fixes) - 1, i + 3)
    a, b = fixes[lo], fixes[hi]
    if float(haversine_m(a.lat, a.lng, np.array([b.lat]), np.array([b.lng]))[0]) < 2:
        return None  # Standing still: heading unknown.
    y = math.sin(math.radians(b.lng - a.lng)) * math.cos(math.radians(b.lat))
    x = math.cos(math.radians(a.lat)) * math.sin(math.radians(b.lat)) - math.sin(
        math.radians(a.lat)
    ) * math.cos(math.radians(b.lat)) * math.cos(math.radians(b.lng - a.lng))
    return math.degrees(math.atan2(y, x)) % 360


class Detector(Protocol):
    def boxes(self, image: Image.Image) -> list[tuple[float, float, float, float]]: ...


class EgoBlurDetector:
    """TorchScript EgoBlur detector (face or licence plate), as in EgoBlur's demo script."""

    def __init__(self, path: Path, threshold: float, device: str = "cpu") -> None:
        import torch

        if not path.is_file():
            raise FileNotFoundError(f"EgoBlur model not found: {path}")
        self.torch, self.device, self.threshold = torch, device, threshold
        self.model = torch.jit.load(str(path), map_location=device).eval()

    def boxes(self, image: Image.Image) -> list[tuple[float, float, float, float]]:
        import torchvision

        # EgoBlur's demo feeds a BGR uint8 CHW tensor (OpenCV channel order).
        bgr = np.asarray(image.convert("RGB"))[..., ::-1].copy()
        tensor = self.torch.from_numpy(np.transpose(bgr, (2, 0, 1))).to(self.device)
        with self.torch.no_grad():
            boxes, _, scores, _ = self.model(tensor)
        keep = torchvision.ops.nms(boxes, scores, 0.3)
        return [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b, s in zip(boxes[keep], scores[keep], strict=True)
            if float(s) >= self.threshold
        ]


def blur(
    image: Image.Image, boxes: list[tuple[float, float, float, float]], scale: float = 1.15
) -> Image.Image:
    """Gaussian-blur each (slightly enlarged) box in place; returns the image."""
    for x0, y0, x1, y1 in boxes:
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * scale, (y1 - y0) * scale
        box = (
            int(round(max(0, cx - w / 2))),
            int(round(max(0, cy - h / 2))),
            int(round(min(image.width, cx + w / 2))),
            int(round(min(image.height, cy + h / 2))),
        )
        if box[2] > box[0] and box[3] > box[1]:
            region = image.crop(box)
            radius = max(8, int(max(region.size) / 4))
            image.paste(region.filter(ImageFilter.GaussianBlur(radius)), box)
    return image


def video_start(path: Path) -> datetime:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=creation_time:stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    stamp = data.get("format", {}).get("tags", {}).get("creation_time")
    if not stamp:
        raise ValueError(f"{path.name} has no creation_time; add a GPX-aligned start manually")
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def frames(path: Path, fps: float):
    """Yield (seconds from start, PIL image) decoded in memory, scaled to FRAME_WIDTH."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    rotation = abs(int(next((d.get("rotation", 0) for d in stream.get("side_data_list", [])), 0)))
    w, h = stream["width"], stream["height"]
    if rotation in (90, 270):
        w, h = h, w
    out_w = min(FRAME_WIDTH, w)
    out_h = int(round(h * out_w / w / 2) * 2)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale={out_w}:{out_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    size = out_w * out_h * 3
    k = 0
    assert proc.stdout is not None
    while chunk := proc.stdout.read(size):
        if len(chunk) < size:
            break
        yield k / fps, Image.frombytes("RGB", (out_w, out_h), chunk)
        k += 1
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name}")


def process(
    folder: Path,
    out_root: Path,
    face: Detector,
    plate: Detector,
    fps: float,
    every_m: float,
    creator: str,
    allow_unknown_accuracy: bool,
) -> dict[str, Any]:
    gpx = folder / "track.gpx"
    fixes = parse_gpx(gpx.read_text())
    if len(fixes) < 2:
        raise ValueError("track.gpx needs at least two timed track points")
    offsets = (
        json.loads((folder / "offsets.json").read_text())
        if (folder / "offsets.json").exists()
        else {}
    )
    out = out_root / folder.name
    pending = out / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    reasons: Counter[str] = Counter()
    privacy = {
        "frames_checked": 0,
        "frames_with_faces": 0,
        "frames_with_plates": 0,
        "face_boxes_blurred": 0,
        "plate_boxes_blurred": 0,
    }
    rows, last = [], None
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".mov", ".mp4"))
    for video in videos:
        start = video_start(video) + timedelta(seconds=float(offsets.get(video.name, 0)))
        for offset_s, image in frames(video, fps):
            reasons["decoded"] += 1
            moment = start + timedelta(seconds=offset_s)
            position, why = locate(fixes, moment.timestamp(), allow_unknown_accuracy)
            if position is None:
                reasons[why] += 1
                continue
            if not in_area(position["lng"], position["lat"]):
                reasons["outside_polygon"] += 1
                continue
            if (
                last is not None
                and float(
                    haversine_m(
                        last["lat"],
                        last["lng"],
                        np.array([position["lat"]]),
                        np.array([position["lng"]]),
                    )[0]
                )
                < every_m
            ):
                reasons["closer_than_every_m"] += 1
                continue
            rule = quality.image_rule(quality.measures(image))
            if rule:
                reasons[rule] += 1
                continue
            faces, plates = face.boxes(image), plate.boxes(image)
            image = blur(image, faces + plates)  # Blurred BEFORE anything is written.
            privacy["frames_checked"] += 1
            privacy["frames_with_faces"] += bool(faces)
            privacy["frames_with_plates"] += bool(plates)
            privacy["face_boxes_blurred"] += len(faces)
            privacy["plate_boxes_blurred"] += len(plates)
            ms = int(moment.timestamp() * 1000)
            frame_id = f"{folder.name}-{video.stem}-{ms}"
            path = pending / f"{frame_id}.jpg"
            image.save(path, quality=92)
            rows.append(
                {
                    "source": "self_collected",
                    "source_id": frame_id,
                    "ref_id": f"self-{frame_id}",
                    "sequence": f"self-{folder.name}-{video.stem}",
                    "creator": creator,
                    "captured_at": ms,
                    "capture_year": moment.astimezone(UTC).year,
                    "lat": position["lat"],
                    "lng": position["lng"],
                    "location_source": "gpx_interpolated",
                    "gps_accuracy_m": position["accuracy_m"],
                    "heading": position["heading"],
                    "heading_source": "gpx_course",
                    "license_url": SELF_COLLECTED_LICENSE,
                    "source_url": None,
                    "is_pano": False,
                    "blurred": True,
                    "blur_boxes": {"faces": len(faces), "plates": len(plates)},
                    "path": str(path.relative_to(out_root.parent)),
                    **signature(path),
                }
            )
            last = position
            reasons["kept_pending_review"] += 1
    report = {
        "walk": folder.name,
        "videos": [v.name for v in videos],
        "fps": fps,
        "every_m": every_m,
        "decisions": dict(reasons),
        "privacy": privacy,
        "gps_rules": {
            "max_gap_s": MAX_GPS_GAP_S,
            "max_accuracy_m": MAX_ACCURACY_M,
            "hdop_to_metres": HDOP_TO_METRES,
        },
    }
    (out / "pending.json").write_text(json.dumps({"references": rows}, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    review_sheet(out, rows)
    return report


def review_sheet(out: Path, rows: list[dict[str, Any]], sample: int = 60) -> None:
    """Contact sheet for privacy/interior spot-checks (random sample, seed 0)."""
    chosen = random.Random(0).sample(rows, min(sample, len(rows)))
    cards = []
    for r in chosen:
        with Image.open(out.parent.parent / r["path"]) as im:
            im.thumbnail((320, 240))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=75)
        cards.append(
            f"<figure><img src='data:image/jpeg;base64,"
            f"{base64.b64encode(buf.getvalue()).decode()}'><figcaption>{r['source_id']}"
            f" · faces {r['blur_boxes']['faces']} · plates {r['blur_boxes']['plates']}"
            "</figcaption></figure>"
        )
    (out / "review.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Walk review</title><style>body{font:14px "
        "system-ui}figure{display:inline-block;margin:6px}</style><h1>Spot-check: missed "
        f"faces/plates, private interiors ({len(chosen)} of {len(rows)} frames)</h1>"
        + "".join(cards)
    )


def approve(out: Path, exclude: set[str]) -> None:
    rows = json.loads((out / "pending.json").read_text())["references"]
    unknown = exclude - {r["source_id"] for r in rows}
    if unknown:
        raise ValueError(f"Unknown frame IDs: {sorted(unknown)[:5]}")
    approved = [r for r in rows if r["source_id"] not in exclude]
    (out / "approved.json").write_text(
        json.dumps(
            {
                "approved_at": datetime.now(UTC).isoformat(),
                "excluded_on_review": sorted(exclude),
                "references": approved,
            },
            indent=1,
        )
    )
    print(f"Approved {len(approved)} frames; excluded {len(exclude)} on review")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("process")
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/gallery_v2/self_collected"))
    p.add_argument("--face-model", type=Path, required=True)
    p.add_argument("--plate-model", type=Path, required=True)
    p.add_argument("--face-threshold", type=float, default=0.5)
    p.add_argument("--plate-threshold", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--every-m", type=float, default=5.0)
    p.add_argument("--creator", default="heimdall-self-collected")
    p.add_argument("--allow-unknown-accuracy", action="store_true")
    a = sub.add_parser("approve")
    a.add_argument("walk_output", type=Path)
    a.add_argument("--all-except", default="", help="Comma-separated frame IDs to reject")
    args = parser.parse_args()
    if args.action == "process":
        face = EgoBlurDetector(args.face_model, args.face_threshold)
        plate = EgoBlurDetector(args.plate_model, args.plate_threshold)
        print(
            json.dumps(
                process(
                    args.folder,
                    args.out,
                    face,
                    plate,
                    args.fps,
                    args.every_m,
                    args.creator,
                    args.allow_unknown_accuracy,
                ),
                indent=1,
            )
        )
    else:
        approve(args.walk_output, {x for x in args.all_except.split(",") if x})


if __name__ == "__main__":
    main()
