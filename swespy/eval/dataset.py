"""Validate user-owned evaluation photos without importing the model."""

from __future__ import annotations

import csv
import hashlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from PIL import Image, UnidentifiedImageError

VIEWPOINTS = ("street", "window", "elevated", "indoor_view")
FIELDS = ["filename", "lat", "lng", "viewpoint_type"]


class DatasetError(ValueError):
    """The private evaluation dataset is missing or invalid."""


@dataclass(frozen=True)
class Photo:
    """One validated private query and its camera-location label."""

    filename: str
    path: Path
    lat: float
    lng: float
    viewpoint_type: str
    sha256: str


def load_dataset(labels: Path, photos: Path, reference_dir: Path | None = None) -> list[Photo]:
    """Load labelled JPEG/PNG queries, rejecting invalid paths and reference copies.

    Args:
        labels: Private CSV with the four required columns.
        photos: Directory containing user-owned query photos.
        reference_dir: Optional local reference folder to check for identical files.

    Returns:
        Validated, nonempty query list in CSV order.

    Raises:
        DatasetError: Labels, coordinates, paths or images are invalid.
    """
    if not labels.is_file():
        raise DatasetError(
            f"Missing labels: {labels}. Copy eval/labels.example.csv to eval/labels.csv "
            "and supply your own photos and camera coordinates; no baseline was generated."
        )
    root = photos.resolve()
    samples: list[Photo] = []
    names: set[str] = set()
    hashes: set[str] = set()
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise DatasetError(
                f"Labels must have exactly these columns in order: {','.join(FIELDS)}"
            )
        for line, row in enumerate(reader, start=2):
            name = (row.get("filename") or "").strip()
            relative = Path(name)
            if (
                not name
                or relative.is_absolute()
                or PureWindowsPath(name).drive
                or ".." in PureWindowsPath(name).parts
                or "\\" in name
                or None in row
            ):
                raise DatasetError(f"Invalid relative filename or CSV row at line {line}")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise DatasetError(f"Missing photo or path outside photos directory at line {line}")
            if name.casefold() in names:
                raise DatasetError(f"Duplicate filename at line {line}")
            names.add(name.casefold())
            try:
                lat, lng = float(row["lat"]), float(row["lng"])
            except (ValueError, TypeError) as exc:
                raise DatasetError(f"Invalid camera coordinates at line {line}") from exc
            if not (
                math.isfinite(lat)
                and math.isfinite(lng)
                and -90 <= lat <= 90
                and -180 <= lng <= 180
            ):
                raise DatasetError(f"Coordinates outside valid ranges at line {line}")
            viewpoint = row["viewpoint_type"]
            if viewpoint not in VIEWPOINTS:
                raise DatasetError(f"Invalid viewpoint_type at line {line}; expected {VIEWPOINTS}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.format not in {"JPEG", "PNG"}:
                            raise DatasetError(f"Photo at line {line} must be a JPEG or PNG")
                        image.verify()
            except (
                UnidentifiedImageError,
                OSError,
                SyntaxError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as exc:
                raise DatasetError(f"Corrupt or unsafe image at line {line}") from exc
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                raise DatasetError(f"Duplicate photo contents at line {line}")
            hashes.add(digest)
            samples.append(Photo(name, path, lat, lng, viewpoint, digest))
    if not samples:
        raise DatasetError("Labels contain no photos. Supply user-owned queries before evaluation.")
    if reference_dir and reference_dir.exists():
        for reference in reference_dir.rglob("*"):
            if reference.is_file() and hashlib.sha256(reference.read_bytes()).hexdigest() in hashes:
                raise DatasetError(
                    "An evaluation photo duplicates local reference data; remove it."
                )
    return samples
