"""Data collection and preparation pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
from loguru import logger

from swespy.config import Settings
from swespy.data.mapillary import MapillaryClient
from swespy.data.s2_grid import assign_cells_to_dataframe


class DataPipeline:
    """End-to-end data pipeline from Mapillary to local parquet metadata."""

    def __init__(self, settings: Settings, project_root: Path | None = None) -> None:
        self.settings = settings
        self.project_root = project_root or Path.cwd()
        self.raw_dir = self.project_root / "data" / "raw"
        self.meta_dir = self.project_root / "data" / "metadata"

    def _subdivide_bbox(self, bbox: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
        min_lon, min_lat, max_lon, max_lat = bbox
        divisions_lon = self.settings.bbox_grid_lon_divisions
        divisions_lat = self.settings.bbox_grid_lat_divisions
        step_lon = (max_lon - min_lon) / divisions_lon
        step_lat = (max_lat - min_lat) / divisions_lat
        sub_boxes: list[tuple[float, float, float, float]] = []

        for i in range(divisions_lon):
            for j in range(divisions_lat):
                sub_min_lon = min_lon + i * step_lon
                sub_max_lon = min_lon + (i + 1) * step_lon
                sub_min_lat = min_lat + j * step_lat
                sub_max_lat = min_lat + (j + 1) * step_lat
                sub_boxes.append((sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat))
        return sub_boxes

    @staticmethod
    def _extract_rows(records: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for record in records:
            geometry = record.get("geometry", {})
            coords = geometry.get("coordinates", [None, None])
            lng, lat = coords[0], coords[1]
            image_id = record.get("id")
            if lat is None or lng is None or image_id is None:
                continue
            rows.append(
                {
                    "image_id": str(image_id),
                    "lat": float(lat),
                    "lng": float(lng),
                    "thumb_url": record.get("thumb_2048_url"),
                }
            )
        return rows

    async def _run_async(self) -> pd.DataFrame:
        sub_boxes = self._subdivide_bbox(self.settings.sweden_bbox)
        logger.info("Running pipeline across {} sub-bounding boxes", len(sub_boxes))
        all_rows: list[dict] = []

        async with MapillaryClient(self.settings) as mapillary:
            for bbox in sub_boxes:
                records = await mapillary.search_images(bbox=bbox, limit=self.settings.mapillary_query_limit)
                all_rows.extend(self._extract_rows(records))

            df = pd.DataFrame(all_rows).drop_duplicates(subset=["image_id"])
            df = assign_cells_to_dataframe(df, level=self.settings.s2_cell_level)
            capped = (
                df.groupby("s2_cell", group_keys=False)
                .apply(
                    lambda group: group.sample(
                        n=min(len(group), self.settings.max_images_per_cell),
                        random_state=42,
                    )
                )
                .reset_index(drop=True)
            )

            await mapillary.batch_download(
                capped.to_dict("records"),
                self.raw_dir,
                concurrency=self.settings.mapillary_download_concurrency,
            )

        capped["local_path"] = capped["image_id"].apply(lambda image_id: str(self.raw_dir / f"{image_id}.jpg"))
        return capped

    def _clean_small_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where file size is below 100KB and delete those files."""
        min_size_bytes = self.settings.min_clean_file_size_kb * 1024

        def keep_row(path_str: str) -> bool:
            path = Path(path_str)
            if not path.exists():
                return False
            size = path.stat().st_size
            if size < min_size_bytes:
                path.unlink(missing_ok=True)
                return False
            return True

        cleaned = df[df["local_path"].apply(keep_row)].copy()
        logger.info("Cleaned dataset size: {}", len(cleaned))
        return cleaned

    def run(self) -> pd.DataFrame:
        """Execute full pipeline and persist metadata parquet."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        df = asyncio.run(self._run_async())
        cleaned = self._clean_small_files(df)
        output = cleaned[["image_id", "lat", "lng", "s2_cell", "local_path"]]
        out_path = self.meta_dir / "images.parquet"
        output.to_parquet(out_path, index=False)
        logger.info("Saved metadata to {}", out_path)
        return output
