"""Data collection and preparation pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

import httpx
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

        width = max_lon - min_lon
        height = max_lat - min_lat
        total_area = width * height
        tile_area = total_area / max(1, divisions_lon * divisions_lat)
        max_tile_area = self.settings.mapillary_max_bbox_area_sqdeg

        if tile_area > max_tile_area:
            scale = (tile_area / max_tile_area) ** 0.5
            divisions_lon = max(divisions_lon, int(divisions_lon * scale) + 1)
            divisions_lat = max(divisions_lat, int(divisions_lat * scale) + 1)
            logger.warning(
                "Increasing grid to {}x{} so each bbox area <= {:.3f} sqdeg (Mapillary limit).",
                divisions_lon,
                divisions_lat,
                max_tile_area,
            )

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
        logger.info(
            "Generated {} sub-bboxes ({}x{}), approx tile area={:.4f} sqdeg",
            len(sub_boxes),
            divisions_lon,
            divisions_lat,
            step_lon * step_lat,
        )
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

    @staticmethod
    def _split_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
        """Split one bbox into 4 smaller quadrants."""
        min_lon, min_lat, max_lon, max_lat = bbox
        mid_lon = (min_lon + max_lon) / 2.0
        mid_lat = (min_lat + max_lat) / 2.0
        return [
            (min_lon, min_lat, mid_lon, mid_lat),
            (mid_lon, min_lat, max_lon, mid_lat),
            (min_lon, mid_lat, mid_lon, max_lat),
            (mid_lon, mid_lat, max_lon, max_lat),
        ]

    async def _search_bbox_with_fallback(
        self,
        mapillary: MapillaryClient,
        bbox: tuple[float, float, float, float],
        limit: int,
        depth: int = 0,
        max_depth: int = 2,
    ) -> list[dict]:
        """Search a bbox and recursively split on repeated API failures."""
        try:
            return await mapillary.search_images(bbox=bbox, limit=limit)
        except httpx.HTTPError as exc:
            if depth >= max_depth:
                logger.error("Skipping bbox {} after fallback attempts: {}", bbox, exc)
                return []
            logger.warning("Splitting failing bbox {} at depth {} due to: {}", bbox, depth, exc)
            child_bboxes = self._split_bbox(bbox)
            merged: list[dict] = []
            per_child_limit = max(1, limit // len(child_bboxes))
            for child_bbox in child_bboxes:
                merged.extend(
                    await self._search_bbox_with_fallback(
                        mapillary=mapillary,
                        bbox=child_bbox,
                        limit=per_child_limit,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )
            return merged

    async def _run_async(self) -> pd.DataFrame:
        sub_boxes = self._subdivide_bbox(self.settings.sweden_bbox)
        logger.info("Running pipeline across {} sub-bounding boxes", len(sub_boxes))
        all_rows: list[dict] = []
        started = time.perf_counter()
        total_boxes = len(sub_boxes)

        async with MapillaryClient(self.settings) as mapillary:
            for idx, bbox in enumerate(sub_boxes, start=1):
                logger.info("Processing bbox {}/{}: {}", idx, len(sub_boxes), bbox)
                records = await self._search_bbox_with_fallback(
                    mapillary=mapillary,
                    bbox=bbox,
                    limit=self.settings.mapillary_query_limit,
                )
                logger.info("bbox {}/{} returned {} records", idx, len(sub_boxes), len(records))
                all_rows.extend(self._extract_rows(records))
                logger.info("Cumulative raw rows so far: {}", len(all_rows))
                elapsed = time.perf_counter() - started
                rate = idx / elapsed if elapsed > 0 else 0.0
                remaining = total_boxes - idx
                eta_seconds = remaining / rate if rate > 0 else 0.0
                logger.info(
                    "Pipeline progress {}/{} ({:.1f}%) rate={:.2f} bbox/s eta={:.1f} min",
                    idx,
                    total_boxes,
                    (idx / max(1, total_boxes)) * 100.0,
                    rate,
                    eta_seconds / 60.0,
                )

            if not all_rows:
                logger.warning("No rows fetched from Mapillary.")
                return pd.DataFrame(columns=["image_id", "lat", "lng", "s2_cell", "local_path"])

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

            if not capped.empty:
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
        if df.empty:
            out_path = self.meta_dir / "images.parquet"
            df.to_parquet(out_path, index=False)
            logger.warning("Saved empty metadata to {}", out_path)
            return df

        cleaned = self._clean_small_files(df)
        output = cleaned[["image_id", "lat", "lng", "s2_cell", "local_path"]]
        out_path = self.meta_dir / "images.parquet"
        output.to_parquet(out_path, index=False)
        logger.info("Saved metadata to {}", out_path)
        return output
