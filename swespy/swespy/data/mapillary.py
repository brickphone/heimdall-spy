"""Mapillary API client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from swespy.config import Settings


class MapillaryClient:
    """Async client for Mapillary Graph API v4."""

    base_url = "https://graph.mapillary.com"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self) -> "MapillaryClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    async def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a GET request with retry and return parsed JSON."""
        response = await self._client.get(
            f"{self.base_url}{path}",
            params=params,
            headers={"Authorization": f"OAuth {self.settings.mapillary_access_token}"},
        )
        response.raise_for_status()
        return response.json()

    async def search_images(self, bbox: tuple[float, float, float, float], limit: int = 2000) -> list[dict]:
        """Search image metadata in a bounding box.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
            limit: Maximum number of records to return.

        Returns:
            Image metadata list from Mapillary.
        """
        logger.info("Searching Mapillary images for bbox={} limit={}", bbox, limit)
        fields = "id,geometry,captured_at,thumb_2048_url"
        results: list[dict] = []
        after: str | None = None

        while len(results) < limit:
            params: dict[str, Any] = {
                "bbox": ",".join(str(v) for v in bbox),
                "fields": fields,
                "limit": min(1000, limit - len(results)),
            }
            if after:
                params["after"] = after

            payload = await self._request("/images", params)
            page = payload.get("data", [])
            if not page:
                break

            results.extend(page)
            paging = payload.get("paging", {})
            cursors = paging.get("cursors", {})
            after = cursors.get("after")
            logger.info("Fetched {} images so far", len(results))
            if not after:
                break

        return results[:limit]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    )
    async def download_image(self, url: str, dest_path: Path) -> Path:
        """Download one image to disk if missing."""
        if dest_path.exists():
            return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        response = await self._client.get(url)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        logger.info("Downloaded image {}", dest_path.name)
        return dest_path

    async def batch_download(
        self,
        image_records: list[dict],
        dest_dir: Path,
        concurrency: int = 20,
    ) -> None:
        """Download image records concurrently with bounded parallelism."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _worker(record: dict) -> None:
            thumb_url = record.get("thumb_2048_url")
            image_id = record.get("id")
            if not thumb_url or not image_id:
                return
            async with semaphore:
                await self.download_image(thumb_url, dest_dir / f"{image_id}.jpg")

        logger.info("Starting batch download for {} images", len(image_records))
        await asyncio.gather(*[_worker(record) for record in image_records])
        logger.info("Finished image download batch")
