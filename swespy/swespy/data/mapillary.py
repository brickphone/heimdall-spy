"""Mapillary API client."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from swespy.config import Settings


def log_before_sleep(retry_state: RetryCallState) -> None:
    """Log retry attempt details before tenacity sleeps."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying Mapillary request attempt={} next_sleep={:.2f}s error={}",
        retry_state.attempt_number,
        retry_state.next_action.sleep if retry_state.next_action else 0.0,
        exc,
    )


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
        before_sleep=log_before_sleep,
        reraise=True,
    )
    async def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a GET request with retry and return parsed JSON."""
        logger.debug("Mapillary request path={} params={}", path, params)
        response = await self._client.get(
            f"{self.base_url}{path}",
            params=params,
            headers={"Authorization": f"OAuth {self.settings.mapillary_access_token}"},
        )
        logger.debug(
            "Mapillary response status={} path={} query_limit={} has_after={}",
            response.status_code,
            path,
            params.get("limit"),
            "after" in params,
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
        page_size = min(self.settings.mapillary_page_size, limit)
        page_index = 0

        while len(results) < limit:
            page_index += 1
            params: dict[str, Any] = {
                "bbox": ",".join(str(v) for v in bbox),
                "fields": fields,
                "limit": min(page_size, limit - len(results)),
            }
            if after:
                params["after"] = after

            logger.info(
                "Requesting bbox={} page={} page_size={} fetched={} remaining={}",
                bbox,
                page_index,
                params["limit"],
                len(results),
                limit - len(results),
            )
            started = time.perf_counter()
            try:
                payload = await self._request("/images", params)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in (429, 500, 502, 503, 504) and page_size > self.settings.mapillary_min_page_size:
                    page_size = max(self.settings.mapillary_min_page_size, page_size // 2)
                    logger.warning(
                        "Mapillary {} for bbox {}. Reducing page_size to {} and retrying.",
                        status_code,
                        bbox,
                        page_size,
                    )
                    continue
                raise

            page = payload.get("data", [])
            elapsed_s = time.perf_counter() - started
            if not page:
                logger.info("No more data for bbox={} page={} duration={:.2f}s", bbox, page_index, elapsed_s)
                break

            results.extend(page)
            paging = payload.get("paging", {})
            cursors = paging.get("cursors", {})
            after = cursors.get("after")
            logger.info(
                "Fetched {} images so far for bbox={} page={} page_items={} duration={:.2f}s next_cursor_present={}",
                len(results),
                bbox,
                page_index,
                len(page),
                elapsed_s,
                bool(after),
            )
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
        total = len(image_records)
        completed = 0
        skipped_existing = 0
        failed = 0
        lock = asyncio.Lock()
        log_every = max(1, total // 20)
        started = time.perf_counter()

        async def _worker(record: dict) -> None:
            nonlocal completed, failed
            thumb_url = record.get("thumb_2048_url") or record.get("thumb_url")
            image_id = record.get("id") or record.get("image_id")
            if not thumb_url or not image_id:
                async with lock:
                    failed += 1
                    done = completed + failed
                    if done % log_every == 0 or done == total:
                        elapsed = time.perf_counter() - started
                        logger.info(
                                "Download progress {}/{} ({:.1f}%) failed={} elapsed={:.1f}s",
                                done,
                                total,
                                (done / max(1, total)) * 100.0,
                                failed,
                                elapsed,
                            )
                return
            dest_path = dest_dir / f"{image_id}.jpg"
            if dest_path.exists():
                async with lock:
                    skipped_existing += 1
                    done = completed + failed + skipped_existing
                    if done % log_every == 0 or done == total:
                        elapsed = time.perf_counter() - started
                        logger.info(
                            "Download progress {}/{} ({:.1f}%) downloaded={} skipped_existing={} failed={} elapsed={:.1f}s",
                            done,
                            total,
                            (done / max(1, total)) * 100.0,
                            completed,
                            skipped_existing,
                            failed,
                            elapsed,
                        )
                return
            async with semaphore:
                try:
                    await self.download_image(thumb_url, dest_path)
                    async with lock:
                        completed += 1
                except httpx.HTTPError:
                    async with lock:
                        failed += 1
                finally:
                    async with lock:
                        done = completed + failed + skipped_existing
                        if done % log_every == 0 or done == total:
                            elapsed = time.perf_counter() - started
                            logger.info(
                                "Download progress {}/{} ({:.1f}%) downloaded={} skipped_existing={} failed={} elapsed={:.1f}s",
                                done,
                                total,
                                (done / max(1, total)) * 100.0,
                                completed,
                                skipped_existing,
                                failed,
                                elapsed,
                            )

        logger.info("Starting batch download for {} images", len(image_records))
        await asyncio.gather(*[_worker(record) for record in image_records])
        elapsed = time.perf_counter() - started
        logger.info(
            "Finished image download batch downloaded={} skipped_existing={} failed={} total={} elapsed={:.1f}s",
            completed,
            skipped_existing,
            failed,
            total,
            elapsed,
        )
