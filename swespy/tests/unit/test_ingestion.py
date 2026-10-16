"""Verify recoverable ingestion failures without contacting Mapillary."""

import io
from pathlib import Path

import httpx
import pandas as pd
import pytest
from PIL import Image

from swespy.config import Settings
from swespy.data.mapillary import MapillaryClient, is_retryable, valid_image
from swespy.data.pipeline import DataPipeline


def png() -> bytes:
    """Return synthetic reference bytes for offline tests."""
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(stream, format="PNG")
    return stream.getvalue()


async def test_download_resumes_valid_files_and_replaces_partial_files(tmp_path: Path) -> None:
    (tmp_path / "1.jpg").write_bytes(png())
    (tmp_path / "2.jpg").write_bytes(b"interrupted download")
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, content=png())

    async with MapillaryClient(Settings()) as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        await client.batch_download(
            [{"id": str(i), "thumb_url": f"https://reference.test/{i}"} for i in (1, 2, 3)],
            tmp_path,
            concurrency=2,
        )
    assert sorted(calls) == ["/2", "/3"]
    assert all(valid_image(tmp_path / f"{i}.jpg") for i in (1, 2, 3))
    assert not list(tmp_path.glob("*.part"))


async def test_corrupt_response_fails_without_publishing_partial_file(tmp_path: Path) -> None:
    async with MapillaryClient(Settings()) as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"invalid image")
            )
        )
        with pytest.raises(RuntimeError, match="metadata was not published"):
            await client.batch_download(
                [{"id": "1", "thumb_url": "https://reference.test/1"}], tmp_path
            )
    assert not (tmp_path / "1.jpg").exists()


async def test_authentication_error_is_not_retried_or_split(tmp_path: Path) -> None:
    calls = 0

    def unauthorized(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    settings = Settings(mapillary_access_token="offline-test")
    async with MapillaryClient(settings) as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(unauthorized))
        with pytest.raises(httpx.HTTPStatusError):
            await DataPipeline(settings, tmp_path)._search_bbox_with_fallback(
                client, settings.sweden_bbox, 1
            )
    assert calls == 1
    request = httpx.Request("GET", "https://reference.test")
    transient = httpx.HTTPStatusError(
        "unavailable", request=request, response=httpx.Response(503, request=request)
    )
    assert is_retryable(transient)


def test_empty_ingestion_preserves_previous_metadata(tmp_path: Path, monkeypatch) -> None:
    pipeline = DataPipeline(Settings(mapillary_access_token="offline-test"), tmp_path)
    pipeline.meta_dir.mkdir(parents=True)
    previous = pipeline.meta_dir / "images.parquet"
    previous.write_bytes(b"previous metadata")

    async def empty() -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr(pipeline, "_run_async", empty)
    with pytest.raises(RuntimeError, match="existing metadata preserved"):
        pipeline.run()
    assert previous.read_bytes() == b"previous metadata"


def test_nationwide_bbox_rejected_before_network(tmp_path: Path) -> None:
    settings = Settings(mapillary_access_token="offline-test", sweden_bbox=(10.5, 55, 24.2, 69.1))
    with pytest.raises(ValueError, match="Linkoping pilot"):
        DataPipeline(settings, tmp_path).run()


def test_metadata_geometry_validation() -> None:
    rows = DataPipeline._extract_rows(
        [
            {"id": "1", "geometry": None},
            {"id": "2", "geometry": {"coordinates": []}},
            {"id": "3", "geometry": {"coordinates": [15.62, float("nan")]}},
            {"id": "4", "geometry": {"coordinates": [15.62, 58.41]}},
        ]
    )
    assert len(rows) == 1
    assert rows[0]["image_id"] == "4"
