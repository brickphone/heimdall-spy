import io
import os

from httpx import ASGITransport, AsyncClient
from PIL import Image


def _make_test_png() -> bytes:
    img = Image.new("RGB", (20, 20), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def test_predict_endpoint_schema() -> None:
    os.environ["TESTING"] = "true"

    from swespy.api.main import create_app
    from swespy.config import get_settings

    get_settings.cache_clear()

    app = create_app()
    transport = ASGITransport(app=app)
    image_bytes = _make_test_png()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.png", image_bytes, "image/png")}
        response = await client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "predicted_lat",
        "predicted_lng",
        "confidence",
        "top_matches",
        "processing_time_ms",
    }
    assert isinstance(data["top_matches"], list)
    assert len(data["top_matches"]) > 0
    first = data["top_matches"][0]
    assert {"lat", "lng", "similarity_score", "s2_cell"}.issubset(first.keys())
