"""Application configuration."""

from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# Approved project area (2026-11-07): Linköping urban area incl. Malmslätt, as (lng, lat)
# vertices. All reference acquisition and gallery membership is restricted to it.
LINKOPING_AREA: tuple[tuple[float, float], ...] = (
    (15.495, 58.392), (15.495, 58.428), (15.535, 58.470), (15.595, 58.472),
    (15.640, 58.452), (15.700, 58.440), (15.745, 58.412), (15.758, 58.372),
    (15.722, 58.350), (15.640, 58.352), (15.555, 58.352), (15.515, 58.372),
)


def in_area(lng: float, lat: float, polygon: tuple[tuple[float, float], ...] = LINKOPING_AREA
            ) -> bool:
    """Even-odd ray-casting test; points exactly on an edge may fall either way."""
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y1 > lat) != (y2 > lat) and lng < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    mapillary_access_token: str = ""
    # MegaLoc assets/descriptor cache and the exported gallery index used for inference.
    megaloc_root: str = "data/megaloc"
    megaloc_device: str = "auto"
    gallery_index: str = (
        "data/benchmarks/runs/linkoping_city-20261106T130343.492589Z/gallery-index.json"
    )
    retrieval_depth: int = 50
    s2_cell_level: int = 14
    # Central Linkoping pilot, not the full city or municipal boundary.
    sweden_bbox: tuple[float, float, float, float] = (15.58, 58.39, 15.66, 58.43)
    area_polygon: tuple[tuple[float, float], ...] = LINKOPING_AREA
    reference_image_limit: int = 500
    batch_size: int = 64
    max_images_per_cell: int = 50
    top_k_results: int = 5
    mapillary_query_limit: int = 2000
    mapillary_page_size: int = 1000
    mapillary_min_page_size: int = 100
    mapillary_max_bbox_area_sqdeg: float = 0.01
    mapillary_download_concurrency: int = 20
    bbox_grid_lon_divisions: int = 6
    bbox_grid_lat_divisions: int = 6
    min_clean_file_size_kb: int = 100
    max_upload_size_mb: int = 10
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png")
    api_rate_limit_per_minute: int = 60
    log_level: str = "INFO"
    testing: bool = False

    # extra="ignore": retired keys (QDRANT_*, CLIP_MODEL_NAME, EMBEDDING_DIM) in old .env files.
    model_config = ConfigDict(
        env_file=".env", env_prefix="", case_sensitive=False, extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""
    return Settings()
