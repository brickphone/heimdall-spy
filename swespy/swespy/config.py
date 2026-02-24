"""Application configuration."""

from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    mapillary_access_token: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "swespy_embeddings"
    clip_model_name: str = "openai/clip-vit-large-patch14"
    s2_cell_level: int = 14
    sweden_bbox: tuple[float, float, float, float] = (10.5, 55.0, 24.2, 69.1)
    image_size: int = 224
    batch_size: int = 64
    max_images_per_cell: int = 50
    top_k_results: int = 5
    embedding_dim: int = 768
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
    vector_upsert_batch_size: int = 256
    train_learning_rate: float = 1e-4
    train_weight_decay: float = 1e-2
    log_level: str = "INFO"
    testing: bool = False

    model_config = ConfigDict(env_file=".env", env_prefix="", case_sensitive=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""
    return Settings()
