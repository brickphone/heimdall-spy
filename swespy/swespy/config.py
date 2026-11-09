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


def _segments_intersect(p1, p2, q1, q2) -> bool:  # type: ignore[no-untyped-def]
    def orient(a, b, c):  # type: ignore[no-untyped-def]
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def polygons_overlap(a: tuple[tuple[float, float], ...], b: tuple[tuple[float, float], ...]
                     ) -> bool:
    """True if two simple polygons share any area (vertex inside, or crossing edges)."""
    if any(in_area(x, y, b) for x, y in a) or any(in_area(x, y, a) for x, y in b):
        return True
    ea = list(zip(a, a[1:] + a[:1], strict=True))
    eb = list(zip(b, b[1:] + b[:1], strict=True))
    return any(_segments_intersect(p1, p2, q1, q2) for p1, p2 in ea for q1, q2 in eb)


# Frozen majority-location baseline (decision 2026-11-08): the city-centre location from
# baseline_v2 (mean of the most populated 1 km UTM cell [536, 6474] of the original
# 2,489-reference gallery). Fixed so the baseline never moves with gallery composition.
MAJORITY_BASELINE_LATLNG: tuple[float, float] = (58.410117221872895, 15.626609382633434)


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
    # Default gallery (decision 2026-11-08): gallery v2 "historic", 91,714 references.
    gallery_index: str = "data/gallery_v2/gallery-index-historic.json"
    retrieval_depth: int = 50
    # Geometric re-ranking (Part 5); weight/threshold tuned on the dev set only.
    rerank_enabled: bool = True
    rerank_k: int = 5  # Candidates verified (sweep 2026-11-09: K=5 matches K=50 on dev, 4 s).
    rerank_device: str = "auto"  # ALIKED device; LightGlue runs on CPU when this is MPS.
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


from swespy.cities import TRAINING_CITIES  # noqa: E402

for _name, _city in TRAINING_CITIES.items():
    if polygons_overlap(_city["polygon"], LINKOPING_AREA):  # type: ignore[arg-type]
        raise ValueError(f"Training city {_name} overlaps the Linköping evaluation polygon")
