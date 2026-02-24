"""Run data ingestion pipeline."""

from pathlib import Path

from loguru import logger

from swespy.config import get_settings
from swespy.data.pipeline import DataPipeline


if __name__ == "__main__":
    settings = get_settings()
    pipeline = DataPipeline(settings=settings, project_root=Path.cwd())
    output = pipeline.run()
    logger.info("Pipeline complete: {} rows", len(output))
