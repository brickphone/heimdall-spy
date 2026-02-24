"""Run model fine-tuning."""

from pathlib import Path

from loguru import logger

from swespy.config import get_settings
from swespy.model.trainer import Trainer


if __name__ == "__main__":
    settings = get_settings()
    trainer = Trainer(settings=settings, project_root=Path.cwd())
    trainer.train(epochs=5)
    logger.info("Training run complete")
