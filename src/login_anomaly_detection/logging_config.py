"""Structured logging setup for the anomaly detection pipeline."""

import logging
import os


def configure_logging() -> logging.Logger:
    """Create and return a configured logger.

    Log level is controlled via the LOG_LEVEL environment variable.
    Defaults to INFO.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("login_anomaly")
    logger.setLevel(level)
    return logger
