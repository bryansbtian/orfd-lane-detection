"""Structured logging setup for offroad_autonomy."""

from __future__ import annotations

import logging
import sys


def setup_logger(
    name: str = "offroad_autonomy",
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger with a clean console format."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
