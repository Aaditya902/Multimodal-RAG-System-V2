"""
Centralized logger configuration.
Following 12-Factor: log to stdout as a stream, not to files.
"""

import logging
import sys
from config import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with consistent formatting.
    Usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        level = logging.DEBUG if config.debug else logging.INFO
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger