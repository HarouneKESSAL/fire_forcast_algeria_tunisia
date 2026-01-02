"""Logging setup for pipeline."""
from __future__ import annotations
import logging
from typing import Optional

def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    logger = logging.getLogger()
    if logger.handlers:
        return  # already configured
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = fmt or "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    logging.basicConfig(level=lvl, format=fmt)
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)

