"""Logging utilities for pipeline runs."""

from __future__ import annotations

import logging
from pathlib import Path

from forgephast._internal.logging import setup_logging as setup_shared_logging


def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger via forgephast shared logging utilities."""
    return setup_shared_logging(
        log_file=log_file,
        level=level,
        mode="a",
        reset_handlers=True,
        stream_stdout=True,
        logger_name=None,
    )
