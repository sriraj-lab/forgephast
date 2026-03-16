"""Multiprocessing-safe logging helpers shared by amplifier and phenotype.

Provides a QueueHandler/QueueListener pattern so worker processes can
log through a single parent-process writer.
"""

from __future__ import annotations

import logging
from logging.handlers import QueueListener
from typing import Any

from forgephast._internal.logging import install_queue_handler as _shared_install_queue_handler


def setup_queue_listener_for_logger(
    logger: logging.Logger,
    *,
    queue: Any,
    respect_handler_level: bool = True,
) -> QueueListener:
    handlers = list(logger.handlers)
    listener = QueueListener(queue, *handlers, respect_handler_level=respect_handler_level)
    listener.start()
    return listener


def install_queue_handler(
    *,
    queue: Any,
    level: int = logging.INFO,
    reset_handlers: bool = True,
    logger_name: str | None = None,
) -> logging.Logger:
    """Replace handlers on the target logger with a QueueHandler."""
    return _shared_install_queue_handler(
        queue=queue,
        level=int(level),
        logger_name=logger_name,
        reset_handlers=reset_handlers,
        propagate=False,
    )
