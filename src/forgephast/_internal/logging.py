"""Shared logging setup."""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any


def _reset_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            # Best-effort cleanup; a stale handler should not block reconfiguration.
            pass


def _build_handlers(
    *,
    log_file: Path | str | None,
    level: int,
    mode: str,
    stream_stdout: bool,
    formatter: logging.Formatter,
) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []
    if stream_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    return handlers


def setup_logging(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    mode: str = "a",
    reset_handlers: bool = True,
    stream_stdout: bool = True,
    logger_name: str | None = None,
    fmt: str | None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure and return a logger with optional file and console handlers."""
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if reset_handlers:
        _reset_logger_handlers(logger)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    for handler in _build_handlers(
        log_file=log_file,
        level=level,
        mode=mode,
        stream_stdout=stream_stdout,
        formatter=formatter,
    ):
        logger.addHandler(handler)

    return logger


def setup_queue_listener(
    *,
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    mode: str = "a",
    stream_stdout: bool = False,
    fmt: str | None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    mp_context: BaseContext | None = None,
    queue_size: int = -1,
) -> tuple[Any, QueueListener]:
    """Create and start a queue listener that writes to file/stdout handlers.

    This is intended for multiprocessing workloads where worker processes emit
    logs via QueueHandler and the parent owns all file I/O.
    """
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] pid=%(process)d %(name)s: %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handlers = _build_handlers(
        log_file=log_file,
        level=level,
        mode=mode,
        stream_stdout=stream_stdout,
        formatter=formatter,
    )
    if not handlers:
        raise ValueError("setup_queue_listener requires at least one output handler (log_file or stream_stdout)")

    ctx = mp_context if mp_context is not None else mp.get_context()
    queue: Any = ctx.Queue(maxsize=int(queue_size))
    listener = QueueListener(queue, *handlers, respect_handler_level=True)
    listener.start()
    return queue, listener


def install_queue_handler(
    queue: Any,
    *,
    level: int = logging.INFO,
    logger_name: str | None = None,
    reset_handlers: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Attach a QueueHandler to a logger for multiprocessing-safe logging."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if reset_handlers:
        _reset_logger_handlers(logger)
    handler = QueueHandler(queue)
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = bool(propagate)
    return logger


def stop_queue_listener(listener: QueueListener, queue: Any | None = None) -> None:
    """Stop queue listener and close queue resources when available."""
    listener.stop()
    if queue is None:
        return
    if hasattr(queue, "close"):
        try:
            queue.close()
        except Exception:
            pass
    if hasattr(queue, "join_thread"):
        try:
            queue.join_thread()
        except Exception:
            pass
