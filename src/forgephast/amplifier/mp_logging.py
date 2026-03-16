"""Multiprocessing-safe logging helpers for the amplifier.

Delegates to the shared implementation in ``forgephast._internal.mp_logging``.
"""

from forgephast._internal.mp_logging import (  # noqa: F401
    install_queue_handler,
    setup_queue_listener_for_logger,
)
