"""Shared subprocess execution helpers."""

from __future__ import annotations

import logging
import subprocess


def run_checked_command(
    cmd: list[str],
    *,
    label: str,
    logger: logging.Logger | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and raise a rich error on failure."""
    if logger is not None:
        logger.info("Running %s: %s", label, " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return completed
