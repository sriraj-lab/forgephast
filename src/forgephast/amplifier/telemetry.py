"""Telemetry helpers for substep logging and lightweight resource snapshots.

This module is dependency-free (stdlib only). Snapshots are best-effort: missing
fields are omitted rather than raising.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import resource
import sys
from time import monotonic, perf_counter
from typing import Any, Iterator


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _try_read_text(path: str | Path) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:
        return None


def _parse_kb_value(text: str) -> int | None:
    # Typical line: "VmRSS:\t  123456 kB"
    parts = text.strip().split()
    if not parts:
        return None
    try:
        return int(parts[1])
    except Exception:
        return None


def resource_snapshot() -> dict[str, Any]:
    """Return a best-effort process resource snapshot (no exceptions)."""
    out: dict[str, Any] = {"ts_utc": _utc_ts(), "pid": int(os.getpid())}

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        out["cpu_user_sec"] = float(usage.ru_utime)
        out["cpu_sys_sec"] = float(usage.ru_stime)
        maxrss = float(getattr(usage, "ru_maxrss", 0.0))
        if maxrss > 0:
            # Linux: kilobytes. macOS: bytes.
            if sys.platform == "darwin":
                out["maxrss_mb"] = maxrss / (1024.0 * 1024.0)
            else:
                out["maxrss_mb"] = maxrss / 1024.0
    except Exception:
        pass

    status = _try_read_text("/proc/self/status")
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                kb = _parse_kb_value(line)
                if kb is not None:
                    out["rss_mb"] = float(kb) / 1024.0
            elif line.startswith("VmSize:"):
                kb = _parse_kb_value(line)
                if kb is not None:
                    out["vms_mb"] = float(kb) / 1024.0

    proc_io = _try_read_text("/proc/self/io")
    if proc_io:
        for line in proc_io.splitlines():
            if line.startswith("read_bytes:"):
                try:
                    out["io_read_mb"] = float(line.split(":", 1)[1].strip()) / (1024.0 * 1024.0)
                except Exception:
                    pass
            elif line.startswith("write_bytes:"):
                try:
                    out["io_write_mb"] = float(line.split(":", 1)[1].strip()) / (1024.0 * 1024.0)
                except Exception:
                    pass

    return out


def _fmt_kv(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(payload.keys()):
        val = payload[key]
        if val is None:
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.6g}")
        else:
            parts.append(f"{key}={val}")
    return " ".join(parts)


@contextmanager
def telemetry_span(logger: logging.Logger, name: str, **ctx: Any) -> Iterator[None]:
    """Context manager that logs substep start/end with elapsed and snapshots."""
    t0 = perf_counter()
    start_snap = resource_snapshot()
    logger.info("SUBSTEP start name=%s %s %s", name, _fmt_kv(ctx), _fmt_kv(start_snap))
    try:
        yield
    finally:
        elapsed = perf_counter() - t0
        end_snap = resource_snapshot()
        logger.info(
            "SUBSTEP end name=%s elapsed_sec=%.6g %s %s",
            name,
            elapsed,
            _fmt_kv(ctx),
            _fmt_kv(end_snap),
        )


@dataclass
class RateLimitedProgress:
    logger: logging.Logger
    name: str
    every_sec: float = 30.0
    _last: float = 0.0

    def log(self, *, force: bool = False, **ctx: Any) -> None:
        now = monotonic()
        if not force and self._last and (now - self._last) < float(self.every_sec):
            return
        self._last = now
        self.logger.info("PROGRESS name=%s %s", self.name, _fmt_kv(ctx))


def rate_limited_progress(logger: logging.Logger, name: str, every_sec: float = 30.0) -> RateLimitedProgress:
    return RateLimitedProgress(logger=logger, name=name, every_sec=float(every_sec))

