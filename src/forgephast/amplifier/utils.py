"""Utility helpers for population-amplifier."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from forgephast._internal.seeding import stable_seed as _shared_stable_seed
from forgephast._internal.logging import setup_logging as _shared_setup_logging

logger = logging.getLogger(__name__)



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def read_table(path: str | Path, sep: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if sep is None:
        if p.suffix.lower() in {".tsv", ".txt"}:
            sep = "\t"
        else:
            sep = ","
    df = pd.read_csv(p, sep=sep)
    # Strip whitespace and carriage-return artifacts from string columns.
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    return df



def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)



def apply_selector(series: pd.Series, selector: Any) -> pd.Series:
    values = series.astype(str)

    if isinstance(selector, dict):
        mask = pd.Series([True] * len(values), index=values.index)
        if "values" in selector:
            valset = {str(v) for v in selector["values"]}
            mask &= values.isin(valset)
        if "equals" in selector:
            mask &= values == str(selector["equals"])
        if "regex" in selector:
            mask &= values.str.contains(str(selector["regex"]), regex=True)
        return mask

    if isinstance(selector, list):
        return values.isin({str(v) for v in selector})

    return values == str(selector)



def derived_seed(base_seed: int, *parts: object) -> int:
    return int(_shared_stable_seed(int(base_seed), *parts, bits=31))



def shard_bounds(total: int, shard_id: int, n_shards: int) -> tuple[int, int]:
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")
    if shard_id < 0 or shard_id >= n_shards:
        raise ValueError(f"shard_id {shard_id} out of range for n_shards={n_shards}")

    base = total // n_shards
    rem = total % n_shards
    size = base + (1 if shard_id < rem else 0)
    start = shard_id * base + min(shard_id, rem)
    return start, size



def _default_log_dir(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            outputs = payload.get("outputs", {}) if isinstance(payload, dict) else {}
            root = outputs.get("root")
            name = payload.get("name") if isinstance(payload, dict) else None
            if root and name:
                return Path(root) / name / "logs" / "runs"
            if root:
                return Path(root) / "logs" / "runs"
        except Exception:
            # Fall back to cwd-based logs when config parsing fails.
            pass
    return Path.cwd() / "logs" / "runs"


def resolve_log_file_path(
    command: str | None = None,
    config_path: str | Path | None = None,
    explicit_log_file: str | Path | None = None,
    explicit_log_dir: str | Path | None = None,
    disable_file_logging: bool = False,
) -> Path | None:
    if disable_file_logging or os.environ.get("POPAMP_DISABLE_FILE_LOG", "").strip() == "1":
        return None

    if explicit_log_file is not None:
        return Path(explicit_log_file)

    env_log_file = os.environ.get("POPAMP_LOG_FILE")
    if env_log_file:
        return Path(env_log_file)

    if explicit_log_dir is not None:
        log_dir = Path(explicit_log_dir)
    else:
        env_log_dir = os.environ.get("POPAMP_LOG_DIR")
        log_dir = Path(env_log_dir) if env_log_dir else _default_log_dir(config_path=config_path)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cmd = (command or "run").replace(" ", "_")
    return log_dir / f"popamp_{cmd}_{ts}.log"


def setup_logging(
    level: int = logging.INFO,
    *,
    log_file: str | Path | None = None,
    reset_handlers: bool = True,
    stream_stdout: bool = True,
    logger_name: str | None = None,
    fmt: str | None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    respect_env: bool = True,
) -> tuple[logging.Logger, Path | None]:
    resolved_log_file = Path(log_file) if log_file is not None else None

    env_level = os.environ.get("POPAMP_LOG_LEVEL") if respect_env else None
    if env_level:
        env_level_upper = env_level.strip().upper()
        env_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        try:
            level = env_map.get(env_level_upper, int(env_level_upper))
        except Exception:
            # If env is malformed, ignore it rather than breaking execution.
            pass

    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] pid=%(process)d %(name)s: %(message)s"

    logger_obj = _shared_setup_logging(
        log_file=resolved_log_file,
        level=level,
        reset_handlers=reset_handlers,
        stream_stdout=stream_stdout,
        logger_name=logger_name,
        fmt=fmt,
        datefmt=datefmt,
    )
    return logger_obj, resolved_log_file



def set_thread_caps() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
