"""Grid expansion and execution helpers for simulation sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import itertools
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class GridFactor:
    """One grid factor which may set one or more override paths."""

    paths: tuple[str, ...]
    values: list[tuple[JsonValue, ...]]


@dataclass(frozen=True)
class GridRuntime:
    """Runtime options for one grid invocation."""

    max_cache_gb: float = 8.0


@dataclass(frozen=True)
class GridSpec:
    """Typed grid spec loaded from YAML."""

    base_config: str
    stage: str
    factors: list[GridFactor]
    parallel_runs: int
    continue_on_error: bool
    runtime: GridRuntime


@dataclass(frozen=True)
class GridCase:
    """One expanded grid case."""

    index: int
    payload: dict[str, Any]
    overrides: dict[str, JsonValue]
    a_key: tuple[tuple[str, str], ...]
    b_key: tuple[tuple[str, str], ...]
    c_key: tuple[tuple[str, str], ...]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _factor_class(path: str) -> str:
    if path.endswith(".params.h2") or path.startswith("variance_budget."):
        return "C"
    if path.startswith("effect_model.") or path.endswith(".params.sigma_beta"):
        return "B"
    return "A"


def _path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    token = ""
    idx_token = ""
    in_index = False
    for ch in path:
        if ch == "." and not in_index:
            if token:
                tokens.append(token)
                token = ""
            continue
        if ch == "[":
            if token:
                tokens.append(token)
                token = ""
            in_index = True
            idx_token = ""
            continue
        if ch == "]":
            if not in_index:
                raise ValueError(f"Invalid override path {path!r}: unmatched ']'")
            if not idx_token.isdigit():
                raise ValueError(f"Invalid override path {path!r}: non-numeric index {idx_token!r}")
            tokens.append(int(idx_token))
            in_index = False
            idx_token = ""
            continue
        if in_index:
            idx_token += ch
        else:
            token += ch
    if in_index:
        raise ValueError(f"Invalid override path {path!r}: missing closing ']'")
    if token:
        tokens.append(token)
    return tokens


def _set_path(payload: dict[str, Any], path: str, value: JsonValue) -> None:
    tokens = _path_tokens(path)
    if not tokens:
        raise ValueError(f"Empty override path: {path!r}")
    cur: Any = payload
    for tok in tokens[:-1]:
        if isinstance(tok, int):
            if not isinstance(cur, list):
                raise ValueError(f"Expected list while traversing {path!r}")
            if tok < 0 or tok >= len(cur):
                raise IndexError(f"Index {tok} out of bounds while traversing {path!r}")
            cur = cur[tok]
        else:
            if not isinstance(cur, dict):
                raise ValueError(f"Expected mapping while traversing {path!r}")
            if tok not in cur:
                raise KeyError(f"Unknown override path segment {tok!r} in {path!r}")
            cur = cur[tok]
    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(cur, list):
            raise ValueError(f"Expected list at leaf while setting {path!r}")
        if last < 0 or last >= len(cur):
            raise IndexError(f"Index {last} out of bounds while setting {path!r}")
        cur[last] = value
    else:
        if not isinstance(cur, dict):
            raise ValueError(f"Expected mapping at leaf while setting {path!r}")
        if last not in cur:
            raise KeyError(f"Unknown override key {last!r} in {path!r}")
        cur[last] = value


def load_grid_spec(path: str | Path) -> GridSpec:
    """Load and validate grid spec YAML."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Grid spec must be a mapping")

    base_config = str(payload.get("base_config", "")).strip()
    if not base_config:
        raise ValueError("grid.base_config is required")

    stage = str(payload.get("stage", "simulate")).strip()
    if stage not in {"simulate", "full"}:
        raise ValueError("grid.stage must be one of {'simulate','full'}")

    factors_payload = payload.get("factors", [])
    if factors_payload is None:
        factors_payload = []
    if not isinstance(factors_payload, list):
        raise ValueError("grid.factors must be a list")
    factors: list[GridFactor] = []
    for item in factors_payload:
        if not isinstance(item, dict):
            raise ValueError("Each factor entry must be a mapping")
        has_path = "path" in item and item.get("path") is not None
        has_paths = "paths" in item and item.get("paths") is not None
        if has_path and has_paths:
            raise ValueError("Each factor must specify exactly one of 'path' or 'paths'")

        raw_values = item.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("Each factor must provide a non-empty values list")

        if has_paths:
            raw_paths = item.get("paths")
            if not isinstance(raw_paths, list) or not raw_paths:
                raise ValueError("factor.paths must be a non-empty list")
            paths = [str(p).strip() for p in raw_paths]
            if any(not p for p in paths):
                raise ValueError("factor.paths entries cannot be empty")
            values: list[tuple[JsonValue, ...]] = []
            for idx, entry in enumerate(raw_values):
                if not isinstance(entry, (list, tuple)):
                    raise ValueError("factor.values entries must be lists for coupled factor.paths")
                if len(entry) != len(paths):
                    raise ValueError(
                        "factor.values[%d] must have length %d to match factor.paths (got %d)"
                        % (idx, len(paths), len(entry))
                    )
                values.append(tuple(entry))
            factors.append(GridFactor(paths=tuple(paths), values=values))
        else:
            factor_path = str(item.get("path", "")).strip()
            if not factor_path:
                raise ValueError("factor.path must be a non-empty string")
            values = [(v,) for v in raw_values]
            factors.append(GridFactor(paths=(factor_path,), values=values))

    parallel_runs = int(payload.get("parallel_runs", 1))
    if parallel_runs <= 0:
        raise ValueError("grid.parallel_runs must be > 0")

    continue_on_error = bool(payload.get("continue_on_error", False))
    runtime_payload = payload.get("runtime", {}) or {}
    if not isinstance(runtime_payload, dict):
        raise ValueError("grid.runtime must be a mapping when provided")
    max_cache_gb = float(runtime_payload.get("max_cache_gb", 8.0))
    if max_cache_gb <= 0:
        raise ValueError("grid.runtime.max_cache_gb must be > 0")

    return GridSpec(
        base_config=base_config,
        stage=stage,
        factors=factors,
        parallel_runs=parallel_runs,
        continue_on_error=continue_on_error,
        runtime=GridRuntime(max_cache_gb=max_cache_gb),
    )


def expand_grid_cases(base_payload: dict[str, Any], factors: Sequence[GridFactor]) -> list[GridCase]:
    """Expand Cartesian factor combinations into concrete case payloads."""
    if not factors:
        payload = copy.deepcopy(base_payload)
        base_name = str(payload.get("name", "run"))
        payload["name"] = f"{base_name}_grid_0001"
        return [
            GridCase(
                index=1,
                payload=payload,
                overrides={},
                a_key=tuple(),
                b_key=tuple(),
                c_key=tuple(),
            )
        ]

    factor_paths = [f.paths for f in factors]
    values_lists = [f.values for f in factors]
    out: list[GridCase] = []
    base_name = str(base_payload.get("name", "run"))
    for idx, choice in enumerate(itertools.product(*values_lists), start=1):
        payload = copy.deepcopy(base_payload)
        overrides: dict[str, JsonValue] = {}
        a_parts: list[tuple[str, str]] = []
        b_parts: list[tuple[str, str]] = []
        c_parts: list[tuple[str, str]] = []
        for paths, values in zip(factor_paths, choice):
            for path, value in zip(paths, values):
                _set_path(payload, path, value)
                overrides[path] = value
                cls = _factor_class(path)
                pair = (path, _canonical(value))
                if cls == "A":
                    a_parts.append(pair)
                elif cls == "B":
                    b_parts.append(pair)
                else:
                    c_parts.append(pair)
        payload["name"] = f"{base_name}_grid_{idx:04d}"
        out.append(
            GridCase(
                index=idx,
                payload=payload,
                overrides=overrides,
                a_key=tuple(a_parts),
                b_key=tuple(b_parts),
                c_key=tuple(c_parts),
            )
        )
    return out


def build_grouped_cases(cases: Sequence[GridCase]) -> list[list[GridCase]]:
    """Group cases by class-A signature and order for cache locality."""
    groups: dict[tuple[tuple[str, str], ...], list[GridCase]] = {}
    for case in cases:
        groups.setdefault(case.a_key, []).append(case)
    grouped = []
    for group_cases in groups.values():
        ordered = sorted(group_cases, key=lambda c: (c.b_key, c.c_key, c.index))
        grouped.append(ordered)
    grouped.sort(key=lambda grp: tuple(case.index for case in grp))
    return grouped


def execute_group(
    cases: Sequence[GridCase],
    *,
    stage: str,
    max_cache_gb: float,
    run_case: Callable[[dict[str, Any], str, dict[str, Any], int], dict[str, Any]],
    continue_on_error: bool,
    hapstore_io_threads: int | None = None,
) -> list[dict[str, Any]]:
    """Execute one class-A case group with one in-memory runtime cache."""
    case_indices = [int(case.index) for case in cases]
    logger.info(
        "GRID_GROUP_START n_cases=%d case_indices=%s stage=%s max_cache_gb=%.2f",
        len(case_indices),
        case_indices,
        stage,
        float(max_cache_gb),
    )
    runtime_state: dict[str, Any] = {
        "max_cache_bytes": int(max_cache_gb * (1024**3)),
        "cache_hits": {},
        "cache_misses": {},
    }
    if hapstore_io_threads is not None:
        runtime_state["hapstore_io_threads"] = int(hapstore_io_threads)
    rows: list[dict[str, Any]] = []
    for case in cases:
        logger.info("GRID_CASE_START case_index=%d stage=%s", int(case.index), stage)
        started = time.perf_counter()
        try:
            result = run_case(case.payload, stage, runtime_state, case.index)
            status = "success"
            error = ""
        except Exception as exc:  # noqa: BLE001
            result = {}
            status = "failed"
            error = str(exc)
            if not continue_on_error:
                row = {
                    "case_index": case.index,
                    "status": status,
                    "error": error,
                    "stage": stage,
                    "a_key": _canonical(case.a_key),
                    "b_key": _canonical(case.b_key),
                    "c_key": _canonical(case.c_key),
                    "overrides": _canonical(case.overrides),
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
                rows.append(row)
                raise RuntimeError(_canonical(row)) from exc
        row = {
            "case_index": case.index,
            "status": status,
            "error": error,
            "stage": stage,
            "a_key": _canonical(case.a_key),
            "b_key": _canonical(case.b_key),
            "c_key": _canonical(case.c_key),
            "overrides": _canonical(case.overrides),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        row.update(result)
        logger.info(
            "GRID_CASE_DONE case_index=%d status=%s elapsed_seconds=%.3f",
            int(case.index),
            status,
            float(row["elapsed_seconds"]),
        )
        rows.append(row)
    logger.info(
        "GRID_GROUP_DONE n_cases=%d cache_hits=%s cache_misses=%s",
        len(case_indices),
        json.dumps(runtime_state.get("cache_hits", {}), sort_keys=True),
        json.dumps(runtime_state.get("cache_misses", {}), sort_keys=True),
    )
    return rows


def write_grid_manifest(
    *,
    base_output_root: str,
    base_name: str,
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Write aggregate grid manifest."""
    summary_dir = Path(base_output_root) / f"{base_name}_grid"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("case_index").reset_index(drop=True)
    out_path = summary_dir / "grid_manifest.tsv"
    frame.to_csv(out_path, sep="\t", index=False)
    timing_cols = [col for col in ["case_index", "stage", "status", "elapsed_seconds"] if col in frame]
    if timing_cols:
        (summary_dir / "grid_timing.tsv").parent.mkdir(parents=True, exist_ok=True)
        frame[timing_cols].to_csv(summary_dir / "grid_timing.tsv", sep="\t", index=False)
    logger.info(
        "GRID_MANIFEST_WRITTEN rows=%d path=%s timing_path=%s",
        int(frame.shape[0]),
        out_path,
        summary_dir / "grid_timing.tsv",
    )
    return out_path
