"""Shared runtime cache helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_hash(payload: dict[str, Any]) -> str:
    """Deterministic hash for JSON-serializable cache keys."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def bump_counter(runtime_state: dict[str, Any] | None, *, hit: bool, name: str) -> None:
    """Track cache hits and misses by name inside a mutable runtime state."""
    if runtime_state is None:
        return
    bucket = "cache_hits" if hit else "cache_misses"
    counters = runtime_state.setdefault(bucket, {})
    counters[name] = int(counters.get(name, 0)) + 1


def snapshot_counters(runtime_state: dict[str, Any] | None) -> tuple[dict[str, int], dict[str, int]]:
    """Return hit and miss counters as plain int dictionaries."""
    if runtime_state is None:
        return {}, {}
    hits = {str(k): int(v) for k, v in runtime_state.get("cache_hits", {}).items()}
    misses = {str(k): int(v) for k, v in runtime_state.get("cache_misses", {}).items()}
    return hits, misses


def counter_delta(current: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    """Compute counter deltas between two snapshots."""
    keys = set(current) | set(baseline)
    return {key: int(current.get(key, 0) - baseline.get(key, 0)) for key in sorted(keys)}
