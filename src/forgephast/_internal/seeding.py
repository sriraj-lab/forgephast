"""Deterministic seed derivation helpers."""

from __future__ import annotations

import hashlib


def stable_seed(base_seed: int, *parts: object, bits: int = 31) -> int:
    """Derive a deterministic integer seed from a base seed and context parts.

    Args:
        base_seed: Base integer seed.
        *parts: Additional context parts to include in the hash.
        bits: Number of low bits to retain (default 31 for broad compatibility).

    Returns:
        Deterministic derived seed in the range [0, 2**bits - 1].
    """
    if bits <= 0 or bits > 63:
        raise ValueError("bits must be in the range [1, 63]")

    hasher = hashlib.sha256()
    hasher.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        hasher.update(b"|")
        hasher.update(str(part).encode("utf-8"))

    raw = int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)
    mask = (1 << bits) - 1
    return raw & mask
