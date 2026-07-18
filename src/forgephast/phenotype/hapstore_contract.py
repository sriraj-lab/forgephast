"""Forgephast assumptions about the HapStore on-disk contract."""

from __future__ import annotations

from typing import Any

HAPSTORE_MISSING_SENTINEL = 255


def validate_hapstore_missing_sentinel(
    store: object,
    chrom: str | int,
    *,
    context: str,
) -> None:
    attrs_reader = getattr(store, "attrs", None)
    if not callable(attrs_reader):
        raise ValueError(
            f"{context}: HapStore does not expose chromosome attributes; "
            "cannot verify the missing-genotype sentinel"
        )
    attrs: Any = attrs_reader(chrom)
    if not isinstance(attrs, dict) or "missing_sentinel" not in attrs:
        raise ValueError(
            f"{context}: HapStore chromosome attributes do not declare missing_sentinel"
        )
    observed = int(attrs["missing_sentinel"])
    if observed != HAPSTORE_MISSING_SENTINEL:
        raise ValueError(
            f"{context}: HapStore missing_sentinel={observed}, expected "
            f"{HAPSTORE_MISSING_SENTINEL}"
        )


__all__ = ["HAPSTORE_MISSING_SENTINEL", "validate_hapstore_missing_sentinel"]
