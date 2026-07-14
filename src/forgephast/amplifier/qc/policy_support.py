"""QC pair-selection and runtime policy helpers."""

from __future__ import annotations

from itertools import combinations

from ..config import RunConfig


def pair_key(a: str, b: str) -> str:
    return f"{a}__{b}" if a <= b else f"{b}__{a}"


def resolve_pairwise_sets(config: RunConfig, pop_names: list[str]) -> list[tuple[str, str]]:
    if not config.qc.pairwise_fst_sets:
        return list(combinations(sorted(pop_names), 2))

    pairs: list[tuple[str, str]] = []
    for pair in config.qc.pairwise_fst_sets:
        if len(pair) != 2:
            raise ValueError(f"invalid pairwise_fst_sets entry: {pair}")
        pairs.append((str(pair[0]), str(pair[1])))
    return pairs


def resolve_within_split_checks(config: RunConfig, pop_names: list[str]) -> list[str]:
    if config.qc.within_pop_split_checks is None:
        return list(pop_names)
    return [str(value) for value in config.qc.within_pop_split_checks]


def resolve_python_qc_numba_threads(config: RunConfig) -> int:
    if config.qc.numba_threads is not None:
        return max(1, int(config.qc.numba_threads))
    return max(1, int(config.qc.matrix_workers))
