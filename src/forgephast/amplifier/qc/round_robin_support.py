"""Round-robin QC support helpers."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

def summarize_fst_values(values: list[float], n_donors: int, n_folds: int) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    summary: dict[str, float | int] = {
        "n_donors": int(n_donors),
        "n_folds": int(n_folds),
        "n_pairs": int(arr.size),
        "n_finite_pairs": int(finite.size),
    }
    if finite.size == 0:
        summary.update(
            {
                "fst_min": float("nan"),
                "fst_max": float("nan"),
                "fst_mean": float("nan"),
                "fst_std": float("nan"),
                "fst_p50": float("nan"),
                "fst_p95": float("nan"),
            }
        )
        return summary

    summary.update(
        {
            "fst_min": float(np.min(finite)),
            "fst_max": float(np.max(finite)),
            "fst_mean": float(np.mean(finite)),
            "fst_std": float(np.std(finite)),
            "fst_p50": float(np.quantile(finite, 0.5)),
            "fst_p95": float(np.quantile(finite, 0.95)),
        }
    )
    return summary


def validate_round_robin_n_folds(n_folds: int) -> None:
    if n_folds < 2 or n_folds > 10:
        raise ValueError("n_folds must be in [2, 10]")


def validate_donor_fold_fraction(fold_fraction: float) -> None:
    if fold_fraction < 0.05 or fold_fraction > 0.5:
        raise ValueError("fold_fraction must be in [0.05, 0.5]")


def round_robin_folds_from_fraction(fold_fraction: float) -> int:
    validate_donor_fold_fraction(fold_fraction)
    n_folds = int(round(1.0 / fold_fraction))
    return max(2, min(10, n_folds))


def round_robin_for_population_chrom(
    *,
    pop: str,
    chrom: int,
    sample_ids: list[str],
    n_folds: int,
    seed: int,
    numba_threads: int,
    even_folds_fn: Callable[[list[str], int, int], list[list[str]]],
    load_sample_haplotypes_fn: Callable[[list[str]], tuple[np.ndarray, np.ndarray]],
    matrix_hudson_fst_from_counts_fn: Callable[..., float],
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    validate_round_robin_n_folds(n_folds)
    folds = even_folds_fn(sample_ids, n_folds=n_folds, seed=seed)
    fold_sizes = [len(x) for x in folds]
    fold_positions: dict[int, np.ndarray] = {}
    fold_counts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    common_pos: np.ndarray | None = None
    for fold_idx, fold_ids in enumerate(folds):
        pos, haplotypes = load_sample_haplotypes_fn(list(fold_ids))
        pos_arr = np.asarray(pos, dtype=np.int64)
        haps_arr = np.asarray(haplotypes, dtype=np.uint8)
        if haps_arr.ndim != 2 or haps_arr.shape[0] % 2 != 0:
            raise ValueError(
                f"round-robin donor haplotypes must be 2D with an even haplotype axis, got {haps_arr.shape}"
            )
        genotypes = haps_arr[0::2, :].astype(np.uint8, copy=False) + haps_arr[1::2, :].astype(np.uint8, copy=False)
        ac = np.sum(genotypes, axis=0, dtype=np.float64)
        an = np.full(pos_arr.shape[0], 2.0 * float(genotypes.shape[0]), dtype=np.float64)
        fold_positions[fold_idx] = pos_arr
        fold_counts[fold_idx] = (ac, an)
        common_pos = pos_arr if common_pos is None else np.intersect1d(common_pos, pos_arr, assume_unique=False)

    records: list[dict[str, float | int | str]] = []
    if common_pos is None or common_pos.size < 2:
        fst_values = [float("nan")] * max(0, len(fold_sizes) * (len(fold_sizes) - 1) // 2)
        record_idx = 0
        for fold_a, fold_b in combinations(range(len(folds)), 2):
            records.append(
                {
                    "population": pop,
                    "fold_a": int(fold_a),
                    "fold_b": int(fold_b),
                    "n_a": int(fold_sizes[fold_a]),
                    "n_b": int(fold_sizes[fold_b]),
                    "fst": float(fst_values[record_idx]),
                }
            )
            record_idx += 1
        summary = summarize_fst_values(fst_values, n_donors=len(sample_ids), n_folds=len(folds))
        summary["fold_min_size"] = int(min(fold_sizes))
        summary["fold_max_size"] = int(max(fold_sizes))
        return records, summary

    aligned_counts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_idx, pos_arr in fold_positions.items():
        idx = np.searchsorted(pos_arr, common_pos)
        if idx.size != common_pos.size or not np.array_equal(pos_arr[idx], common_pos):
            raise RuntimeError(f"round-robin donor variant alignment mismatch for fold {fold_idx}")
        ac, an = fold_counts[fold_idx]
        aligned_counts[fold_idx] = (ac[idx], an[idx])

    fst_values: list[float] = []
    for fold_a, fold_b in combinations(range(len(folds)), 2):
        ac_a, an_a = aligned_counts[fold_a]
        ac_b, an_b = aligned_counts[fold_b]
        fst = float(
            matrix_hudson_fst_from_counts_fn(
                ac_a,
                an_a,
                ac_b,
                an_b,
                numba_threads=max(1, int(numba_threads)),
            )
        )
        records.append(
            {
                "population": pop,
                "fold_a": int(fold_a),
                "fold_b": int(fold_b),
                "n_a": int(fold_sizes[fold_a]),
                "n_b": int(fold_sizes[fold_b]),
                "fst": float(fst),
            }
        )
        fst_values.append(float(fst))
    summary = summarize_fst_values(fst_values, n_donors=len(sample_ids), n_folds=len(folds))
    summary["fold_min_size"] = int(min(fold_sizes))
    summary["fold_max_size"] = int(max(fold_sizes))
    return records, summary

def pair_count(n: int) -> int:
    n_i = int(n)
    if n_i <= 1:
        return 0
    return (n_i * (n_i - 1)) // 2


def min_shards_for_target_pairs(target_pairs: int) -> int:
    k = int(target_pairs)
    if k <= 0:
        return 0
    n = 2
    while pair_count(n) < k:
        n += 1
    return n


def subset_shards_for_target_pairs(
    *,
    shard_ids: list[int],
    target_pairs: int,
    seed: int,
) -> list[int]:
    if not shard_ids:
        return []
    k = max(1, int(target_pairs))
    n = min(len(shard_ids), min_shards_for_target_pairs(k))
    if n >= len(shard_ids):
        return list(shard_ids)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(shard_ids), size=int(n), replace=False))
    return [shard_ids[int(i)] for i in idx.tolist()]
