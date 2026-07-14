"""Matrix-based QC helpers for HapStore-backed FST computations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from hapstore import HapStore

try:
    from numba import njit, prange, set_num_threads

    HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised by environments without numba
    HAS_NUMBA = False
    njit = None
    prange = range
    set_num_threads = None

logger = logging.getLogger(__name__)


StoreLike = Union[str, Path, HapStore]
_MISSING_SENTINEL = 255
_LAST_NUMBA_THREADS: int | None = None


def _configure_numba_threads(numba_threads: int | None) -> None:
    global _LAST_NUMBA_THREADS
    if not HAS_NUMBA or set_num_threads is None or numba_threads is None:
        return
    threads = max(1, int(numba_threads))
    if _LAST_NUMBA_THREADS == threads:
        return
    set_num_threads(threads)
    _LAST_NUMBA_THREADS = threads


def _hudson_fst_from_counts_numpy(
    ac_a: np.ndarray,
    an_a: np.ndarray,
    ac_b: np.ndarray,
    an_b: np.ndarray,
) -> float:
    ac_a = np.asarray(ac_a, dtype=np.float64)
    an_a = np.asarray(an_a, dtype=np.float64)
    ac_b = np.asarray(ac_b, dtype=np.float64)
    an_b = np.asarray(an_b, dtype=np.float64)

    valid = (an_a >= 4.0) & (an_b >= 4.0)
    if not np.any(valid):
        return float("nan")

    ac_a = ac_a[valid]
    an_a = an_a[valid]
    ac_b = ac_b[valid]
    an_b = an_b[valid]

    # Match PLINK Hudson finite-sample correction: n is chromosome count (AN).
    n1 = an_a
    n2 = an_b
    valid_n = (n1 > 1.0) & (n2 > 1.0)
    if not np.any(valid_n):
        return float("nan")

    ac_a = ac_a[valid_n]
    an_a = an_a[valid_n]
    ac_b = ac_b[valid_n]
    an_b = an_b[valid_n]
    n1 = n1[valid_n]
    n2 = n2[valid_n]

    p1 = ac_a / an_a
    p2 = ac_b / an_b
    num = (p1 - p2) ** 2 - (p1 * (1.0 - p1) / (n1 - 1.0)) - (p2 * (1.0 - p2) / (n2 - 1.0))
    den = p1 * (1.0 - p2) + p2 * (1.0 - p1)
    valid_den = den > 0.0
    if not np.any(valid_den):
        return float("nan")

    return float(np.sum(num[valid_den]) / np.sum(den[valid_den]))


if HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _hudson_fst_from_counts_numba(
        ac_a: np.ndarray,
        an_a: np.ndarray,
        ac_b: np.ndarray,
        an_b: np.ndarray,
    ) -> float:
        num_sum = 0.0
        den_sum = 0.0
        n = ac_a.shape[0]
        for i in prange(n):
            an1 = an_a[i]
            an2 = an_b[i]
            if an1 < 4.0 or an2 < 4.0:
                continue
            n1 = an1
            n2 = an2
            if n1 <= 1.0 or n2 <= 1.0:
                continue
            p1 = ac_a[i] / an1
            p2 = ac_b[i] / an2
            den = p1 * (1.0 - p2) + p2 * (1.0 - p1)
            if den <= 0.0:
                continue
            num = (p1 - p2) * (p1 - p2) - (p1 * (1.0 - p1) / (n1 - 1.0)) - (p2 * (1.0 - p2) / (n2 - 1.0))
            num_sum += num
            den_sum += den
        if den_sum <= 0.0:
            return np.nan
        return num_sum / den_sum


    @njit(parallel=True, fastmath=True)
    def _count_haplotype_block_numba(h: np.ndarray, missing_sentinel: int) -> tuple[np.ndarray, np.ndarray]:
        n_haps, n_variants = h.shape
        ac = np.zeros(n_variants, dtype=np.float64)
        an = np.zeros(n_variants, dtype=np.float64)
        for j in prange(n_variants):
            ac_j = 0.0
            an_j = 0.0
            for i in range(n_haps):
                allele = h[i, j]
                if allele != missing_sentinel:
                    an_j += 1.0
                    if allele == 1:
                        ac_j += 1.0
            ac[j] = ac_j
            an[j] = an_j
        return ac, an


def hudson_fst_from_counts(
    ac_a: np.ndarray,
    an_a: np.ndarray,
    ac_b: np.ndarray,
    an_b: np.ndarray,
    *,
    numba_threads: int | None = None,
) -> float:
    """Compute Hudson FST from AC/AN arrays with optional Numba acceleration."""
    ac_a_arr = np.asarray(ac_a, dtype=np.float64)
    an_a_arr = np.asarray(an_a, dtype=np.float64)
    ac_b_arr = np.asarray(ac_b, dtype=np.float64)
    an_b_arr = np.asarray(an_b, dtype=np.float64)

    if HAS_NUMBA:
        _configure_numba_threads(numba_threads)
        return float(_hudson_fst_from_counts_numba(ac_a_arr, an_a_arr, ac_b_arr, an_b_arr))
    return _hudson_fst_from_counts_numpy(ac_a_arr, an_a_arr, ac_b_arr, an_b_arr)


def _count_haplotype_block(
    h: np.ndarray,
    *,
    numba_threads: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if HAS_NUMBA and np.issubdtype(h.dtype, np.integer):
        _configure_numba_threads(numba_threads)
        if h.dtype == np.uint8 and h.flags.c_contiguous:
            return _count_haplotype_block_numba(h, _MISSING_SENTINEL)
        h_u8 = np.ascontiguousarray(h, dtype=np.uint8)
        return _count_haplotype_block_numba(h_u8, _MISSING_SENTINEL)

    ac = np.sum(h == 1, axis=0, dtype=np.float64)
    an = np.sum(h != _MISSING_SENTINEL, axis=0, dtype=np.float64)
    return ac, an



def sample_population_indices(
    *,
    all_sample_ids: Sequence[str],
    groups: dict[str, list[str]],
    max_diploids_per_population: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resolve per-population sample indices over a store sample axis."""
    id_to_idx = {str(sid): idx for idx, sid in enumerate(all_sample_ids)}
    out: dict[str, np.ndarray] = {}

    for pop in sorted(groups.keys()):
        pop_ids = sorted({str(x) for x in groups[pop]})
        missing = [sid for sid in pop_ids if sid not in id_to_idx]
        if missing:
            raise ValueError(
                f"{len(missing)} samples for population {pop!r} are missing from hapstore sample axis; "
                f"first={missing[:5]}"
            )

        indices = np.asarray([id_to_idx[sid] for sid in pop_ids], dtype=np.int64)
        if max_diploids_per_population is not None and indices.size > int(max_diploids_per_population):
            rng = np.random.default_rng(int(seed))
            chosen = np.sort(rng.choice(indices.size, size=int(max_diploids_per_population), replace=False))
            indices = indices[chosen]
        out[pop] = indices

    return out



def sample_variant_indices(
    *,
    n_variants: int,
    max_variants: int | None,
    seed: int,
) -> np.ndarray | None:
    """Return sorted sampled variant indices, or None for full variant axis."""
    if n_variants <= 0:
        raise ValueError(f"n_variants must be > 0, got {n_variants}")
    if max_variants is None or max_variants >= n_variants:
        return None
    if max_variants <= 0:
        raise ValueError(f"max_variants must be > 0 when provided, got {max_variants}")

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n_variants, size=int(max_variants), replace=False))
    return np.asarray(idx, dtype=np.int64)



def hapstore_counts_for_samples(
    *,
    store: StoreLike,
    chrom: str | int,
    sample_indices: np.ndarray | None,
    variant_indices: np.ndarray | None,
    variant_block_size: int,
    numba_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AC/AN across selected samples and variants from a HapStore chromosome."""
    hs = _as_store(store)
    n_samples = hs.n_samples(chrom)
    n_variants = hs.n_variants(chrom)

    if variant_block_size <= 0:
        raise ValueError(f"variant_block_size must be > 0, got {variant_block_size}")

    sample_mask = None
    if sample_indices is not None:
        sample_idx = np.asarray(sample_indices, dtype=np.int64)
        if sample_idx.ndim != 1:
            raise ValueError(f"sample_indices must be 1D, got shape={sample_idx.shape}")
        if sample_idx.size and (int(sample_idx.min()) < 0 or int(sample_idx.max()) >= n_samples):
            raise ValueError(
                f"sample_indices out of bounds for n_samples={n_samples}: "
                f"min={int(sample_idx.min())}, max={int(sample_idx.max())}"
            )
        sample_mask = np.zeros(n_samples, dtype=bool)
        sample_mask[sample_idx] = True

    if variant_indices is None:
        ac = np.zeros(n_variants, dtype=np.float64)
        an = np.zeros(n_variants, dtype=np.float64)
        for start in range(0, n_variants, int(variant_block_size)):
            end = min(start + int(variant_block_size), n_variants)
            h = hs.haplotypes(chrom, variant_slice=(start, end), sample_mask=sample_mask)
            ac_blk, an_blk = _count_haplotype_block(h, numba_threads=numba_threads)
            ac[start:end] = ac_blk
            an[start:end] = an_blk
        return ac, an

    variant_idx = np.asarray(variant_indices, dtype=np.int64)
    if variant_idx.ndim != 1:
        raise ValueError(f"variant_indices must be 1D, got shape={variant_idx.shape}")
    if variant_idx.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if int(variant_idx.min()) < 0 or int(variant_idx.max()) >= n_variants:
        raise ValueError(
            f"variant_indices out of bounds for n_variants={n_variants}: "
            f"min={int(variant_idx.min())}, max={int(variant_idx.max())}"
        )

    order = np.argsort(variant_idx)
    sorted_idx = variant_idx[order]
    ac_sorted = np.zeros(sorted_idx.size, dtype=np.float64)
    an_sorted = np.zeros(sorted_idx.size, dtype=np.float64)

    for src_start, src_end, out_start, out_end in _contiguous_runs(sorted_idx):
        h = hs.haplotypes(chrom, variant_slice=(int(src_start), int(src_end)), sample_mask=sample_mask)
        ac_blk, an_blk = _count_haplotype_block(h, numba_threads=numba_threads)
        ac_sorted[out_start:out_end] = ac_blk
        an_sorted[out_start:out_end] = an_blk

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size, dtype=np.int64)
    return ac_sorted[inv], an_sorted[inv]



def aggregate_counts_across_stores(
    *,
    stores: Sequence[StoreLike],
    chrom: str | int,
    variant_indices: np.ndarray | None,
    variant_block_size: int,
    numba_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AC/AN by summing counts from multiple stores for one chromosome."""
    if not stores:
        raise ValueError("stores must not be empty")

    resolved = [_as_store(x) for x in stores]
    _validate_variant_axis_match(resolved, chrom=chrom)

    ac_sum: np.ndarray | None = None
    an_sum: np.ndarray | None = None
    for hs in resolved:
        ac, an = hapstore_counts_for_samples(
            store=hs,
            chrom=chrom,
            sample_indices=None,
            variant_indices=variant_indices,
            variant_block_size=variant_block_size,
            numba_threads=numba_threads,
        )
        if ac_sum is None:
            ac_sum = np.asarray(ac, dtype=np.float64)
            an_sum = np.asarray(an, dtype=np.float64)
        else:
            ac_sum += ac
            an_sum += an

    if ac_sum is None or an_sum is None:
        raise RuntimeError("aggregate_counts_across_stores failed to compute counts")
    return ac_sum, an_sum



def _as_store(value: StoreLike) -> HapStore:
    return value if isinstance(value, HapStore) else HapStore(value, mode="r")



def _validate_variant_axis_match(stores: Sequence[HapStore], chrom: str | int) -> None:
    reference = stores[0].variant_positions(chrom)
    for idx, hs in enumerate(stores[1:], start=1):
        current = hs.variant_positions(chrom)
        if current.shape != reference.shape or not np.array_equal(current, reference):
            raise ValueError(
                "variant axis mismatch across stores for aggregate_counts_across_stores: "
                f"store[0]={stores[0].path}, store[{idx}]={hs.path}"
            )



def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Map sorted variant indices into contiguous source runs.

    Returns tuples: (src_start, src_end_exclusive, out_start, out_end_exclusive).
    """
    if indices.size == 0:
        return []

    runs: list[tuple[int, int, int, int]] = []
    run_src_start = int(indices[0])
    run_out_start = 0
    prev = int(indices[0])

    for out_idx in range(1, indices.size):
        cur = int(indices[out_idx])
        if cur != prev + 1:
            runs.append((run_src_start, prev + 1, run_out_start, out_idx))
            run_src_start = cur
            run_out_start = out_idx
        prev = cur

    runs.append((run_src_start, prev + 1, run_out_start, indices.size))
    return runs
