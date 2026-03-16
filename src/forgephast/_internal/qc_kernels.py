"""Computation-only kernels used by population genetics QC.

This module intentionally excludes:
- statistical tests/p-values
- divergence metrics (KS/JS/etc.)
- plotting helpers

It is meant to provide reusable, deterministic array computations that callers
can assemble into higher-level QC analyses.
"""

from __future__ import annotations

import numpy as np


def haplotypes_to_genotypes_u8(
    haplotypes: np.ndarray,
    *,
    missing_sentinel: int = 255,
) -> np.ndarray:
    """Convert HapStore-style haplotypes (2N, V) -> genotypes (N, V) with missing propagation.

    HapStore convention: uint8 haplotypes with values {0, 1, 255} where 255 denotes missing.
    Output uses uint8 genotypes with values {0, 1, 2, 255}.
    """
    h = np.asarray(haplotypes, dtype=np.uint8)
    if h.ndim != 2:
        raise ValueError(f"haplotypes must be 2D, got shape={h.shape}")
    if h.shape[0] % 2 != 0:
        raise ValueError(f"haplotype row count must be even, got shape={h.shape}")

    missing = np.uint8(int(missing_sentinel))
    first = h[0::2, :]
    second = h[1::2, :]
    out = (first + second).astype(np.uint8, copy=False)
    miss_mask = (first == missing) | (second == missing)
    if np.any(miss_mask):
        out = out.copy()
        out[miss_mask] = missing
    return out


def haplotypes_to_genotypes_f32(
    haplotypes: np.ndarray,
    *,
    missing_sentinel: int = 255,
) -> np.ndarray:
    """Convert haplotypes (2N, V) -> float32 genotype matrix (N, V) with NaN missing."""
    g_u8 = haplotypes_to_genotypes_u8(haplotypes, missing_sentinel=missing_sentinel)
    g = g_u8.astype(np.float32, copy=False)
    missing = np.uint8(int(missing_sentinel))
    if np.any(g_u8 == missing):
        g = g.copy()
        g[g_u8 == missing] = np.nan
    return g


def genotype_counts_u8(
    genotypes_u8: np.ndarray,
    *,
    missing_sentinel: int = 255,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-locus genotype counts from a uint8 genotype matrix.

    Parameters
    ----------
    genotypes_u8
        Shape (N, V) with values {0, 1, 2, missing_sentinel}.

    Returns
    -------
    n_obs, n_hom_ref, n_het, n_hom_alt
        Each is shape (V,) int64.
    """
    g = np.asarray(genotypes_u8, dtype=np.uint8)
    if g.ndim != 2:
        raise ValueError(f"genotypes_u8 must be 2D, got shape={g.shape}")

    missing = np.uint8(int(missing_sentinel))
    observed = g != missing
    n_obs = np.sum(observed, axis=0, dtype=np.int64)
    n_hom_ref = np.sum(g == np.uint8(0), axis=0, dtype=np.int64)
    n_het = np.sum(g == np.uint8(1), axis=0, dtype=np.int64)
    n_hom_alt = np.sum(g == np.uint8(2), axis=0, dtype=np.int64)
    return n_obs, n_hom_ref, n_het, n_hom_alt


def maf_from_haplotypes(
    haplotypes_u8: np.ndarray,
    *,
    missing_sentinel: int = 255,
    min_an: int = 4,
) -> np.ndarray:
    """Compute per-locus MAF from HapStore-style haplotypes, preserving indexing.

    Returns
    -------
    maf
        Shape (V,) float64, with NaN for loci with AN < min_an or invalid.
    """
    h = np.asarray(haplotypes_u8, dtype=np.uint8)
    if h.ndim != 2:
        raise ValueError(f"haplotypes_u8 must be 2D, got shape={h.shape}")

    missing = np.uint8(int(missing_sentinel))
    ac = np.sum(h == np.uint8(1), axis=0, dtype=np.float64)
    an = np.sum(h != missing, axis=0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = ac / an
    maf = np.minimum(p, 1.0 - p)
    maf = np.clip(maf, 0.0, 0.5).astype(np.float64, copy=False)
    invalid = (~np.isfinite(maf)) | (~np.isfinite(an)) | (an < float(min_an))
    if np.any(invalid):
        maf = maf.copy()
        maf[invalid] = np.nan
    return maf


def hwe_chisq_stat_from_genotype_counts(
    n_hom_ref: np.ndarray,
    n_het: np.ndarray,
    n_hom_alt: np.ndarray,
) -> np.ndarray:
    """Compute per-locus HWE chi-square statistic from genotype counts.

    Notes
    -----
    - This returns the chi-square statistic only (no p-values).
    - Loci with n_obs==0 or with allele frequency p in {0,1} return NaN.
    """
    hom_ref = np.asarray(n_hom_ref, dtype=np.float64)
    het = np.asarray(n_het, dtype=np.float64)
    hom_alt = np.asarray(n_hom_alt, dtype=np.float64)

    if hom_ref.shape != het.shape or het.shape != hom_alt.shape:
        raise ValueError("count arrays must have identical shapes")

    n_obs = hom_ref + het + hom_alt
    out = np.full(n_obs.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(n_obs) & (n_obs > 0.0)
    if not np.any(valid):
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        p = (2.0 * hom_alt + het) / (2.0 * n_obs)
    valid = valid & np.isfinite(p) & (p > 0.0) & (p < 1.0)
    if not np.any(valid):
        return out

    q = 1.0 - p
    exp_ref = n_obs * q * q
    exp_het = n_obs * 2.0 * p * q
    exp_alt = n_obs * p * p
    valid = valid & (exp_ref > 0.0) & (exp_het > 0.0) & (exp_alt > 0.0)
    if not np.any(valid):
        return out

    diff_ref = hom_ref[valid] - exp_ref[valid]
    diff_het = het[valid] - exp_het[valid]
    diff_alt = hom_alt[valid] - exp_alt[valid]
    out[valid] = (
        (diff_ref * diff_ref) / exp_ref[valid]
        + (diff_het * diff_het) / exp_het[valid]
        + (diff_alt * diff_alt) / exp_alt[valid]
    )
    return out


def inbreeding_f_from_genotype_counts(
    n_obs: np.ndarray,
    n_het: np.ndarray,
    n_hom_alt: np.ndarray,
    *,
    min_expected_het: float = 1e-6,
) -> np.ndarray:
    """Compute per-locus inbreeding coefficient F from genotype counts.

    F = 1 - (observed_het / expected_het), where expected het count is computed
    from p estimated in the same cohort.
    """
    obs = np.asarray(n_obs, dtype=np.float64)
    het = np.asarray(n_het, dtype=np.float64)
    hom_alt = np.asarray(n_hom_alt, dtype=np.float64)
    if obs.shape != het.shape or het.shape != hom_alt.shape:
        raise ValueError("count arrays must have identical shapes")

    out = np.full(obs.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(obs) & (obs > 0.0) & np.isfinite(het) & np.isfinite(hom_alt)
    if not np.any(valid):
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        p = (2.0 * hom_alt + het) / (2.0 * obs)
    valid = valid & np.isfinite(p)
    if not np.any(valid):
        return out

    exp_het = obs * 2.0 * p * (1.0 - p)
    valid = valid & np.isfinite(exp_het) & (exp_het > float(min_expected_het))
    if not np.any(valid):
        return out

    out[valid] = 1.0 - (het[valid] / exp_het[valid])
    return out


try:
    from numba import njit, prange, set_num_threads  # type: ignore

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False
    njit = None
    prange = range
    set_num_threads = None


if HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _king_within_pairs_numba(
        g: np.ndarray,
        pairs_i: np.ndarray,
        pairs_j: np.ndarray,
        missing_sentinel: int,
    ) -> np.ndarray:
        n_pairs = pairs_i.shape[0]
        n_variants = g.shape[1]
        out = np.empty(n_pairs, dtype=np.float64)
        for p in prange(n_pairs):
            i = pairs_i[p]
            j = pairs_j[p]
            hethet = 0.0
            ibs0 = 0.0
            het1 = 0.0
            het2 = 0.0
            for v in range(n_variants):
                a = g[i, v]
                b = g[j, v]
                if a == missing_sentinel or b == missing_sentinel:
                    continue
                if a == 1:
                    het1 += 1.0
                if b == 1:
                    het2 += 1.0
                if a == 1 and b == 1:
                    hethet += 1.0
                if (a == 0 and b == 2) or (a == 2 and b == 0):
                    ibs0 += 1.0
            hmin = het1 if het1 < het2 else het2
            hmax = het2 if het1 < het2 else het1
            if hmin <= 0.0:
                out[p] = np.nan
            else:
                out[p] = (2.0 * hethet - 4.0 * ibs0 + hmin - hmax) / (4.0 * hmin)
        return out


    @njit(parallel=True, fastmath=True)
    def _king_cross_pairs_numba(
        g_a: np.ndarray,
        g_b: np.ndarray,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        missing_sentinel: int,
    ) -> np.ndarray:
        n_pairs = idx_a.shape[0]
        n_variants = g_a.shape[1]
        out = np.empty(n_pairs, dtype=np.float64)
        for p in prange(n_pairs):
            i = idx_a[p]
            j = idx_b[p]
            hethet = 0.0
            ibs0 = 0.0
            het1 = 0.0
            het2 = 0.0
            for v in range(n_variants):
                a = g_a[i, v]
                b = g_b[j, v]
                if a == missing_sentinel or b == missing_sentinel:
                    continue
                if a == 1:
                    het1 += 1.0
                if b == 1:
                    het2 += 1.0
                if a == 1 and b == 1:
                    hethet += 1.0
                if (a == 0 and b == 2) or (a == 2 and b == 0):
                    ibs0 += 1.0
            hmin = het1 if het1 < het2 else het2
            hmax = het2 if het1 < het2 else het1
            if hmin <= 0.0:
                out[p] = np.nan
            else:
                out[p] = (2.0 * hethet - 4.0 * ibs0 + hmin - hmax) / (4.0 * hmin)
        return out


def king_within_pairs(
    genotypes: np.ndarray,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    *,
    missing_sentinel: int = 255,
    numba_threads: int | None = None,
) -> np.ndarray:
    """Compute KING-robust kinship for within-cohort pairs (compute kernel)."""
    g = np.asarray(genotypes, dtype=np.uint8)
    i = np.asarray(pairs_i, dtype=np.int64)
    j = np.asarray(pairs_j, dtype=np.int64)
    if i.size == 0:
        return np.zeros(0, dtype=np.float64)

    if HAS_NUMBA and set_num_threads is not None and numba_threads is not None:
        set_num_threads(max(1, int(numba_threads)))
    if HAS_NUMBA:
        return _king_within_pairs_numba(g, i, j, int(missing_sentinel))

    missing = np.uint8(int(missing_sentinel))
    out = np.empty(i.size, dtype=np.float64)
    for p, (a_idx, b_idx) in enumerate(zip(i.tolist(), j.tolist())):
        a = g[int(a_idx), :]
        b = g[int(b_idx), :]
        mask = (a != missing) & (b != missing)
        if not np.any(mask):
            out[p] = np.nan
            continue
        a = a[mask]
        b = b[mask]
        het1 = float(np.sum(a == 1))
        het2 = float(np.sum(b == 1))
        hmin = float(min(het1, het2))
        hmax = float(max(het1, het2))
        if hmin <= 0.0:
            out[p] = np.nan
            continue
        hethet = float(np.sum((a == 1) & (b == 1)))
        ibs0 = float(np.sum(((a == 0) & (b == 2)) | ((a == 2) & (b == 0))))
        out[p] = (2.0 * hethet - 4.0 * ibs0 + hmin - hmax) / (4.0 * hmin)
    return out


def king_cross_pairs(
    genotypes_a: np.ndarray,
    genotypes_b: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    missing_sentinel: int = 255,
    numba_threads: int | None = None,
) -> np.ndarray:
    """Compute KING-robust kinship for cross-cohort pairs (compute kernel)."""
    ga = np.asarray(genotypes_a, dtype=np.uint8)
    gb = np.asarray(genotypes_b, dtype=np.uint8)
    i = np.asarray(idx_a, dtype=np.int64)
    j = np.asarray(idx_b, dtype=np.int64)
    if i.size == 0:
        return np.zeros(0, dtype=np.float64)

    if HAS_NUMBA and set_num_threads is not None and numba_threads is not None:
        set_num_threads(max(1, int(numba_threads)))
    if HAS_NUMBA:
        return _king_cross_pairs_numba(ga, gb, i, j, int(missing_sentinel))

    missing = np.uint8(int(missing_sentinel))
    out = np.empty(i.size, dtype=np.float64)
    for p, (a_idx, b_idx) in enumerate(zip(i.tolist(), j.tolist())):
        a = ga[int(a_idx), :]
        b = gb[int(b_idx), :]
        mask = (a != missing) & (b != missing)
        if not np.any(mask):
            out[p] = np.nan
            continue
        a = a[mask]
        b = b[mask]
        het1 = float(np.sum(a == 1))
        het2 = float(np.sum(b == 1))
        hmin = float(min(het1, het2))
        hmax = float(max(het1, het2))
        if hmin <= 0.0:
            out[p] = np.nan
            continue
        hethet = float(np.sum((a == 1) & (b == 1)))
        ibs0 = float(np.sum(((a == 0) & (b == 2)) | ((a == 2) & (b == 0))))
        out[p] = (2.0 * hethet - 4.0 * ibs0 + hmin - hmax) / (4.0 * hmin)
    return out


if HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _roh_metrics_numba(
        g: np.ndarray,
        cm: np.ndarray,
        missing_sentinel: int,
        min_segment_cm: float,
        max_gap_cm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_variants = g.shape
        total_cm = np.zeros(n_samples, dtype=np.float64)
        n_segments = np.zeros(n_samples, dtype=np.int64)
        max_seg = np.zeros(n_samples, dtype=np.float64)

        for i in prange(n_samples):
            in_seg = False
            seg_start_cm = 0.0
            last_cm = 0.0
            nseg = 0
            tot = 0.0
            maxs = 0.0

            for j in range(n_variants):
                gt = g[i, j]
                cur_cm = cm[j]

                if in_seg and j > 0 and (cur_cm - cm[j - 1]) > max_gap_cm:
                    seg_len = cm[j - 1] - seg_start_cm
                    if seg_len >= min_segment_cm:
                        tot += seg_len
                        nseg += 1
                        if seg_len > maxs:
                            maxs = seg_len
                    seg_start_cm = cur_cm
                    last_cm = cur_cm
                    continue

                if gt == missing_sentinel or gt == 1:
                    if in_seg:
                        seg_len = last_cm - seg_start_cm
                        if seg_len >= min_segment_cm:
                            tot += seg_len
                            nseg += 1
                            if seg_len > maxs:
                                maxs = seg_len
                        in_seg = False
                    last_cm = cur_cm
                    continue

                if not in_seg:
                    in_seg = True
                    seg_start_cm = cur_cm
                last_cm = cur_cm

            if in_seg:
                seg_len = last_cm - seg_start_cm
                if seg_len >= min_segment_cm:
                    tot += seg_len
                    nseg += 1
                    if seg_len > maxs:
                        maxs = seg_len

            total_cm[i] = tot
            n_segments[i] = nseg
            max_seg[i] = maxs

        return total_cm, n_segments, max_seg


def roh_metrics(
    genotypes: np.ndarray,
    cm_positions: np.ndarray,
    *,
    missing_sentinel: int = 255,
    min_segment_cm: float,
    max_gap_cm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-sample ROH totals and segments (compute kernel)."""
    g = np.asarray(genotypes, dtype=np.uint8)
    cm = np.asarray(cm_positions, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"genotypes must be 2D, got shape={g.shape}")
    if cm.ndim != 1 or cm.shape[0] != g.shape[1]:
        raise ValueError("cm_positions must be 1D and match genotype variant axis")

    if HAS_NUMBA:
        return _roh_metrics_numba(
            g,
            cm,
            int(missing_sentinel),
            float(min_segment_cm),
            float(max_gap_cm),
        )

    n_samples, n_variants = g.shape
    total_cm = np.zeros(n_samples, dtype=np.float64)
    n_segments = np.zeros(n_samples, dtype=np.int64)
    max_seg = np.zeros(n_samples, dtype=np.float64)

    missing = int(np.uint8(int(missing_sentinel)))
    for i in range(n_samples):
        in_seg = False
        seg_start = 0.0
        last_cm = 0.0
        nseg = 0
        tot = 0.0
        maxs = 0.0
        for j in range(n_variants):
            cur_cm = float(cm[j])
            if in_seg and j > 0 and (cur_cm - float(cm[j - 1])) > float(max_gap_cm):
                seg_len = float(cm[j - 1]) - seg_start
                if seg_len >= float(min_segment_cm):
                    tot += seg_len
                    nseg += 1
                    maxs = max(maxs, seg_len)
                seg_start = cur_cm
                last_cm = cur_cm
                continue

            gt = int(g[i, j])
            if gt == missing or gt == 1:
                if in_seg:
                    seg_len = last_cm - seg_start
                    if seg_len >= float(min_segment_cm):
                        tot += seg_len
                        nseg += 1
                        maxs = max(maxs, seg_len)
                    in_seg = False
                last_cm = cur_cm
                continue

            if not in_seg:
                in_seg = True
                seg_start = cur_cm
            last_cm = cur_cm

        if in_seg:
            seg_len = last_cm - seg_start
            if seg_len >= float(min_segment_cm):
                tot += seg_len
                nseg += 1
                maxs = max(maxs, seg_len)

        total_cm[i] = tot
        n_segments[i] = nseg
        max_seg[i] = maxs

    return total_cm, n_segments, max_seg


__all__ = [
    "haplotypes_to_genotypes_u8",
    "haplotypes_to_genotypes_f32",
    "genotype_counts_u8",
    "maf_from_haplotypes",
    "hwe_chisq_stat_from_genotype_counts",
    "inbreeding_f_from_genotype_counts",
    "king_within_pairs",
    "king_cross_pairs",
    "roh_metrics",
    "HAS_NUMBA",
]
