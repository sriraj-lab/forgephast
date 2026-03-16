"""LD heatmap utilities for QC diagnostics.

This is intended for qualitative visualization: a banded 2D mean-r^2 heatmap
across the chromosome (binned by cM position).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _stride_thin_indices(n: int, max_n: int) -> np.ndarray:
    if max_n <= 0:
        raise ValueError("max_n must be > 0")
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    stride = int(math.ceil(n / float(max_n)))
    stride = max(1, stride)
    return np.arange(0, n, stride, dtype=np.int64)


def ld_heatmap_binned_from_genotypes(
    *,
    genotypes: np.ndarray,
    variant_positions_bp: np.ndarray,
    map_bp: np.ndarray,
    map_cm: np.ndarray,
    n_position_bins: int,
    max_variants: int,
    min_maf: float,
    max_distance_cm: float,
    max_pairs_total: int,
    max_neighbors_per_anchor: int,
    seed: int,
    bin_edges_cm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute a binned mean-r^2 LD heatmap.

    Returns (mean_r2, edges_cm, meta).
    - mean_r2 is (B, B) with NaNs where no pairs were observed.
    - edges_cm is (B+1,) bin edges in cM.
    """
    g0 = np.asarray(genotypes, dtype=np.float64)
    pos_bp0 = np.asarray(variant_positions_bp, dtype=np.int64)
    map_bp_arr = np.asarray(map_bp, dtype=np.float64)
    map_cm_arr = np.asarray(map_cm, dtype=np.float64)

    if g0.ndim != 2:
        raise ValueError(f"genotypes must be 2D, got shape={g0.shape}")
    if pos_bp0.ndim != 1:
        raise ValueError(f"variant_positions_bp must be 1D, got shape={pos_bp0.shape}")
    if g0.shape[1] != pos_bp0.shape[0]:
        raise ValueError(
            "genotype/variant mismatch: "
            f"genotypes.shape[1]={g0.shape[1]} vs len(variant_positions_bp)={pos_bp0.shape[0]}"
        )
    if n_position_bins < 5:
        raise ValueError("n_position_bins must be >= 5")
    if max_variants <= 1:
        raise ValueError("max_variants must be > 1")
    if min_maf <= 0.0 or min_maf >= 0.5:
        raise ValueError("min_maf must be in (0.0, 0.5)")
    if max_distance_cm <= 0.0:
        raise ValueError("max_distance_cm must be > 0")
    if max_pairs_total <= 0:
        raise ValueError("max_pairs_total must be > 0")
    if max_neighbors_per_anchor <= 0:
        raise ValueError("max_neighbors_per_anchor must be > 0")

    n_samples, n_variants0 = g0.shape
    if n_samples < 3 or n_variants0 < 2:
        edges = np.asarray(bin_edges_cm, dtype=np.float64) if bin_edges_cm is not None else np.linspace(0.0, 1.0, n_position_bins + 1)
        mean = np.full((n_position_bins, n_position_bins), np.nan, dtype=np.float64)
        return mean, edges, {"status": "insufficient_data", "n_variants_used": int(n_variants0)}

    thin_idx = _stride_thin_indices(int(n_variants0), int(max_variants))
    g = g0[:, thin_idx]
    pos_bp = pos_bp0[thin_idx]

    missing = np.isnan(g) | (g < 0.0) | (g > 2.0)
    observed = (~missing).astype(np.float64)
    alt_sum = np.where(missing, 0.0, g).sum(axis=0, dtype=np.float64)
    an = 2.0 * observed.sum(axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = alt_sum / an
    maf = np.minimum(p, 1.0 - p)
    keep_maf = np.isfinite(maf) & (maf >= float(min_maf)) & (an >= 4.0)
    if np.count_nonzero(keep_maf) < 2:
        edges = np.asarray(bin_edges_cm, dtype=np.float64) if bin_edges_cm is not None else np.linspace(0.0, 1.0, n_position_bins + 1)
        mean = np.full((n_position_bins, n_position_bins), np.nan, dtype=np.float64)
        return mean, edges, {"status": "no_variants_after_maf", "n_variants_used": 0}

    g = g[:, keep_maf]
    pos_bp = pos_bp[keep_maf]

    g_nan = np.where(missing[:, keep_maf], np.nan, g)
    means = np.nanmean(g_nan, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    if np.any(~np.isfinite(g_nan)):
        g = g.copy()
        miss = ~np.isfinite(g_nan)
        g[miss] = np.take(means, np.where(miss)[1])

    centered = g - means
    std = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64))
    keep_std = std > 1e-12
    if np.count_nonzero(keep_std) < 2:
        edges = np.asarray(bin_edges_cm, dtype=np.float64) if bin_edges_cm is not None else np.linspace(0.0, 1.0, n_position_bins + 1)
        mean = np.full((n_position_bins, n_position_bins), np.nan, dtype=np.float64)
        return mean, edges, {"status": "no_variants_after_std", "n_variants_used": 0}

    centered = centered[:, keep_std]
    std = std[keep_std]
    pos_bp = pos_bp[keep_std]
    z = centered / std

    order = np.argsort(pos_bp)
    pos_bp = pos_bp[order]
    z = z[:, order]

    cm = np.interp(pos_bp.astype(np.float64), map_bp_arr, map_cm_arr)
    if cm.shape[0] < 2:
        edges = np.asarray(bin_edges_cm, dtype=np.float64) if bin_edges_cm is not None else np.linspace(0.0, 1.0, n_position_bins + 1)
        mean = np.full((n_position_bins, n_position_bins), np.nan, dtype=np.float64)
        return mean, edges, {"status": "no_cm_positions", "n_variants_used": 0}

    if bin_edges_cm is None:
        cm_min = float(np.min(cm))
        cm_max = float(np.max(cm))
        if not np.isfinite(cm_min) or not np.isfinite(cm_max) or cm_max <= cm_min:
            raise ValueError("Unable to determine cM range for heatmap binning")
        edges = np.linspace(cm_min, cm_max, int(n_position_bins) + 1, dtype=np.float64)
    else:
        edges = np.asarray(bin_edges_cm, dtype=np.float64)
        if edges.ndim != 1 or edges.shape[0] != int(n_position_bins) + 1:
            raise ValueError("bin_edges_cm must have shape (n_position_bins+1,)")
        if not np.all(np.isfinite(edges)):
            raise ValueError("bin_edges_cm must be finite")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges_cm must be strictly increasing")

    pos_bin = np.searchsorted(edges, cm, side="right") - 1
    pos_bin = np.clip(pos_bin, 0, int(n_position_bins) - 1).astype(np.int64, copy=False)

    upper = np.searchsorted(cm, cm + float(max_distance_cm), side="right")
    rng = np.random.default_rng(int(seed))
    anchor_order = rng.permutation(max(0, z.shape[1] - 1))

    sums = np.zeros((int(n_position_bins), int(n_position_bins)), dtype=np.float64)
    counts = np.zeros((int(n_position_bins), int(n_position_bins)), dtype=np.int64)
    used_pairs = 0

    denom = float(n_samples)
    for i in anchor_order.tolist():
        if used_pairs >= int(max_pairs_total):
            break
        j_end = int(upper[i])
        if j_end <= i + 1:
            continue

        cand = np.arange(i + 1, j_end, dtype=np.int64)
        if cand.size > int(max_neighbors_per_anchor):
            choice = np.sort(
                rng.choice(cand.size, size=int(max_neighbors_per_anchor), replace=False).astype(np.int64)
            )
            cand = cand[choice]
        if cand.size == 0:
            continue

        remaining = int(max_pairs_total) - int(used_pairs)
        if cand.size > remaining:
            cand = cand[:remaining]
        if cand.size == 0:
            break

        r = (z[:, i] @ z[:, cand]) / denom
        r2 = np.clip(r * r, 0.0, 1.0)

        bi = int(pos_bin[i])
        bj = pos_bin[cand]
        bi_arr = np.full(bj.shape[0], bi, dtype=np.int64)

        np.add.at(sums, (bi_arr, bj), r2)
        np.add.at(counts, (bi_arr, bj), 1)
        np.add.at(sums, (bj, bi_arr), r2)
        np.add.at(counts, (bj, bi_arr), 1)

        used_pairs += int(cand.size)

    mean = np.full_like(sums, np.nan, dtype=np.float64)
    mask = counts > 0
    mean[mask] = sums[mask] / counts[mask]

    meta: dict[str, Any] = {
        "status": "ok",
        "n_samples": int(n_samples),
        "n_variants_input": int(n_variants0),
        "n_variants_thinned": int(pos_bp0[thin_idx].shape[0]),
        "n_variants_used": int(cm.shape[0]),
        "n_position_bins": int(n_position_bins),
        "min_maf": float(min_maf),
        "max_distance_cm": float(max_distance_cm),
        "max_pairs_total": int(max_pairs_total),
        "max_neighbors_per_anchor": int(max_neighbors_per_anchor),
        "seed": int(seed),
        "used_pairs": int(used_pairs),
    }
    return mean, edges, meta

