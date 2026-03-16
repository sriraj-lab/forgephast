"""Reusable population-genetics QC primitives."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def hudson_fst_from_counts(
    ac_a: np.ndarray,
    an_a: np.ndarray,
    ac_b: np.ndarray,
    an_b: np.ndarray,
) -> float:
    """Compute genome-wide Hudson FST from per-variant allele counts."""
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

    # PLINK's Hudson implementation uses chromosome counts (AN), not diploid sample counts.
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


def ld_curve_from_genotypes(
    *,
    genotypes: np.ndarray,
    variant_positions_bp: np.ndarray,
    map_bp: np.ndarray,
    map_cm: np.ndarray,
    max_distance_cm: float,
    bin_width_cm: float,
    min_maf: float,
    seed: int,
    max_pairs_per_bin: int,
    max_neighbors_per_anchor: int = 128,
) -> pd.DataFrame:
    """Estimate LD-decay curve from a genotype matrix using sampled variant pairs.

    The algorithm samples nearby variant pairs up to `max_pairs_per_bin` per cM bin,
    computes r^2 from standardized genotypes, and reports per-bin mean/median r^2.
    """
    g = np.asarray(genotypes, dtype=np.float64)
    pos_bp = np.asarray(variant_positions_bp, dtype=np.int64)
    map_bp_arr = np.asarray(map_bp, dtype=np.float64)
    map_cm_arr = np.asarray(map_cm, dtype=np.float64)

    if g.ndim != 2:
        raise ValueError(f"genotypes must be 2D, got shape={g.shape}")
    if pos_bp.ndim != 1:
        raise ValueError(f"variant_positions_bp must be 1D, got shape={pos_bp.shape}")
    if g.shape[1] != pos_bp.shape[0]:
        raise ValueError(
            "genotype/variant mismatch: "
            f"genotypes.shape[1]={g.shape[1]} vs len(variant_positions_bp)={pos_bp.shape[0]}"
        )
    if map_bp_arr.ndim != 1 or map_cm_arr.ndim != 1 or map_bp_arr.shape[0] != map_cm_arr.shape[0]:
        raise ValueError(
            "map_bp/map_cm must be 1D arrays with identical length, got "
            f"{map_bp_arr.shape} and {map_cm_arr.shape}"
        )
    if max_distance_cm <= 0.0:
        raise ValueError("max_distance_cm must be > 0")
    if bin_width_cm <= 0.0:
        raise ValueError("bin_width_cm must be > 0")
    if bin_width_cm >= max_distance_cm:
        raise ValueError("bin_width_cm must be < max_distance_cm")
    if min_maf <= 0.0 or min_maf >= 0.5:
        raise ValueError("min_maf must be in (0.0, 0.5)")
    if max_pairs_per_bin <= 0:
        raise ValueError("max_pairs_per_bin must be > 0")
    if max_neighbors_per_anchor <= 0:
        raise ValueError("max_neighbors_per_anchor must be > 0")

    n_samples, n_variants = g.shape
    if n_samples < 3 or n_variants < 2:
        return _empty_ld_curve()

    # Interpret negative values as missing.
    missing = np.isnan(g) | (g < 0.0)
    observed = (~missing).astype(np.float64)
    alt_sum = np.where(missing, 0.0, g).sum(axis=0, dtype=np.float64)
    an = 2.0 * observed.sum(axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = alt_sum / an
    maf = np.minimum(p, 1.0 - p)
    keep_maf = np.isfinite(maf) & (maf >= float(min_maf)) & (an >= 4.0)
    if not np.any(keep_maf):
        return _empty_ld_curve()

    g = g[:, keep_maf]
    pos_bp = pos_bp[keep_maf]

    # Impute missing to column mean before z-scoring.
    means = np.nanmean(np.where(g < 0.0, np.nan, g), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    missing2 = np.isnan(g) | (g < 0.0)
    if np.any(missing2):
        g = g.copy()
        g[missing2] = np.take(means, np.where(missing2)[1])

    centered = g - means
    std = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64))
    keep_std = std > 1e-12
    if np.count_nonzero(keep_std) < 2:
        return _empty_ld_curve()

    centered = centered[:, keep_std]
    std = std[keep_std]
    pos_bp = pos_bp[keep_std]
    z = centered / std

    # Ensure monotonic variant axis for distance lookups.
    order = np.argsort(pos_bp)
    pos_bp = pos_bp[order]
    z = z[:, order]

    cm = np.interp(pos_bp.astype(np.float64), map_bp_arr, map_cm_arr)
    if cm.shape[0] < 2:
        return _empty_ld_curve()

    n_bins = int(math.ceil(float(max_distance_cm) / float(bin_width_cm)))
    bucket_vals: list[list[float]] = [[] for _ in range(n_bins)]
    bucket_counts = np.zeros(n_bins, dtype=np.int64)
    filled_bins = 0

    upper = np.searchsorted(cm, cm + float(max_distance_cm), side="right")
    rng = np.random.default_rng(int(seed))
    anchor_order = rng.permutation(max(0, z.shape[1] - 1))

    for i in anchor_order.tolist():
        if filled_bins == n_bins:
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

        dist = cm[cand] - cm[i]
        bins = np.floor(dist / float(bin_width_cm)).astype(np.int64)
        keep = (bins >= 0) & (bins < n_bins)
        if not np.any(keep):
            continue
        cand = cand[keep]
        bins = bins[keep]
        if cand.size == 0:
            continue

        not_full = bucket_counts[bins] < int(max_pairs_per_bin)
        if not np.any(not_full):
            continue
        cand = cand[not_full]
        bins = bins[not_full]
        if cand.size == 0:
            continue

        r = (z[:, i] @ z[:, cand]) / float(n_samples)
        r2 = np.clip(r * r, 0.0, 1.0)

        for b, value in zip(bins.tolist(), r2.tolist()):
            if bucket_counts[b] >= int(max_pairs_per_bin):
                continue
            bucket_vals[b].append(float(value))
            bucket_counts[b] += 1
            if bucket_counts[b] == int(max_pairs_per_bin):
                filled_bins += 1

    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        vals = bucket_vals[b]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        start = float(b) * float(bin_width_cm)
        end = (float(b) + 1.0) * float(bin_width_cm)
        rows.append(
            {
                "bin_idx": int(b),
                "bin_start_cm": start,
                "bin_end_cm": end,
                "bin_mid_cm": 0.5 * (start + end),
                "n_pairs": int(arr.size),
                "mean_r2": float(np.mean(arr)),
                "median_r2": float(np.median(arr)),
            }
        )

    if not rows:
        return _empty_ld_curve()
    return pd.DataFrame.from_records(rows).sort_values("bin_idx").reset_index(drop=True)


def half_decay_distance(curve_df: pd.DataFrame, threshold_r2: float) -> float:
    """Return first distance (cM) where median r^2 falls below threshold."""
    if curve_df.empty:
        return float("nan")
    curve = curve_df.sort_values("bin_mid_cm").reset_index(drop=True)
    below = curve[curve["median_r2"].to_numpy(dtype=np.float64, copy=False) <= float(threshold_r2)]
    if below.empty:
        return float("nan")
    return float(below.iloc[0]["bin_mid_cm"])


def ld_decay_metrics_from_curves(
    *,
    donor_curve: pd.DataFrame,
    synthetic_curve: pd.DataFrame,
    min_pairs_per_bin: int,
    half_decay_r2_threshold: float,
) -> dict[str, float | int | str]:
    """Compute donor-vs-synthetic LD decay agreement metrics."""
    merged = donor_curve.merge(
        synthetic_curve,
        on="bin_idx",
        suffixes=("_donor", "_synthetic"),
        how="inner",
    )
    if merged.empty:
        return {
            "status": "no_overlapping_bins",
            "n_bins_used": 0,
        }

    used = merged[
        (merged["n_pairs_donor"].astype(int) >= int(min_pairs_per_bin))
        & (merged["n_pairs_synthetic"].astype(int) >= int(min_pairs_per_bin))
    ].copy()
    if used.empty:
        return {
            "status": "insufficient_pairs_per_bin",
            "n_bins_used": 0,
        }

    weights = np.minimum(
        used["n_pairs_donor"].to_numpy(dtype=np.float64, copy=False),
        used["n_pairs_synthetic"].to_numpy(dtype=np.float64, copy=False),
    )
    y_donor = used["median_r2_donor"].to_numpy(dtype=np.float64, copy=False)
    y_synth = used["median_r2_synthetic"].to_numpy(dtype=np.float64, copy=False)
    x = used["bin_mid_cm_donor"].to_numpy(dtype=np.float64, copy=False)
    diff = y_synth - y_donor

    rmse = float(np.sqrt(np.sum(weights * diff * diff) / np.sum(weights)))
    auc_donor = float(np.trapezoid(y_donor, x=x))
    auc_synth = float(np.trapezoid(y_synth, x=x))
    if abs(auc_donor) > 1e-12:
        auc_rel_diff = float(abs(auc_synth - auc_donor) / abs(auc_donor))
    else:
        auc_rel_diff = float(abs(auc_synth - auc_donor))

    donor_half = half_decay_distance(
        donor_curve[donor_curve["bin_idx"].isin(used["bin_idx"])],
        threshold_r2=half_decay_r2_threshold,
    )
    synth_half = half_decay_distance(
        synthetic_curve[synthetic_curve["bin_idx"].isin(used["bin_idx"])],
        threshold_r2=half_decay_r2_threshold,
    )

    if np.isfinite(donor_half) and np.isfinite(synth_half):
        if abs(donor_half) > 1e-12:
            half_rel_diff = float(abs(synth_half - donor_half) / abs(donor_half))
        else:
            half_rel_diff = float(abs(synth_half - donor_half))
    else:
        half_rel_diff = float("nan")

    return {
        "status": "ok",
        "n_bins_used": int(len(used)),
        "rmse_median_r2": rmse,
        "auc_donor": auc_donor,
        "auc_synthetic": auc_synth,
        "auc_rel_diff": auc_rel_diff,
        "half_decay_cm_donor": float(donor_half),
        "half_decay_cm_synthetic": float(synth_half),
        "half_decay_rel_diff": float(half_rel_diff),
    }


def _empty_ld_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["bin_idx", "bin_start_cm", "bin_end_cm", "bin_mid_cm", "n_pairs", "mean_r2", "median_r2"]
    )
