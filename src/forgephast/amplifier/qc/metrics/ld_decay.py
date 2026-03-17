"""LD-decay math and QC helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...config import RunConfig
from ...donors import load_manifest
from ...maps import load_genetic_map
from ...utils import derived_seed, write_json
from ..context import align_haplotype_axes_by_position
from .fst import sample_variant_indices

try:
    from numba import njit, prange, set_num_threads

    HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised when numba is unavailable
    HAS_NUMBA = False
    njit = None
    prange = range
    set_num_threads = None

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


if HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _r2_against_candidates_numba(z: np.ndarray, anchor_idx: int, candidates: np.ndarray) -> np.ndarray:
        n_samples = z.shape[0]
        denom = float(n_samples)
        out = np.empty(candidates.shape[0], dtype=np.float64)
        for k in prange(candidates.shape[0]):
            j = candidates[k]
            s = 0.0
            for i in range(n_samples):
                s += z[i, anchor_idx] * z[i, j]
            r = s / denom
            r2 = r * r
            if r2 < 0.0:
                r2 = 0.0
            elif r2 > 1.0:
                r2 = 1.0
            out[k] = r2
        return out


    @njit(fastmath=True)
    def _append_bucket_values_numba(
        bucket_values: np.ndarray,
        bucket_counts: np.ndarray,
        bins: np.ndarray,
        r2_values: np.ndarray,
        max_pairs_per_bin: int,
    ) -> int:
        newly_filled = 0
        for i in range(bins.shape[0]):
            b = bins[i]
            c = bucket_counts[b]
            if c >= max_pairs_per_bin:
                continue
            bucket_values[b, c] = r2_values[i]
            c_new = c + 1
            bucket_counts[b] = c_new
            if c_new == max_pairs_per_bin:
                newly_filled += 1
        return newly_filled


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
    numba_threads: int | None = None,
) -> pd.DataFrame:
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

    _configure_numba_threads(numba_threads)

    n_samples, n_variants = g.shape
    if n_samples < 3 or n_variants < 2:
        return _empty_ld_curve()

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

    order = np.argsort(pos_bp)
    pos_bp = pos_bp[order]
    z = z[:, order]

    cm = np.interp(pos_bp.astype(np.float64), map_bp_arr, map_cm_arr)
    if cm.shape[0] < 2:
        return _empty_ld_curve()

    n_bins = int(math.ceil(float(max_distance_cm) / float(bin_width_cm)))
    max_pairs = int(max_pairs_per_bin)
    bucket_vals = np.empty((n_bins, max_pairs), dtype=np.float64)
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

        if HAS_NUMBA:
            r2 = _r2_against_candidates_numba(z, int(i), np.asarray(cand, dtype=np.int64))
            filled_bins += int(
                _append_bucket_values_numba(
                    bucket_vals,
                    bucket_counts,
                    np.asarray(bins, dtype=np.int64),
                    np.asarray(r2, dtype=np.float64),
                    max_pairs,
                )
            )
        else:
            r = (z[:, i] @ z[:, cand]) / float(n_samples)
            r2 = np.clip(r * r, 0.0, 1.0)
            for b, value in zip(bins.tolist(), r2.tolist()):
                c = int(bucket_counts[b])
                if c >= max_pairs:
                    continue
                bucket_vals[b, c] = float(value)
                c_new = c + 1
                bucket_counts[b] = c_new
                if c_new == max_pairs:
                    filled_bins += 1

    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        count = int(bucket_counts[b])
        if count == 0:
            continue
        arr = np.asarray(bucket_vals[b, :count], dtype=np.float64)
        start = float(b) * float(bin_width_cm)
        end = (float(b) + 1.0) * float(bin_width_cm)
        rows.append(
            {
                "bin_idx": int(b),
                "bin_start_cm": start,
                "bin_end_cm": end,
                "bin_mid_cm": 0.5 * (start + end),
                "n_pairs": int(count),
                "mean_r2": float(np.mean(arr)),
                "median_r2": float(np.median(arr)),
            }
        )

    if not rows:
        return _empty_ld_curve()
    return pd.DataFrame.from_records(rows).sort_values("bin_idx").reset_index(drop=True)


def ld_decay_metrics_from_curves(
    *,
    donor_curve: pd.DataFrame,
    synthetic_curve: pd.DataFrame,
    min_pairs_per_bin: int,
    half_decay_r2_threshold: float,
) -> dict[str, float | int | str]:
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

    donor_half = _half_decay_distance(
        donor_curve[donor_curve["bin_idx"].isin(used["bin_idx"])],
        threshold_r2=half_decay_r2_threshold,
    )
    synth_half = _half_decay_distance(
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


def _half_decay_distance(curve_df: pd.DataFrame, threshold_r2: float) -> float:
    if curve_df.empty:
        return float("nan")
    curve = curve_df.sort_values("bin_mid_cm").reset_index(drop=True)
    below = curve[curve["median_r2"].to_numpy(dtype=np.float64, copy=False) <= float(threshold_r2)]
    if below.empty:
        return float("nan")
    return float(below.iloc[0]["bin_mid_cm"])


def _empty_ld_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["bin_idx", "bin_start_cm", "bin_end_cm", "bin_mid_cm", "n_pairs", "mean_r2", "median_r2"]
    )


def _haplotypes_to_genotypes(haplotypes: np.ndarray) -> np.ndarray:
    h = np.asarray(haplotypes, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"haplotypes must be 2D, got shape={h.shape}")
    if h.shape[0] % 2 != 0:
        raise ValueError(f"haplotype row count must be even, got shape={h.shape}")
    return h[0::2, :] + h[1::2, :]


def compute_ld_decay_qc_impl(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
    resolve_donor_panel_source_for_population_fn: Callable[[int, str], tuple[str, Path]],
    resolve_map_path_for_population_fn: Callable[[int, str], Path],
    load_sample_haplotypes_from_panel_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    load_synthetic_population_haplotypes_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], Path]:
    from .ld_heatmap import ld_heatmap_binned_from_genotypes
    from ..plots import save_ld_heatmap_pair_png

    manifest = load_manifest(config)
    donors_by_population = manifest.donors_by_population
    pop_names = sorted(groups.keys())
    if not pop_names:
        payload = {
            "enabled": True,
            "chrom": int(chrom),
            "status": "no_populations",
            "gate_pass": None,
            "population_summary": {},
        }
        out_path = diagnostics_dir / f"ld_decay_chr{chrom}.json"
        write_json(out_path, payload)
        return payload, out_path

    table_rows: list[dict[str, Any]] = []
    pop_summary: dict[str, dict[str, Any]] = {}
    n_fail = 0
    n_eval = 0
    max_variant_ids_sampled = 0
    ld_sample_size = int(config.qc.ext_max_diploids_per_population)
    ld_max_variants = int(config.qc.ext_max_variants)

    for pop in pop_names:
        if pop not in donors_by_population:
            pop_summary[pop] = {"status": "missing_donor_population", "pass": None}
            continue

        donor_ids_all = donors_by_population[pop]
        synth_ids_all = groups[pop]
        donor_n = min(len(donor_ids_all), int(ld_sample_size))
        synth_n = min(len(synth_ids_all), int(ld_sample_size))
        donor_ids = sorted(donor_ids_all) if len(donor_ids_all) <= donor_n else sorted(donor_ids_all)
        synth_ids = sorted(synth_ids_all) if len(synth_ids_all) <= synth_n else sorted(synth_ids_all)
        if len(donor_ids) > donor_n:
            rng = np.random.default_rng(int(derived_seed(int(config.qc.seed), "ld_donor", pop, chrom)))
            idx = np.sort(rng.choice(len(donor_ids), size=int(donor_n), replace=False))
            donor_ids = [donor_ids[int(i)] for i in idx.tolist()]
        if len(synth_ids) > synth_n:
            rng = np.random.default_rng(int(derived_seed(int(config.qc.seed), "ld_synth", pop, chrom)))
            idx = np.sort(rng.choice(len(synth_ids), size=int(synth_n), replace=False))
            synth_ids = [synth_ids[int(i)] for i in idx.tolist()]

        donor_source_kind, donor_source_path = resolve_donor_panel_source_for_population_fn(int(chrom), str(pop))
        map_path = resolve_map_path_for_population_fn(int(chrom), str(pop))
        map_bp, map_cm = load_genetic_map(map_path)

        donor_pos, donor_haps = load_sample_haplotypes_from_panel_fn(
            source_path=donor_source_path,
            chrom=int(chrom),
            sample_ids=donor_ids,
            source_kind=donor_source_kind,
        )
        synth_pos, synth_haps = load_synthetic_population_haplotypes_fn(
            inputs=inputs,
            chrom=int(chrom),
            pop=str(pop),
            sample_ids=synth_ids,
        )
        aligned_pos, donor_haps_aligned, synth_haps_aligned = align_haplotype_axes_by_position(
            donor_pos=donor_pos,
            donor_haps=donor_haps,
            synth_pos=synth_pos,
            synth_haps=synth_haps,
        )

        variant_indices = sample_variant_indices(
            n_variants=int(aligned_pos.shape[0]),
            max_variants=int(ld_max_variants),
            seed=int(derived_seed(int(config.qc.seed), "ld_variants", pop, chrom)),
        )
        if variant_indices is None:
            sampled_pos = aligned_pos
            donor_haps_sampled = donor_haps_aligned
            synth_haps_sampled = synth_haps_aligned
        else:
            sampled_pos = aligned_pos[variant_indices]
            donor_haps_sampled = donor_haps_aligned[:, variant_indices]
            synth_haps_sampled = synth_haps_aligned[:, variant_indices]

        max_variant_ids_sampled = max(max_variant_ids_sampled, int(sampled_pos.shape[0]))
        donor_genotypes = _haplotypes_to_genotypes(donor_haps_sampled)
        synth_genotypes = _haplotypes_to_genotypes(synth_haps_sampled)

        max_pairs_per_bin = max(int(config.qc.ld_decay_min_pairs_per_bin) * 10, 2000)
        donor_curve = ld_curve_from_genotypes(
            genotypes=donor_genotypes,
            variant_positions_bp=sampled_pos,
            map_bp=map_bp,
            map_cm=map_cm,
            max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
            bin_width_cm=float(config.qc.ld_decay_bin_width_cm),
            min_maf=float(config.qc.ld_decay_min_maf),
            seed=int(derived_seed(int(config.qc.seed), "ld_curve", "donor", pop, chrom)),
            max_pairs_per_bin=max_pairs_per_bin,
            max_neighbors_per_anchor=128,
            numba_threads=numba_threads,
        )
        synth_curve = ld_curve_from_genotypes(
            genotypes=synth_genotypes,
            variant_positions_bp=sampled_pos,
            map_bp=map_bp,
            map_cm=map_cm,
            max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
            bin_width_cm=float(config.qc.ld_decay_bin_width_cm),
            min_maf=float(config.qc.ld_decay_min_maf),
            seed=int(derived_seed(int(config.qc.seed), "ld_curve", "synthetic", pop, chrom)),
            max_pairs_per_bin=max_pairs_per_bin,
            max_neighbors_per_anchor=128,
            numba_threads=numba_threads,
        )

        for source, curve in (("donor", donor_curve), ("synthetic", synth_curve)):
            if curve.empty:
                continue
            for row in curve.to_dict(orient="records"):
                table_rows.append(
                    {
                        "population": pop,
                        "source": source,
                        "chrom": int(chrom),
                        **row,
                    }
                )

        metrics = ld_decay_metrics_from_curves(
            donor_curve=donor_curve,
            synthetic_curve=synth_curve,
            min_pairs_per_bin=int(config.qc.ld_decay_min_pairs_per_bin),
            half_decay_r2_threshold=float(config.qc.ld_decay_half_decay_r2_threshold),
        )
        pop_pass: bool | None = None
        if str(metrics.get("status")) == "ok":
            rmse_ok = float(metrics["rmse_median_r2"]) <= float(config.qc.ld_decay_rmse_max)
            half_rel = float(metrics["half_decay_rel_diff"])
            half_d = float(metrics["half_decay_cm_donor"])
            half_s = float(metrics["half_decay_cm_synthetic"])
            if np.isfinite(half_rel):
                half_ok = half_rel <= float(config.qc.ld_decay_half_decay_rel_diff_max)
            elif (not np.isfinite(half_d)) and (not np.isfinite(half_s)):
                half_ok = True
            else:
                half_ok = False
            pop_pass = bool(rmse_ok and half_ok)
            n_eval += 1
            if not pop_pass:
                n_fail += 1

        metrics_no_auc = {str(k): v for k, v in metrics.items() if not str(k).startswith("auc_")}
        pop_summary[pop] = {
            "status": str(metrics.get("status")),
            "pass": pop_pass,
            "n_donor_samples": int(len(donor_ids)),
            "n_synthetic_samples": int(len(synth_ids)),
            "n_variants_aligned": int(aligned_pos.shape[0]),
            "n_variants_sampled": int(sampled_pos.shape[0]),
            **metrics_no_auc,
        }

        if bool(config.qc.plots_enabled):
            heatmap_plot: Path | None = None
            try:
                plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
                plot_dir.mkdir(parents=True, exist_ok=True)
                heatmap_plot = plot_dir / f"ld_heatmap_{pop}_chr{chrom}.png"

                n_bins = int(config.qc.ld_heatmap_n_position_bins)
                max_vars = min(int(ld_max_variants), int(config.qc.ext_max_variants))
                max_vars = max(2, int(max_vars))
                stride = (int(aligned_pos.shape[0]) + max_vars - 1) // max_vars
                stride = max(1, int(stride))
                thin_idx = np.arange(0, int(aligned_pos.shape[0]), int(stride), dtype=np.int64)

                pos_thin = aligned_pos[thin_idx]
                donor_haps_thin = donor_haps_aligned[:, thin_idx]
                synth_haps_thin = synth_haps_aligned[:, thin_idx]
                donor_g_thin = _haplotypes_to_genotypes(donor_haps_thin)
                synth_g_thin = _haplotypes_to_genotypes(synth_haps_thin)

                edges_cm = np.linspace(
                    float(np.min(map_cm)),
                    float(np.max(map_cm)),
                    n_bins + 1,
                    dtype=np.float64,
                )

                donor_mean, _, donor_meta = ld_heatmap_binned_from_genotypes(
                    genotypes=donor_g_thin,
                    variant_positions_bp=pos_thin,
                    map_bp=map_bp,
                    map_cm=map_cm,
                    n_position_bins=n_bins,
                    max_variants=int(pos_thin.shape[0]),
                    min_maf=float(config.qc.ld_decay_min_maf),
                    max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
                    max_pairs_total=300000,
                    max_neighbors_per_anchor=256,
                    seed=int(derived_seed(int(config.qc.seed), "ld_heatmap_donor", pop, chrom)),
                    bin_edges_cm=edges_cm,
                )
                synth_mean, _, synth_meta = ld_heatmap_binned_from_genotypes(
                    genotypes=synth_g_thin,
                    variant_positions_bp=pos_thin,
                    map_bp=map_bp,
                    map_cm=map_cm,
                    n_position_bins=n_bins,
                    max_variants=int(pos_thin.shape[0]),
                    min_maf=float(config.qc.ld_decay_min_maf),
                    max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
                    max_pairs_total=300000,
                    max_neighbors_per_anchor=256,
                    seed=int(derived_seed(int(config.qc.seed), "ld_heatmap_synth", pop, chrom)),
                    bin_edges_cm=edges_cm,
                )

                save_ld_heatmap_pair_png(
                    heatmap_plot,
                    donor_mean_r2=donor_mean,
                    synthetic_mean_r2=synth_mean,
                    edges_cm=edges_cm,
                    title=f"LD heatmap chr{chrom} {pop} (mean r²)",
                )
                pop_summary[pop]["heatmap"] = {
                    "status": "ok",
                    "plot_path": str(heatmap_plot),
                    "donor_meta": donor_meta,
                    "synthetic_meta": synth_meta,
                }
            except Exception as exc:
                pop_summary[pop]["heatmap"] = {
                    "status": "error",
                    "error": str(exc),
                    "plot_path": str(heatmap_plot) if heatmap_plot is not None else None,
                }

    table_path = diagnostics_dir / f"ld_decay_chr{chrom}.tsv"
    table_df = pd.DataFrame.from_records(table_rows)
    if table_df.empty:
        table_df = pd.DataFrame(
            columns=[
                "population",
                "source",
                "chrom",
                "bin_idx",
                "bin_start_cm",
                "bin_end_cm",
                "bin_mid_cm",
                "n_pairs",
                "mean_r2",
                "median_r2",
            ]
        )
    table_df.to_csv(table_path, sep="\t", index=False)

    if n_eval == 0:
        gate_pass = None
        status = "no_evaluable_populations"
    else:
        gate_pass = bool(n_fail == 0)
        status = "ok"

    payload = {
        "enabled": True,
        "chrom": int(chrom),
        "status": status,
        "gate_pass": gate_pass,
        "n_populations_total": int(len(pop_names)),
        "n_populations_evaluable": int(n_eval),
        "n_populations_failed": int(n_fail),
        "variant_sampling": {
            "source": "python_matrix",
            "n_variant_ids": int(max_variant_ids_sampled),
            "max_variants": int(ld_max_variants),
            "min_maf": float(config.qc.ld_decay_min_maf),
            "max_distance_cm": float(config.qc.ld_decay_max_distance_cm),
            "bin_width_cm": float(config.qc.ld_decay_bin_width_cm),
            "min_pairs_per_bin": int(config.qc.ld_decay_min_pairs_per_bin),
            "seed": int(config.qc.seed),
        },
        "heatmap": {
            "enabled": bool(config.qc.plots_enabled),
            "n_position_bins": int(config.qc.ld_heatmap_n_position_bins),
            "max_variants": int(config.qc.ext_max_variants),
            "min_maf": float(config.qc.ld_decay_min_maf),
            "max_distance_cm": float(config.qc.ld_decay_max_distance_cm),
            "max_pairs_total": 300000,
            "max_neighbors_per_anchor": 256,
            "seed_base": int(config.qc.seed),
        },
        "thresholds": {
            "rmse_max": float(config.qc.ld_decay_rmse_max),
            "half_decay_rel_diff_max": float(config.qc.ld_decay_half_decay_rel_diff_max),
            "half_decay_r2_threshold": float(config.qc.ld_decay_half_decay_r2_threshold),
        },
        "table_path": str(table_path),
        "population_summary": pop_summary,
    }

    out_path = diagnostics_dir / f"ld_decay_chr{chrom}.json"
    write_json(out_path, payload)
    return payload, out_path
