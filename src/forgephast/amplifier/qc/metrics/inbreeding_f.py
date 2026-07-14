"""Inbreeding-coefficient QC (donor vs synthetic).

This module performs inbreeding-F analysis (summaries, optional plots)
while delegating compute-only kernels (counts, F from counts) to
forgephast._internal.qc_kernels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forgephast._internal.qc_kernels import (
    genotype_counts_u8,
    haplotypes_to_genotypes_u8,
    inbreeding_f_from_genotype_counts,
)

from ...config import RunConfig
from .statistics import (
    histogram_probabilities as _hist_probs,
    maf_from_genotype_counts as _maf_from_genotype_counts,
    root_mean_square_error as _rmse,
)

logger = logging.getLogger(__name__)

_MISSING_SENTINEL = np.uint8(255)


def compute_inbreeding_f_metrics(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    ext_data: Any | None = None,
    ext_data_error: str | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Compute donor-vs-synthetic inbreeding-F agreement."""
    enabled = bool(config.qc.inbreeding_f_enabled)
    payload: dict[str, Any] = {
        "enabled": enabled,
        "chrom": int(chrom),
        "status": "disabled" if not enabled else "init",
    }
    created: list[Path] = []
    if not enabled:
        return payload, created

    if ext_data is None:
        payload["status"] = "no_ext_cache"
        payload["error"] = ext_data_error
        payload["population_summary"] = {}
        payload["gate_pass"] = None
        return payload, created

    pops = list(getattr(ext_data, "pops", []))
    if not pops:
        payload["status"] = "no_populations"
        payload["population_summary"] = {}
        payload["gate_pass"] = None
        return payload, created

    subset_id = str(getattr(ext_data, "variant_subset_id", ""))

    min_maf = float(config.qc.inbreeding_f_min_maf)
    min_n_obs = int(config.qc.inbreeding_f_min_n_obs)
    hist_min = float(config.qc.inbreeding_f_hist_min)
    hist_max = float(config.qc.inbreeding_f_hist_max)
    n_bins = int(config.qc.inbreeding_f_n_bins)

    if hist_max <= hist_min:
        raise ValueError("inbreeding_f_hist_max must be > inbreeding_f_hist_min")

    edges = np.linspace(hist_min, hist_max, n_bins + 1, dtype=np.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    plot_series: dict[str, dict[str, np.ndarray]] = {}

    gating_active = (config.qc.inbreeding_f_hist_rmse_max is not None) or (
        config.qc.inbreeding_f_mean_shift_abs_max is not None
    )
    overall_gate_pass = True

    for pop in pops:
        pop_data = ext_data.per_pop[pop]
        donor_haps = pop_data.donor_haps
        synth_haps = pop_data.synth_haps

        donor_g = haplotypes_to_genotypes_u8(donor_haps, missing_sentinel=int(_MISSING_SENTINEL))
        synth_g = haplotypes_to_genotypes_u8(synth_haps, missing_sentinel=int(_MISSING_SENTINEL))

        d_obs, d_hom_ref, d_het, d_hom_alt = genotype_counts_u8(
            donor_g, missing_sentinel=int(_MISSING_SENTINEL)
        )
        s_obs, s_hom_ref, s_het, s_hom_alt = genotype_counts_u8(
            synth_g, missing_sentinel=int(_MISSING_SENTINEL)
        )

        d_maf = _maf_from_genotype_counts(d_obs, d_het, d_hom_alt)
        s_maf = _maf_from_genotype_counts(s_obs, s_het, s_hom_alt)

        d_f = inbreeding_f_from_genotype_counts(d_obs, d_het, d_hom_alt)
        s_f = inbreeding_f_from_genotype_counts(s_obs, s_het, s_hom_alt)

        valid_d = (
            np.isfinite(d_f)
            & (d_obs.astype(np.int64, copy=False) >= int(min_n_obs))
            & (d_maf >= min_maf)
        )
        valid_s = (
            np.isfinite(s_f)
            & (s_obs.astype(np.int64, copy=False) >= int(min_n_obs))
            & (s_maf >= min_maf)
        )
        valid = valid_d & valid_s
        n_used = int(np.count_nonzero(valid))

        if n_used == 0:
            summary[pop] = {
                "status": "no_variants",
                "n_donor_samples": int(len(pop_data.donor_ids)),
                "n_synth_samples": int(len(pop_data.synth_ids)),
                "n_variants_used": 0,
                "gate_pass": None if not gating_active else True,
            }
            continue

        d_vals_raw = d_f[valid].astype(np.float64, copy=False)
        s_vals_raw = s_f[valid].astype(np.float64, copy=False)
        d_vals = np.clip(d_vals_raw, hist_min, hist_max)
        s_vals = np.clip(s_vals_raw, hist_min, hist_max)

        probs_d, counts_d = _hist_probs(d_vals, edges=edges)
        probs_s, counts_s = _hist_probs(s_vals, edges=edges)
        hist_rmse = float(_rmse(probs_s, probs_d))

        mean_d = float(np.mean(d_vals_raw))
        mean_s = float(np.mean(s_vals_raw))
        med_d = float(np.median(d_vals_raw))
        med_s = float(np.median(s_vals_raw))
        mean_shift = mean_s - mean_d
        median_shift = med_s - med_d

        gate_pass_pop: bool | None
        if not gating_active:
            gate_pass_pop = None
        else:
            gate_pass_pop = True
            if (
                config.qc.inbreeding_f_hist_rmse_max is not None
                and hist_rmse > float(config.qc.inbreeding_f_hist_rmse_max)
            ):
                gate_pass_pop = False
            if (
                config.qc.inbreeding_f_mean_shift_abs_max is not None
                and abs(mean_shift) > float(config.qc.inbreeding_f_mean_shift_abs_max)
            ):
                gate_pass_pop = False
            if gate_pass_pop is False:
                overall_gate_pass = False

        summary[pop] = {
            "status": "ok",
            "n_donor_samples": int(len(pop_data.donor_ids)),
            "n_synth_samples": int(len(pop_data.synth_ids)),
            "n_variants_used": n_used,
            "hist_rmse": hist_rmse,
            "mean_f_donor": mean_d,
            "mean_f_synthetic": mean_s,
            "median_f_donor": med_d,
            "median_f_synthetic": med_s,
            "mean_shift": mean_shift,
            "median_shift": median_shift,
            "gate_pass": gate_pass_pop,
        }

        plot_series[pop] = {
            "donor": d_vals,
            "synthetic": s_vals,
        }

        for source, probs, counts in (
            ("donor", probs_d, counts_d),
            ("synthetic", probs_s, counts_s),
        ):
            for i in range(edges.size - 1):
                rows.append(
                    {
                        "population": pop,
                        "source": source,
                        "chrom": int(chrom),
                        "bin_start": float(edges[i]),
                        "bin_end": float(edges[i + 1]),
                        "bin_mid": float(mids[i]),
                        "prob": float(probs[i]),
                        "count": int(counts[i]),
                    }
                )

    table_path = diagnostics_dir / f"inbreeding_f_hist_chr{chrom}.tsv"
    df = pd.DataFrame.from_records(rows)
    df.to_csv(table_path, sep="\t", index=False)
    created.append(table_path)

    plot_paths: dict[str, str] = {}
    plot_error: str | None = None
    if config.qc.plots_enabled and plot_series:
        try:
            from ..plots import save_hist_overlay_png

            plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
            plot_dir.mkdir(parents=True, exist_ok=True)
            for pop, series in plot_series.items():
                out = plot_dir / f"inbreeding_f_chr{chrom}_{pop}.png"
                save_hist_overlay_png(
                    out,
                    series=series,
                    bins=edges,
                    title=f"Inbreeding F chr{chrom} {pop}",
                    xlabel="F",
                )
                plot_paths[str(pop)] = str(out)
                created.append(out)
        except Exception as exc:
            logger.warning("Inbreeding-F QC plot generation failed; continuing without plots: %s", exc)
            plot_error = str(exc)
            plot_paths = {}

    payload.update(
        {
            "status": "ok",
            "seed": int(config.qc.seed),
            "variant_subset_id": subset_id,
            "inbreeding_f_min_maf": min_maf,
            "inbreeding_f_min_n_obs": min_n_obs,
            "inbreeding_f_hist_min": hist_min,
            "inbreeding_f_hist_max": hist_max,
            "inbreeding_f_n_bins": n_bins,
            "table_path": str(table_path),
            "plot_paths": plot_paths,
            "plot_error": plot_error,
            "population_summary": summary,
            "gate_pass": None if not gating_active else bool(overall_gate_pass),
            "gate_thresholds": {
                "hist_rmse_max": config.qc.inbreeding_f_hist_rmse_max,
                "mean_shift_abs_max": config.qc.inbreeding_f_mean_shift_abs_max,
            }
            if gating_active
            else None,
        }
    )
    logger.info("Inbreeding-F QC complete chrom=%s pops=%s table=%s", chrom, len(pops), table_path)
    return payload, created
