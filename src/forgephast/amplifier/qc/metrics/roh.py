"""Runs of homozygosity QC (donor vs synthetic)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import RunConfig
from ...maps import interpolate_cm, load_genetic_map
from ..io import haplotypes_to_genotypes_u8

from forgephast._internal.qc_kernels import roh_metrics as _utils_roh_metrics

logger = logging.getLogger(__name__)

_MISSING_SENTINEL = np.uint8(255)

try:
    from numba import njit, prange

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False
    njit = None
    prange = range


def _stride_thin_indices(n: int, max_n: int) -> np.ndarray:
    if max_n <= 0:
        raise ValueError("max_n must be > 0")
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    stride = int(math.ceil(n / float(max_n)))
    stride = max(1, stride)
    return np.arange(0, n, stride, dtype=np.int64)


def _summarize(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


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

                # Variant-axis gap break: if we are currently in a segment, and the
                # distance from the previous variant is too large, close the segment
                # at the previous variant.
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

                # Missing/het breaks ROH.
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

                # Homozygous
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


def _roh_metrics(
    genotypes: np.ndarray,
    cm_positions: np.ndarray,
    *,
    min_segment_cm: float,
    max_gap_cm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _utils_roh_metrics(
        genotypes,
        cm_positions,
        missing_sentinel=int(_MISSING_SENTINEL),
        min_segment_cm=float(min_segment_cm),
        max_gap_cm=float(max_gap_cm),
    )


def compute_roh_metrics(
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
    """Compute ROH summary stats for donors and synthetic samples."""
    from ..pipeline import _resolve_map_path_for_population  # local import avoids heavy QC module import unless needed
    from ..plots import save_roh_total_cm_png

    enabled = bool(config.qc.roh_enabled)
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
        return payload, created

    pop_names = list(getattr(ext_data, "pops", []))
    if not pop_names:
        payload["status"] = "no_populations"
        payload["population_summary"] = {}
        return payload, created

    sampled_pos = np.asarray(getattr(ext_data, "sampled_pos"), dtype=np.int64)
    subset_id = str(getattr(ext_data, "variant_subset_id", ""))

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    plot_series: dict[str, dict[str, np.ndarray]] = {}

    for pop in pop_names:
        pop_data = ext_data.per_pop[pop]
        donor_ids_all = pop_data.donor_ids
        synth_ids_all = pop_data.synth_ids
        donor_haps_all = pop_data.donor_haps
        synth_haps_all = pop_data.synth_haps

        roh_n_donor = min(int(config.qc.roh_max_diploids_per_population), int(len(donor_ids_all)))
        roh_n_synth = min(int(config.qc.roh_max_diploids_per_population), int(len(synth_ids_all)))
        donor_ids = donor_ids_all[:roh_n_donor]
        synth_ids = synth_ids_all[:roh_n_synth]
        donor_haps = donor_haps_all[: 2 * roh_n_donor, :]
        synth_haps = synth_haps_all[: 2 * roh_n_synth, :]

        map_path = _resolve_map_path_for_population(config, chrom=chrom, pop=pop)
        map_bp, map_cm = load_genetic_map(map_path)
        cm_positions = interpolate_cm(sampled_pos.astype(np.int64, copy=False), map_bp, map_cm)
        if cm_positions.size != sampled_pos.shape[0]:
            raise ValueError(f"map interpolation mismatch for pop {pop}")

        thin_idx = _stride_thin_indices(int(sampled_pos.shape[0]), int(config.qc.roh_max_variants))
        donor_haps = donor_haps[:, thin_idx]
        synth_haps = synth_haps[:, thin_idx]
        cm_positions = cm_positions[thin_idx]

        donor_g = haplotypes_to_genotypes_u8(donor_haps)
        synth_g = haplotypes_to_genotypes_u8(synth_haps)

        donor_total, donor_nseg, donor_maxseg = _roh_metrics(
            donor_g,
            cm_positions,
            min_segment_cm=float(config.qc.roh_min_segment_cm),
            max_gap_cm=float(config.qc.roh_max_gap_cm),
        )
        synth_total, synth_nseg, synth_maxseg = _roh_metrics(
            synth_g,
            cm_positions,
            min_segment_cm=float(config.qc.roh_min_segment_cm),
            max_gap_cm=float(config.qc.roh_max_gap_cm),
        )

        plot_series[pop] = {
            "donor": donor_total,
            "synthetic": synth_total,
        }

        summ_d = _summarize(donor_total)
        summ_s = _summarize(synth_total)

        summary[pop] = {
            "status": "ok",
            "n_donor_samples": int(len(donor_ids)),
            "n_synth_samples": int(len(synth_ids)),
            "n_positions_used": int(cm_positions.size),
            "donor_total_cm": summ_d,
            "synthetic_total_cm": summ_s,
            "donor_mean_n_segments": float(np.mean(donor_nseg)) if donor_nseg.size else float("nan"),
            "synthetic_mean_n_segments": float(np.mean(synth_nseg)) if synth_nseg.size else float("nan"),
            "donor_mean_max_segment_cm": float(np.mean(donor_maxseg)) if donor_maxseg.size else float("nan"),
            "synthetic_mean_max_segment_cm": float(np.mean(synth_maxseg)) if synth_maxseg.size else float("nan"),
        }

        for source, tot, nseg, maxseg in (
            ("donor", donor_total, donor_nseg, donor_maxseg),
            ("synthetic", synth_total, synth_nseg, synth_maxseg),
        ):
            s_tot = _summarize(tot)
            rows.append(
                {
                    "population": pop,
                    "source": source,
                    "chrom": int(chrom),
                    "n_samples": int(tot.size),
                    "mean_total_cm": float(s_tot["mean"]),
                    "median_total_cm": float(s_tot["median"]),
                    "p90_total_cm": float(s_tot["p90"]),
                    "mean_n_segments": float(np.mean(nseg)) if nseg.size else float("nan"),
                    "mean_max_segment_cm": float(np.mean(maxseg)) if maxseg.size else float("nan"),
                }
            )

    table_path = diagnostics_dir / f"roh_summary_chr{chrom}.tsv"
    pd.DataFrame.from_records(rows).to_csv(table_path, sep="\t", index=False)
    created.append(table_path)

    plot_path: Path | None = None
    plot_error: str | None = None
    if config.qc.plots_enabled and plot_series:
        try:
            plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"roh_total_cm_chr{chrom}.png"
            save_roh_total_cm_png(plot_path, pop_to_series=plot_series)
            created.append(plot_path)
        except Exception as exc:
            logger.warning("ROH QC plot generation failed; continuing without plot: %s", exc)
            plot_error = str(exc)
            plot_path = None

    payload.update(
        {
            "status": "ok",
            "seed": int(config.qc.seed),
            "ext_max_diploids_per_population": int(config.qc.ext_max_diploids_per_population),
            "ext_max_variants": int(config.qc.ext_max_variants),
            "variant_subset_id": subset_id,
            "roh_min_segment_cm": float(config.qc.roh_min_segment_cm),
            "roh_max_gap_cm": float(config.qc.roh_max_gap_cm),
            "roh_max_variants": int(config.qc.roh_max_variants),
            "table_path": str(table_path),
            "plot_path": str(plot_path) if plot_path is not None else None,
            "plot_error": plot_error,
            "population_summary": summary,
        }
    )

    logger.info(
        "ROH QC complete chrom=%s pops=%s table=%s plot=%s",
        chrom,
        len(pop_names),
        table_path,
        plot_path,
    )
    return payload, created
