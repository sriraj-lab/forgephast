"""QC and population differentiation diagnostics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pysam
from hapstore import HapStore

from .metrics.ld_decay import (
    ld_curve_from_genotypes as local_ld_curve_from_genotypes,
    ld_decay_metrics_from_curves as local_ld_decay_metrics_from_curves,
)

from forgephast._internal.qc_stats import (
    ld_curve_from_genotypes as utils_ld_curve_from_genotypes,
    ld_decay_metrics_from_curves as utils_ld_decay_metrics_from_curves,
)

from ..config import RunConfig
from ..donors import load_manifest
from ..maps import interpolate_cm, load_genetic_map
from .metrics.fst import (
    aggregate_counts_across_stores,
    hapstore_counts_for_samples,
    hudson_fst_from_counts as matrix_hudson_fst_from_counts,
    sample_population_indices,
    sample_variant_indices,
)
from ..telemetry import RateLimitedProgress, telemetry_span
from ..utils import derived_seed, ensure_dir, write_json
from ..vcfio import (
    load_donor_haplotypes,
    load_donor_haplotypes_cached,
    merge_shard_vcfs,
    merge_shard_vcfs_bcftools,
)
from .registry import METRIC_REGISTRY

logger = logging.getLogger(__name__)


def _append_suffix(path_prefix: Path, suffix: str) -> Path:
    """Append a suffix to a PLINK output prefix without clobbering dotted names."""
    return Path(f"{path_prefix}{suffix}")


@dataclass
class QcReport:
    chrom: int
    vcf_path: Path | None
    sample_manifest_path: Path
    pairwise_fst: dict[str, float]
    within_split_fst: dict[str, float]
    gate_pass: bool
    diagnostics_path: Path
    mode: str
    population_vcf_paths: dict[str, Path]
    hapstore_path: Path | None = None
    population_hapstore_paths: dict[str, Path] | None = None
    donor_pairwise_fst: dict[str, float] | None = None
    ld_decay: dict[str, Any] | None = None
    ld_decay_table_path: Path | None = None
    elapsed_seconds: float = 0.0
    stage_timings: dict[str, float] | None = None


@dataclass
class DonorWithinPopRoundRobinReport:
    chrom: int
    n_folds: int
    fold_fraction: float
    seed: int
    workers: int
    diagnostics_path: Path
    table_path: Path
    summary_by_population: dict[str, dict[str, float | int]]
    n_records: int


@dataclass
class SyntheticPremergeWithinPopRoundRobinReport:
    chrom: int
    n_shards: int
    seed: int
    workers: int
    diagnostics_path: Path
    table_path: Path
    summary_by_population: dict[str, dict[str, float | int]]
    n_candidate_pairs: int
    n_selected_pairs: int
    n_records: int


@dataclass
class _QcInputs:
    mode: str
    sample_manifest: Path
    combined_vcf: Path | None
    combined_hapstore: Path | None
    population_vcfs: dict[str, Path]
    population_hapstores: dict[str, Path]


def _pair_key(a: str, b: str) -> str:
    return f"{a}__{b}" if a <= b else f"{b}__{a}"


def _default_pairwise_sets(pop_names: list[str]) -> list[tuple[str, str]]:
    return list(combinations(sorted(pop_names), 2))


def _resolve_pairwise_sets(config: RunConfig, pop_names: list[str]) -> list[tuple[str, str]]:
    if not config.qc.pairwise_fst_sets:
        return _default_pairwise_sets(pop_names)

    pairs: list[tuple[str, str]] = []
    for pair in config.qc.pairwise_fst_sets:
        if len(pair) != 2:
            raise ValueError(f"invalid pairwise_fst_sets entry: {pair}")
        a, b = str(pair[0]), str(pair[1])
        pairs.append((a, b))
    return pairs


def _resolve_within_split_checks(config: RunConfig, pop_names: list[str]) -> list[str]:
    # None => default behavior (all populations). Empty list => explicit disable.
    if config.qc.within_pop_split_checks is None:
        return list(pop_names)
    return [str(x) for x in config.qc.within_pop_split_checks]


def _resolve_python_qc_numba_threads(config: RunConfig) -> int:
    if config.qc.numba_threads is not None:
        return max(1, int(config.qc.numba_threads))
    if config.qc.synthetic_rr_min_cpus_per_match is not None:
        return max(1, int(config.qc.synthetic_rr_min_cpus_per_match))
    return max(1, int(config.qc.matrix_workers))


def _resolve_requested_populations(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    if not requested:
        return sorted(available)
    available_set = set(available)
    out: list[str] = []
    for pop in requested:
        pop_s = str(pop)
        if pop_s not in available_set:
            raise ValueError(f"Requested population {pop_s!r} is not present in donor manifest")
        if pop_s not in out:
            out.append(pop_s)
    return out


def _compute_msprime_tract_composition_gate(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pop_names: list[str],
) -> dict[str, Any]:
    pop_spec_by_name = {str(pop.name): pop for pop in config.populations}
    admixed_targets = [
        pop_name
        for pop_name in pop_names
        if (
            pop_name in pop_spec_by_name
            and pop_spec_by_name[pop_name].engine == "msprime_tracts"
            and pop_spec_by_name[pop_name].msprime is not None
            and len(pop_spec_by_name[pop_name].msprime.sources) > 1
        )
    ]
    if not admixed_targets:
        return {
            "enabled": False,
            "status": "no_admixed_msprime_populations",
            "gate_pass": None,
            "population_summary": {},
        }

    tolerance = float(config.qc.msprime_source_weight_abs_tolerance)
    overall_gate_pass = True
    population_summary: dict[str, Any] = {}
    for pop_name in admixed_targets:
        pop_spec = pop_spec_by_name[pop_name]
        assert pop_spec.msprime is not None
        configured = {
            str(source.population): float(source.weight) for source in pop_spec.msprime.sources
        }
        aggregate_marker_counts = {source: 0 for source in configured}
        expected_summary_paths = [
            config.run_root
            / "synthetic"
            / f"chr{chrom}"
            / f"shard_{shard_id}_of_{n_shards}"
            / f"{pop_name}.chr{chrom}.tract_summary.json"
            for shard_id in range(int(n_shards))
        ]
        found_paths: list[Path] = []
        missing_paths: list[Path] = []
        total_tract_count = 0
        for path in expected_summary_paths:
            if not path.exists():
                missing_paths.append(path)
                continue
            found_paths.append(path)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            total_tract_count += int(payload.get("tract_count", 0))
            counts = payload.get("source_marker_counts", {})
            if isinstance(counts, dict):
                for source in configured:
                    aggregate_marker_counts[source] += int(counts.get(source, 0))

        if not found_paths:
            overall_gate_pass = False
            population_summary[pop_name] = {
                "status": "missing_tract_summaries",
                "gate_pass": False,
                "configured_source_weights": configured,
                "observed_source_proportions": None,
                "source_weight_abs_diff": None,
                "max_abs_diff": None,
                "n_shards_expected": int(len(expected_summary_paths)),
                "n_shards_found": 0,
                "first_missing_summary_path": str(missing_paths[0]) if missing_paths else None,
            }
            continue

        if missing_paths:
            overall_gate_pass = False
            population_summary[pop_name] = {
                "status": "missing_shard_tract_summaries",
                "gate_pass": False,
                "configured_source_weights": configured,
                "observed_source_proportions": None,
                "source_weight_abs_diff": None,
                "max_abs_diff": None,
                "n_shards_expected": int(len(expected_summary_paths)),
                "n_shards_found": int(len(found_paths)),
                "first_missing_summary_path": str(missing_paths[0]),
            }
            continue

        total_markers = int(sum(int(v) for v in aggregate_marker_counts.values()))
        if total_markers <= 0:
            overall_gate_pass = False
            population_summary[pop_name] = {
                "status": "invalid_marker_count",
                "gate_pass": False,
                "configured_source_weights": configured,
                "observed_source_proportions": None,
                "source_weight_abs_diff": None,
                "max_abs_diff": None,
                "n_shards_expected": int(len(expected_summary_paths)),
                "n_shards_found": int(len(found_paths)),
                "total_tract_count": int(total_tract_count),
            }
            continue

        observed = {
            source: float(aggregate_marker_counts[source]) / float(total_markers)
            for source in configured
        }
        abs_diff = {
            source: abs(float(observed[source]) - float(configured[source]))
            for source in configured
        }
        max_abs_diff = max(abs_diff.values()) if abs_diff else float("nan")
        pop_gate = bool(max_abs_diff <= tolerance) if np.isfinite(max_abs_diff) else False
        if not pop_gate:
            overall_gate_pass = False
        population_summary[pop_name] = {
            "status": "ok",
            "gate_pass": bool(pop_gate),
            "configured_source_weights": configured,
            "observed_source_proportions": observed,
            "source_weight_abs_diff": abs_diff,
            "max_abs_diff": float(max_abs_diff),
            "tolerance": float(tolerance),
            "n_shards_expected": int(len(expected_summary_paths)),
            "n_shards_found": int(len(found_paths)),
            "total_tract_count": int(total_tract_count),
            "source_marker_counts": {k: int(v) for k, v in aggregate_marker_counts.items()},
        }

    return {
        "enabled": True,
        "status": "ok",
        "gate_pass": bool(overall_gate_pass),
        "tolerance": float(tolerance),
        "population_summary": population_summary,
    }


def _is_admixed_qc_enabled(config: RunConfig, population_msprime_is_admixed: dict[str, bool]) -> bool:
    mode = str(config.qc.admixed_qc_mode).strip().lower()
    if mode == "off":
        return False
    return bool(any(bool(x) for x in population_msprime_is_admixed.values()))


def _is_admixed_involved_pair(pair: tuple[str, str], population_msprime_is_admixed: dict[str, bool]) -> bool:
    a, b = pair
    return bool(population_msprime_is_admixed.get(a, False) or population_msprime_is_admixed.get(b, False))


def _ks_one_sample_exponential(values: np.ndarray, rate: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size == 0:
        return float("nan")
    if not np.isfinite(rate) or rate <= 0.0:
        return float("nan")
    xs = np.sort(x)
    n = float(xs.size)
    cdf_exp = 1.0 - np.exp(-float(rate) * xs)
    ecdf_right = np.arange(1, xs.size + 1, dtype=np.float64) / n
    ecdf_left = np.arange(0, xs.size, dtype=np.float64) / n
    ks = max(float(np.max(np.abs(ecdf_right - cdf_exp))), float(np.max(np.abs(ecdf_left - cdf_exp))))
    return float(ks)


def _ks_histogram_exponential(counts: np.ndarray, bin_edges: np.ndarray, rate: float) -> float:
    c = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if c.size == 0 or edges.size != (c.size + 1):
        return float("nan")
    n = float(np.sum(c))
    if n <= 0.0 or not np.isfinite(rate) or rate <= 0.0:
        return float("nan")
    cdf_obs = np.cumsum(c) / n
    x = edges[1:]
    cdf_exp = 1.0 - np.exp(-float(rate) * x)
    return float(np.max(np.abs(cdf_obs - cdf_exp)))


def _cdf_finite_chr_poisson_switch(
    x: np.ndarray,
    *,
    rate: float,
    chrom_length_cm: float,
) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64)
    if not np.isfinite(rate) or rate <= 0.0:
        return np.full(xv.shape, np.nan, dtype=np.float64)
    if not np.isfinite(chrom_length_cm) or chrom_length_cm <= 0.0:
        return np.full(xv.shape, np.nan, dtype=np.float64)
    m = float(rate) * float(chrom_length_cm)
    u = np.clip(xv / float(chrom_length_cm), 0.0, 1.0)
    tail = np.exp(-m * u) * (1.0 + m * (1.0 - u)) / (1.0 + m)
    cdf = 1.0 - tail
    cdf = np.where(xv <= 0.0, 0.0, cdf)
    cdf = np.where(xv >= float(chrom_length_cm), 1.0, cdf)
    return cdf.astype(np.float64, copy=False)


def _ks_one_sample_finite_chr_poisson_switch(
    values: np.ndarray,
    *,
    rate: float,
    chrom_length_cm: float,
) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size == 0:
        return float("nan")
    xs = np.sort(x)
    cdf_exp = _cdf_finite_chr_poisson_switch(xs, rate=rate, chrom_length_cm=chrom_length_cm)
    if not np.all(np.isfinite(cdf_exp)):
        return float("nan")
    n = float(xs.size)
    ecdf_right = np.arange(1, xs.size + 1, dtype=np.float64) / n
    ecdf_left = np.arange(0, xs.size, dtype=np.float64) / n
    ks = max(float(np.max(np.abs(ecdf_right - cdf_exp))), float(np.max(np.abs(ecdf_left - cdf_exp))))
    return float(ks)


def _ks_histogram_finite_chr_poisson_switch(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    *,
    rate: float,
    chrom_length_cm: float,
) -> float:
    c = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if c.size == 0 or edges.size != (c.size + 1):
        return float("nan")
    n = float(np.sum(c))
    if n <= 0.0:
        return float("nan")
    cdf_obs = np.cumsum(c) / n
    x = edges[1:]
    cdf_exp = _cdf_finite_chr_poisson_switch(x, rate=rate, chrom_length_cm=chrom_length_cm)
    if not np.all(np.isfinite(cdf_exp)):
        return float("nan")
    return float(np.max(np.abs(cdf_obs - cdf_exp)))


def _compute_admixed_tract_length_gate(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pop_names: list[str],
    diagnostics_dir: Path,
) -> dict[str, Any]:
    if not bool(config.qc.admixed_tract_length_gate_enabled):
        return {
            "enabled": False,
            "status": "disabled_by_config",
            "gate_pass": None,
            "population_summary": {},
        }

    pop_spec_by_name = {str(pop.name): pop for pop in config.populations}
    admixed_targets = [
        pop_name
        for pop_name in pop_names
        if (
            pop_name in pop_spec_by_name
            and pop_spec_by_name[pop_name].engine == "msprime_tracts"
            and pop_spec_by_name[pop_name].msprime is not None
            and len(pop_spec_by_name[pop_name].msprime.sources) > 1
        )
    ]
    if not admixed_targets:
        return {
            "enabled": False,
            "status": "no_admixed_msprime_populations",
            "gate_pass": None,
            "population_summary": {},
        }

    ks_max = float(config.qc.admixed_tract_length_ks_max)
    min_tracts = int(config.qc.admixed_tract_length_min_tracts)
    overall_gate_pass = True
    pop_summary: dict[str, Any] = {}

    for pop_name in admixed_targets:
        pop_spec = pop_spec_by_name[pop_name]
        assert pop_spec.msprime is not None
        t_gen = (
            float(pop_spec.msprime.admixture_time_generations)
            if pop_spec.msprime.admixture_time_generations is not None
            else None
        )
        summary_paths = [
            config.run_root
            / "synthetic"
            / f"chr{chrom}"
            / f"shard_{shard_id}_of_{n_shards}"
            / f"{pop_name}.chr{chrom}.tract_summary.json"
            for shard_id in range(int(n_shards))
        ]
        tsv_paths = [
            config.run_root
            / "synthetic"
            / f"chr{chrom}"
            / f"shard_{shard_id}_of_{n_shards}"
            / f"{pop_name}.chr{chrom}.tracts.tsv"
            for shard_id in range(int(n_shards))
        ]
        found_summaries = [path for path in summary_paths if path.exists()]
        missing_summaries = [path for path in summary_paths if not path.exists()]
        if missing_summaries:
            overall_gate_pass = False
            pop_summary[pop_name] = {
                "status": "missing_shard_tract_summaries",
                "gate_pass": False,
                "ks_stat": None,
                "ks_threshold": float(ks_max),
                "n_tracts": 0,
                "n_shards_expected": int(len(summary_paths)),
                "n_shards_found": int(len(found_summaries)),
                "first_missing_summary_path": str(missing_summaries[0]),
            }
            continue

        span_vals: list[float] = []
        morgans_vals: list[float] = []
        for path in found_summaries:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            span = payload.get("chrom_span_bp")
            morgans = payload.get("chrom_map_morgans")
            if span is not None and np.isfinite(float(span)) and float(span) > 0.0:
                span_vals.append(float(span))
            if morgans is not None and np.isfinite(float(morgans)) and float(morgans) > 0.0:
                morgans_vals.append(float(morgans))
        chrom_span_bp = float(np.median(np.asarray(span_vals, dtype=np.float64))) if span_vals else float("nan")
        chrom_map_morgans = (
            float(np.median(np.asarray(morgans_vals, dtype=np.float64))) if morgans_vals else float("nan")
        )

        if (
            t_gen is None
            or not np.isfinite(float(t_gen))
            or float(t_gen) <= 0.0
            or not np.isfinite(chrom_span_bp)
            or not np.isfinite(chrom_map_morgans)
            or chrom_span_bp <= 0.0
            or chrom_map_morgans <= 0.0
        ):
            pop_summary[pop_name] = {
                "status": "unsupported_model",
                "gate_pass": None,
                "reason": "pulse_model_parameters_unavailable",
                "ks_stat": None,
                "ks_threshold": float(ks_max),
                "n_tracts": None,
                "admixture_time_generations": t_gen,
                "chrom_span_bp": chrom_span_bp,
                "chrom_map_morgans": chrom_map_morgans,
            }
            continue

        configured_weights = np.asarray([float(src.weight) for src in pop_spec.msprime.sources], dtype=np.float64)
        switch_fraction = float(1.0 - np.sum(np.square(configured_weights)))
        chrom_length_cm = float(100.0 * float(chrom_map_morgans))
        if not np.isfinite(switch_fraction) or switch_fraction <= 0.0:
            pop_summary[pop_name] = {
                "status": "unsupported_model",
                "gate_pass": None,
                "reason": "invalid_source_switch_fraction",
                "switch_fraction": (float(switch_fraction) if np.isfinite(switch_fraction) else None),
            }
            continue

        # Finite-window pulse-switch approximation:
        # average ancestry-switch rate per cM is t * (1 - sum_i p_i^2) / 100.
        rate_cm = (float(t_gen) * float(switch_fraction)) / 100.0
        cm_per_bp = (100.0 * float(chrom_map_morgans) / float(chrom_span_bp))
        plot_values: np.ndarray | None = None
        plot_hist_counts: np.ndarray | None = None
        plot_hist_edges: np.ndarray | None = None
        use_tsv = any(path.exists() for path in tsv_paths)
        if use_tsv:
            map_bp_arr: np.ndarray | None = None
            map_cm_arr: np.ndarray | None = None
            tract_lengths: list[np.ndarray] = []
            for path in tsv_paths:
                if not path.exists():
                    continue
                cols = pd.read_csv(path, sep="\t", nrows=0).columns
                if "cm_length" in cols:
                    frame = pd.read_csv(path, sep="\t", usecols=["cm_length"])
                    vals = np.asarray(frame["cm_length"].to_numpy(dtype=np.float64), dtype=np.float64)
                    source_type = "tract_tsv_cm"
                elif "start_bp" in cols and "end_bp" in cols:
                    if map_bp_arr is None or map_cm_arr is None:
                        map_path = _resolve_map_path_for_population(config, chrom=int(chrom), pop=str(pop_name))
                        map_bp_arr, map_cm_arr = load_genetic_map(map_path)
                    frame = pd.read_csv(path, sep="\t", usecols=["start_bp", "end_bp"])
                    start_bp = np.asarray(frame["start_bp"].to_numpy(dtype=np.float64), dtype=np.float64)
                    end_bp = np.asarray(frame["end_bp"].to_numpy(dtype=np.float64), dtype=np.float64)
                    vals = np.interp(end_bp, map_bp_arr, map_cm_arr) - np.interp(start_bp, map_bp_arr, map_cm_arr)
                    source_type = "tract_tsv_bp_map_interp_cm"
                elif "bp_length" in cols:
                    frame = pd.read_csv(path, sep="\t", usecols=["bp_length"])
                    vals = np.asarray(frame["bp_length"].to_numpy(dtype=np.float64), dtype=np.float64) * float(cm_per_bp)
                    source_type = "tract_tsv_bp_to_cm_approx"
                else:
                    continue
                vals = vals[np.isfinite(vals) & (vals >= 0.0)]
                tract_lengths.append(vals)
            if tract_lengths:
                lengths = np.concatenate(tract_lengths, axis=0)
            else:
                lengths = np.zeros(0, dtype=np.float64)
            n_tracts = int(lengths.size)
            ks_stat = _ks_one_sample_finite_chr_poisson_switch(
                lengths,
                rate=rate_cm,
                chrom_length_cm=chrom_length_cm,
            )
            if lengths.size > 0:
                plot_values = lengths
            if not tract_lengths:
                source_type = "tract_tsv_missing_length_columns"
        else:
            total_tracts = 0
            weighted_ks_max = float("nan")
            source_type = "summary_histogram_cm"
            best_hist_n = -1
            for path in found_summaries:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                hist_cm = payload.get("tract_length_cm_histogram", {})
                if isinstance(hist_cm, dict) and hist_cm.get("counts") and hist_cm.get("bin_edges_cm"):
                    counts = np.asarray(hist_cm.get("counts", []), dtype=np.int64)
                    edges = np.asarray(hist_cm.get("bin_edges_cm", []), dtype=np.float64)
                else:
                    hist_bp = payload.get("tract_length_bp_histogram", {})
                    counts = np.asarray(hist_bp.get("counts", []), dtype=np.int64)
                    edges_bp = np.asarray(hist_bp.get("bin_edges_bp", []), dtype=np.float64)
                    edges = edges_bp * float(cm_per_bp)
                    source_type = "summary_histogram_bp_to_cm_approx"
                shard_n = int(np.sum(counts))
                shard_ks = _ks_histogram_finite_chr_poisson_switch(
                    counts=counts,
                    bin_edges=edges,
                    rate=rate_cm,
                    chrom_length_cm=chrom_length_cm,
                )
                if np.isfinite(shard_ks):
                    weighted_ks_max = shard_ks if not np.isfinite(weighted_ks_max) else max(weighted_ks_max, shard_ks)
                total_tracts += shard_n
                if shard_n > best_hist_n and counts.size > 0 and edges.size == (counts.size + 1):
                    best_hist_n = shard_n
                    plot_hist_counts = np.asarray(counts, dtype=np.float64)
                    plot_hist_edges = np.asarray(edges, dtype=np.float64)
            n_tracts = int(total_tracts)
            ks_stat = float(weighted_ks_max) if np.isfinite(weighted_ks_max) else float("nan")

        if n_tracts < min_tracts:
            pop_gate = False
            status = "insufficient_tracts"
        elif not np.isfinite(ks_stat):
            pop_gate = False
            status = "invalid_ks_stat"
        else:
            pop_gate = bool(float(ks_stat) <= float(ks_max))
            status = "ok"
        if pop_gate is False:
            overall_gate_pass = False

        tract_plot_path: Path | None = None
        tract_plot_error: str | None = None
        if bool(config.qc.plots_enabled):
            try:
                from .plots import save_tract_length_ks_png

                plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
                plot_dir.mkdir(parents=True, exist_ok=True)
                tract_plot_path = plot_dir / f"admixed_tract_ks_{pop_name}_chr{chrom}.png"
                save_tract_length_ks_png(
                    tract_plot_path,
                    rate=float(rate_cm),
                    title=f"Admixed tract-length fit chr{chrom} {pop_name}",
                    xlabel="Tract length (cM)",
                    values=plot_values,
                    hist_counts=plot_hist_counts,
                    hist_bin_edges=plot_hist_edges,
                    model="finite_chr_poisson_switch",
                    chrom_length=float(chrom_length_cm),
                )
            except Exception as exc:
                tract_plot_error = str(exc)
                tract_plot_path = None

        pop_summary[pop_name] = {
            "status": status,
            "gate_pass": bool(pop_gate),
            "n_tracts": int(n_tracts),
            "min_tracts_required": int(min_tracts),
            "ks_stat": (float(ks_stat) if np.isfinite(ks_stat) else None),
            "ks_threshold": float(ks_max),
            "admixture_time_generations": float(t_gen),
            "chrom_span_bp": float(chrom_span_bp),
            "chrom_map_morgans": float(chrom_map_morgans),
            "switch_fraction": float(switch_fraction),
            "expected_switch_rate_cm": float(rate_cm),
            "expected_switch_rate_bp_approx": float(rate_cm * cm_per_bp),
            "expected_mean_tract_length_cm_infinite_exp": float(1.0 / float(rate_cm)),
            "expected_mean_tract_length_cm_finite_chr": float(chrom_length_cm / (1.0 + (rate_cm * chrom_length_cm))),
            "data_source": source_type,
            "null_model": "finite_chr_poisson_switch",
            "plot_path": str(tract_plot_path) if tract_plot_path is not None else None,
            "plot_error": tract_plot_error,
        }

    return {
        "enabled": True,
        "status": "ok",
        "gate_pass": bool(overall_gate_pass),
        "population_summary": pop_summary,
        "ks_threshold": float(ks_max),
        "min_tracts_required": int(min_tracts),
    }


def _compute_admixed_ancestry_variance_gate(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pop_names: list[str],
) -> dict[str, Any]:
    if not bool(config.qc.admixed_ancestry_variance_gate_enabled):
        return {
            "enabled": False,
            "status": "disabled_by_config",
            "gate_pass": None,
            "population_summary": {},
        }

    pop_spec_by_name = {str(pop.name): pop for pop in config.populations}
    admixed_targets = [
        pop_name
        for pop_name in pop_names
        if (
            pop_name in pop_spec_by_name
            and pop_spec_by_name[pop_name].engine == "msprime_tracts"
            and pop_spec_by_name[pop_name].msprime is not None
            and len(pop_spec_by_name[pop_name].msprime.sources) > 1
        )
    ]
    if not admixed_targets:
        return {
            "enabled": False,
            "status": "no_admixed_msprime_populations",
            "gate_pass": None,
            "population_summary": {},
        }

    rel_tol = float(config.qc.admixed_ancestry_variance_rel_tolerance)
    min_n = int(config.qc.admixed_min_individuals_for_variance_gate)
    overall_gate_pass = True
    pop_summary: dict[str, Any] = {}

    for pop_name in admixed_targets:
        pop_spec = pop_spec_by_name[pop_name]
        assert pop_spec.msprime is not None
        t_gen = (
            float(pop_spec.msprime.admixture_time_generations)
            if pop_spec.msprime.admixture_time_generations is not None
            else None
        )
        configured = {str(s.population): float(s.weight) for s in pop_spec.msprime.sources}
        ind_paths = [
            config.run_root
            / "synthetic"
            / f"chr{chrom}"
            / f"shard_{shard_id}_of_{n_shards}"
            / f"{pop_name}.chr{chrom}.individual_ancestry.tsv"
            for shard_id in range(int(n_shards))
        ]
        found_paths = [path for path in ind_paths if path.exists()]
        missing_paths = [path for path in ind_paths if not path.exists()]
        if missing_paths:
            overall_gate_pass = False
            pop_summary[pop_name] = {
                "status": "missing_individual_ancestry_artifacts",
                "gate_pass": False,
                "n_individuals": 0,
                "n_shards_expected": int(len(ind_paths)),
                "n_shards_found": int(len(found_paths)),
                "first_missing_path": str(missing_paths[0]),
            }
            continue

        if t_gen is None or not np.isfinite(float(t_gen)) or float(t_gen) <= 0.0:
            pop_summary[pop_name] = {
                "status": "unsupported_model",
                "gate_pass": None,
                "reason": "pulse_model_parameters_unavailable",
            }
            continue

        summary_paths = [
            config.run_root
            / "synthetic"
            / f"chr{chrom}"
            / f"shard_{shard_id}_of_{n_shards}"
            / f"{pop_name}.chr{chrom}.tract_summary.json"
            for shard_id in range(int(n_shards))
        ]
        found_summaries = [path for path in summary_paths if path.exists()]
        missing_summaries = [path for path in summary_paths if not path.exists()]
        if missing_summaries:
            overall_gate_pass = False
            pop_summary[pop_name] = {
                "status": "missing_shard_tract_summaries",
                "gate_pass": False,
                "n_individuals": None,
                "n_shards_expected": int(len(summary_paths)),
                "n_shards_found": int(len(found_summaries)),
                "first_missing_summary_path": str(missing_summaries[0]),
            }
            continue

        l_vals: list[float] = []
        for path in found_summaries:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            l = payload.get("chrom_map_morgans")
            if l is not None and np.isfinite(float(l)) and float(l) > 0.0:
                l_vals.append(float(l))
        if not l_vals:
            pop_summary[pop_name] = {
                "status": "unsupported_model",
                "gate_pass": None,
                "reason": "chrom_map_morgans_unavailable",
            }
            continue
        chrom_map_morgans = float(np.median(np.asarray(l_vals, dtype=np.float64)))

        frames = [pd.read_csv(path, sep="\t", dtype=str) for path in found_paths]
        if not frames:
            overall_gate_pass = False
            pop_summary[pop_name] = {
                "status": "empty_individual_ancestry_artifacts",
                "gate_pass": False,
                "n_individuals": 0,
            }
            continue
        merged = pd.concat(frames, ignore_index=True)
        if "sample_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["sample_id"])
        else:
            merged = merged.drop_duplicates(subset=["individual_index", "population", "chrom"])
        n_individuals = int(len(merged))
        if n_individuals < min_n:
            overall_gate_pass = False
            pop_summary[pop_name] = {
                "status": "insufficient_individuals",
                "gate_pass": False,
                "n_individuals": n_individuals,
                "min_individuals_required": int(min_n),
            }
            continue

        per_source: dict[str, Any] = {}
        pop_gate = True
        for source, p in configured.items():
            col_cm = f"prop_cm__{source}"
            col_bp = f"prop_bp__{source}"
            col_markers = f"prop_markers__{source}"
            if col_cm in merged.columns:
                col = col_cm
                measure = "cm"
            elif col_bp in merged.columns:
                col = col_bp
                measure = "bp"
            elif col_markers in merged.columns:
                col = col_markers
                measure = "markers_fallback"
            else:
                pop_gate = False
                per_source[source] = {
                    "status": "missing_column",
                    "column": col_bp,
                    "fallback_column": col_markers,
                    "observed_variance": None,
                    "expected_variance": None,
                    "relative_error": None,
                    "source_pass": False,
                }
                continue
            vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                pop_gate = False
                per_source[source] = {
                    "status": "no_finite_values",
                    "column": col,
                    "observed_variance": None,
                    "expected_variance": None,
                    "relative_error": None,
                    "source_pass": False,
                }
                continue
            obs_var = float(np.var(vals, ddof=0))
            # Pulse-admixture chr-specific ancestry variance approximation (uniform recombination):
            # Var(q_i,chr) ~= p_i(1-p_i) / (t * L_chr), where L_chr is in Morgans.
            #
            # Note: for msprime_tracts runs we prefer `prop_cm__*` (genetic-length-weighted ancestry),
            # which matches the recombination-scale assumptions of this expected value.
            exp_var = float((float(p) * (1.0 - float(p))) / (float(t_gen) * float(chrom_map_morgans)))
            expected_model = "pulse_uniform_recomb_approx"
            rel_err = (
                abs(obs_var - exp_var) / max(exp_var, 1e-12)
                if np.isfinite(obs_var) and np.isfinite(exp_var)
                else float("nan")
            )
            source_pass = bool(np.isfinite(rel_err) and (rel_err <= rel_tol))
            if source_pass is False:
                pop_gate = False
            per_source[source] = {
                "status": "ok",
                "column": col,
                "measure": measure,
                "observed_variance": obs_var,
                "expected_variance": exp_var,
                "expected_model": str(expected_model),
                "relative_error": (float(rel_err) if np.isfinite(rel_err) else None),
                "source_pass": bool(source_pass),
            }

        if pop_gate is False:
            overall_gate_pass = False
        pop_summary[pop_name] = {
            "status": "ok",
            "gate_pass": bool(pop_gate),
            "n_individuals": n_individuals,
            "min_individuals_required": int(min_n),
            "admixture_time_generations": float(t_gen),
            "chrom_map_morgans": float(chrom_map_morgans),
            "relative_tolerance": float(rel_tol),
            "per_source": per_source,
        }

    return {
        "enabled": True,
        "status": "ok",
        "gate_pass": bool(overall_gate_pass),
        "population_summary": pop_summary,
        "relative_tolerance": float(rel_tol),
        "min_individuals_required": int(min_n),
    }


def _apply_admixed_non_gating_to_metric(
    metric_payload: dict[str, Any],
    *,
    admixed_pop_names: set[str],
    reason: str,
) -> dict[str, Any]:
    if not isinstance(metric_payload, dict):
        return metric_payload
    summary = metric_payload.get("population_summary")
    if not isinstance(summary, dict):
        return metric_payload

    for pop_name, pop_payload in summary.items():
        if pop_name not in admixed_pop_names:
            continue
        if not isinstance(pop_payload, dict):
            continue
        pop_payload["gate_pass"] = None
        pop_payload["gate_reason"] = str(reason)

    non_admixed_gate_values: list[bool] = []
    for pop_name, pop_payload in summary.items():
        if pop_name in admixed_pop_names:
            continue
        if not isinstance(pop_payload, dict):
            continue
        pop_gate = pop_payload.get("gate_pass")
        if isinstance(pop_gate, bool):
            non_admixed_gate_values.append(bool(pop_gate))

    if non_admixed_gate_values:
        metric_payload["gate_pass"] = bool(all(non_admixed_gate_values))
    else:
        metric_payload["gate_pass"] = None
    metric_payload["admixed_non_gating_applied"] = True
    metric_payload["admixed_non_gating_reason"] = str(reason)
    return metric_payload


def _compute_admixed_pca_cline_diagnostic(
    *,
    config: RunConfig,
    chrom: int,
    ext_data: Any | None,
    inputs: _QcInputs,
    groups: dict[str, list[str]],
    pop_names: list[str],
    diagnostics_dir: Path,
) -> dict[str, Any]:
    if not bool(config.qc.admixed_diag_pca_cline_enabled):
        return {"enabled": False, "status": "disabled_by_config", "population_summary": {}}

    pop_spec_by_name = {str(pop.name): pop for pop in config.populations}
    donors_by_population = load_manifest(config).donors_by_population
    out: dict[str, Any] = {}
    for pop_name in pop_names:
        pop_spec = pop_spec_by_name.get(pop_name)
        if pop_spec is None or pop_spec.msprime is None or len(pop_spec.msprime.sources) < 2:
            continue

        # NOTE: This diagnostic was originally 2-way-only. For k-way msprime populations (k>2),
        # we generalize it to a "mixture PCA" plot: all donor source populations + the synthetic
        # admixed population in the same (PC1, PC2) space.
        if len(pop_spec.msprime.sources) != 2:
            sources = [str(source.population) for source in pop_spec.msprime.sources]
            weights = np.asarray([float(source.weight) for source in pop_spec.msprime.sources], dtype=np.float64)
            if len(sources) != int(weights.size) or not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
                out[pop_name] = {"status": "invalid_source_weights", "gate_pass": None}
                continue
            weights = weights / float(np.sum(weights))

            synth_ids_all = list(groups.get(pop_name, []))
            if not synth_ids_all:
                out[pop_name] = {"status": "missing_synthetic_population", "gate_pass": None}
                continue

            cap = int(config.qc.ext_max_diploids_per_population)
            donor_ids_by_source: dict[str, list[str]] = {}
            missing_sources: list[str] = []
            for src in sources:
                donor_ids_all = list(donors_by_population.get(src, []))
                if not donor_ids_all:
                    missing_sources.append(str(src))
                    continue
                donor_ids_by_source[src] = _subsample_ids(
                    donor_ids_all,
                    target_n=min(len(donor_ids_all), cap),
                    seed=int(derived_seed(int(config.qc.seed), "admixed_pca_source_donor", src, chrom)),
                )
            if missing_sources:
                out[pop_name] = {
                    "status": "missing_source_donors",
                    "missing_sources": missing_sources,
                    "gate_pass": None,
                }
                continue

            synth_ids = _subsample_ids(
                synth_ids_all,
                target_n=min(len(synth_ids_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_synth", pop_name, chrom)),
            )

            per_src_pos: dict[str, np.ndarray] = {}
            per_src_haps: dict[str, np.ndarray] = {}
            try:
                for src in sources:
                    src_kind, src_path = _resolve_donor_panel_source_for_population(
                        config,
                        chrom=int(chrom),
                        pop=str(src),
                    )
                    pos_src, haps_src = _load_sample_haplotypes_from_panel(
                        config=config,
                        vcf_path=src_path,
                        chrom=int(chrom),
                        sample_ids=donor_ids_by_source[src],
                        source_kind=src_kind,
                    )
                    per_src_pos[src] = np.asarray(pos_src, dtype=np.int64)
                    per_src_haps[src] = np.asarray(haps_src, dtype=np.uint8, copy=False)

                pos_s, synth_raw = _load_synthetic_population_haplotypes(
                    config=config,
                    inputs=inputs,
                    chrom=int(chrom),
                    pop=str(pop_name),
                    sample_ids=synth_ids,
                )
            except Exception:
                out[pop_name] = {"status": "missing_ext_data_populations", "gate_pass": None}
                continue

            common = np.asarray(pos_s, dtype=np.int64)
            for src in sources:
                common = np.intersect1d(common, per_src_pos[src], assume_unique=False)
            if common.size < 2:
                out[pop_name] = {"status": "insufficient_common_variants", "gate_pass": None}
                continue

            donor_blocks: list[np.ndarray] = []
            donor_labels: list[str] = []
            src_means_pc1: dict[str, float] = {}
            src_means_pc2: dict[str, float] = {}
            n_donors_by_source: dict[str, int] = {}

            # Align donor + synthetic matrices to common position axis.
            per_src_geno: dict[str, np.ndarray] = {}
            for src in sources:
                src_pos = per_src_pos[src]
                idx = np.searchsorted(src_pos, common)
                if idx.size != common.size or not np.array_equal(src_pos[idx], common):
                    out[pop_name] = {"status": "position_alignment_failed", "gate_pass": None}
                    per_src_geno = {}
                    break
                haps = np.asarray(per_src_haps[src][:, idx], dtype=np.uint8, copy=False)
                geno = haps[0::2, :].astype(np.float64, copy=False) + haps[1::2, :].astype(np.float64, copy=False)
                per_src_geno[src] = geno
            if not per_src_geno:
                continue

            pos_s_arr = np.asarray(pos_s, dtype=np.int64)
            idx_s = np.searchsorted(pos_s_arr, common)
            if idx_s.size != common.size or not np.array_equal(pos_s_arr[idx_s], common):
                out[pop_name] = {"status": "position_alignment_failed", "gate_pass": None}
                continue
            synth = np.asarray(synth_raw[:, idx_s], dtype=np.uint8, copy=False)
            gs = synth[0::2, :].astype(np.float64, copy=False) + synth[1::2, :].astype(np.float64, copy=False)

            g_all = np.vstack([per_src_geno[src] for src in sources])
            mean = np.nanmean(g_all, axis=0)
            std = np.nanstd(g_all, axis=0)
            std = np.where(std > 1e-8, std, 1.0)
            g_all_std = (np.where(np.isfinite(g_all), g_all, mean) - mean) / std
            gs_std = (np.where(np.isfinite(gs), gs, mean) - mean) / std

            u, s, vt = np.linalg.svd(g_all_std, full_matrices=False)
            if vt.shape[0] <= 1:
                out[pop_name] = {"status": "svd_failed", "gate_pass": None}
                continue
            pc1 = vt[0]
            pc2 = vt[1]

            score_all_1 = g_all_std @ pc1
            score_all_2 = g_all_std @ pc2
            score_s_1 = gs_std @ pc1
            score_s_2 = gs_std @ pc2

            offsets: dict[str, tuple[int, int]] = {}
            cursor = 0
            for src in sources:
                n = int(per_src_geno[src].shape[0])
                offsets[src] = (cursor, cursor + n)
                cursor += n

            for src in sources:
                lo, hi = offsets[src]
                src_means_pc1[src] = float(np.mean(score_all_1[lo:hi]))
                src_means_pc2[src] = float(np.mean(score_all_2[lo:hi]))
                n_donors_by_source[src] = int(hi - lo)
                donor_labels.extend([src] * int(hi - lo))

            donor_scores = np.column_stack([score_all_1, score_all_2])
            synth_scores = np.column_stack([score_s_1, score_s_2])

            expected_pc1 = float(np.sum(weights * np.asarray([src_means_pc1[src] for src in sources], dtype=np.float64)))
            expected_pc2 = float(np.sum(weights * np.asarray([src_means_pc2[src] for src in sources], dtype=np.float64)))
            synth_mean_pc1 = float(np.mean(score_s_1))
            synth_mean_pc2 = float(np.mean(score_s_2))

            centroids = np.column_stack(
                [
                    np.asarray([src_means_pc1[src] for src in sources], dtype=np.float64),
                    np.asarray([src_means_pc2[src] for src in sources], dtype=np.float64),
                ]
            )
            if centroids.shape[0] >= 2:
                d = centroids[:, None, :] - centroids[None, :, :]
                source_sep = float(np.max(np.sqrt(np.sum(d * d, axis=2))))
            else:
                source_sep = float("nan")
            offset_abs = float(np.sqrt((synth_mean_pc1 - expected_pc1) ** 2 + (synth_mean_pc2 - expected_pc2) ** 2))

            plot_path: Path | None = None
            plot_error: str | None = None
            if bool(config.qc.plots_enabled):
                try:
                    from .plots import save_pca_scatter_png

                    plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = plot_dir / f"admixed_pca_cline_{pop_name}_chr{chrom}.png"
                    save_pca_scatter_png(
                        plot_path,
                        donor_scores=donor_scores,
                        donor_labels=donor_labels,
                        synthetic_scores=synth_scores,
                        synthetic_labels=[pop_name] * int(score_s_1.size),
                        title=f"Admixed PCA mixture chr{chrom} {pop_name}",
                    )
                except Exception as exc:
                    plot_error = str(exc)
                    plot_path = None

            out[pop_name] = {
                "status": "ok",
                "sources": list(sources),
                "weights": {src: float(w) for src, w in zip(sources, weights)},
                "source_pc_means": {src: {"pc1": src_means_pc1[src], "pc2": src_means_pc2[src]} for src in sources},
                "expected_weighted_centroid": {"pc1": expected_pc1, "pc2": expected_pc2},
                "synthetic_centroid": {"pc1": synth_mean_pc1, "pc2": synth_mean_pc2},
                "centroid_offset_abs": float(offset_abs),
                "centroid_offset_rel_to_source_sep": (
                    float(offset_abs / max(source_sep, 1e-12)) if np.isfinite(source_sep) else None
                ),
                "source_centroid_separation": (float(source_sep) if np.isfinite(source_sep) else None),
                "n_donors_by_source": n_donors_by_source,
                "n_synthetic": int(score_s_1.size),
                "plot_path": str(plot_path) if plot_path is not None else None,
                "plot_error": plot_error,
            }
            continue

        source_a = str(pop_spec.msprime.sources[0].population)
        source_b = str(pop_spec.msprime.sources[1].population)
        donor_a: np.ndarray
        donor_b: np.ndarray
        synth: np.ndarray
        if (
            ext_data is not None
            and source_a in ext_data.per_pop
            and source_b in ext_data.per_pop
            and pop_name in ext_data.per_pop
        ):
            donor_a = ext_data.per_pop[source_a].donor_haps
            donor_b = ext_data.per_pop[source_b].donor_haps
            synth = ext_data.per_pop[pop_name].synth_haps
        else:
            donor_ids_a_all = list(donors_by_population.get(source_a, []))
            donor_ids_b_all = list(donors_by_population.get(source_b, []))
            synth_ids_all = list(groups.get(pop_name, []))
            if not donor_ids_a_all or not donor_ids_b_all or not synth_ids_all:
                out[pop_name] = {"status": "missing_ext_data_populations", "gate_pass": None}
                continue
            cap = int(config.qc.ext_max_diploids_per_population)
            donor_ids_a = _subsample_ids(
                donor_ids_a_all,
                target_n=min(len(donor_ids_a_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_source_donor", source_a, chrom)),
            )
            donor_ids_b = _subsample_ids(
                donor_ids_b_all,
                target_n=min(len(donor_ids_b_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_source_donor", source_b, chrom)),
            )
            synth_ids = _subsample_ids(
                synth_ids_all,
                target_n=min(len(synth_ids_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_synth", pop_name, chrom)),
            )
            try:
                source_kind_a, source_path_a = _resolve_donor_panel_source_for_population(
                    config,
                    chrom=int(chrom),
                    pop=str(source_a),
                )
                source_kind_b, source_path_b = _resolve_donor_panel_source_for_population(
                    config,
                    chrom=int(chrom),
                    pop=str(source_b),
                )
                pos_a, donor_a_raw = _load_sample_haplotypes_from_panel(
                    config=config,
                    vcf_path=source_path_a,
                    chrom=int(chrom),
                    sample_ids=donor_ids_a,
                    source_kind=source_kind_a,
                )
                pos_b, donor_b_raw = _load_sample_haplotypes_from_panel(
                    config=config,
                    vcf_path=source_path_b,
                    chrom=int(chrom),
                    sample_ids=donor_ids_b,
                    source_kind=source_kind_b,
                )
                pos_s, synth_raw = _load_synthetic_population_haplotypes(
                    config=config,
                    inputs=inputs,
                    chrom=int(chrom),
                    pop=str(pop_name),
                    sample_ids=synth_ids,
                )
            except Exception:
                out[pop_name] = {"status": "missing_ext_data_populations", "gate_pass": None}
                continue
            common = np.intersect1d(np.intersect1d(pos_a, pos_b, assume_unique=False), pos_s, assume_unique=False)
            if common.size < 2:
                out[pop_name] = {"status": "insufficient_common_variants", "gate_pass": None}
                continue
            idx_a = np.searchsorted(pos_a, common)
            idx_b = np.searchsorted(pos_b, common)
            idx_s = np.searchsorted(pos_s, common)
            if (
                idx_a.size != common.size
                or idx_b.size != common.size
                or idx_s.size != common.size
                or not np.array_equal(pos_a[idx_a], common)
                or not np.array_equal(pos_b[idx_b], common)
                or not np.array_equal(pos_s[idx_s], common)
            ):
                out[pop_name] = {"status": "position_alignment_failed", "gate_pass": None}
                continue
            donor_a = np.asarray(donor_a_raw[:, idx_a], dtype=np.uint8, copy=False)
            donor_b = np.asarray(donor_b_raw[:, idx_b], dtype=np.uint8, copy=False)
            synth = np.asarray(synth_raw[:, idx_s], dtype=np.uint8, copy=False)
        ga = donor_a[0::2, :].astype(np.float64, copy=False) + donor_a[1::2, :].astype(np.float64, copy=False)
        gb = donor_b[0::2, :].astype(np.float64, copy=False) + donor_b[1::2, :].astype(np.float64, copy=False)
        gs = synth[0::2, :].astype(np.float64, copy=False) + synth[1::2, :].astype(np.float64, copy=False)
        g_all = np.vstack([ga, gb])
        mean = np.nanmean(g_all, axis=0)
        std = np.nanstd(g_all, axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        g_all_std = (np.where(np.isfinite(g_all), g_all, mean) - mean) / std
        gs_std = (np.where(np.isfinite(gs), gs, mean) - mean) / std
        u, s, vt = np.linalg.svd(g_all_std, full_matrices=False)
        if vt.shape[0] <= 0:
            out[pop_name] = {"status": "svd_failed", "gate_pass": None}
            continue
        pc1 = vt[0]
        pc2 = vt[1] if vt.shape[0] > 1 else np.zeros_like(pc1)
        score_a = g_all_std[: ga.shape[0], :] @ pc1
        score_b = g_all_std[ga.shape[0] :, :] @ pc1
        score_s = gs_std @ pc1
        score2_a = g_all_std[: ga.shape[0], :] @ pc2
        score2_b = g_all_std[ga.shape[0] :, :] @ pc2
        score2_s = gs_std @ pc2
        mean_a = float(np.mean(score_a))
        mean_b = float(np.mean(score_b))
        mean_s = float(np.mean(score_s))
        source_sep = abs(mean_a - mean_b)
        p_a = float(pop_spec.msprime.sources[0].weight)
        p_b = float(pop_spec.msprime.sources[1].weight)
        expected_mid = (p_a * mean_a) + (p_b * mean_b)
        plot_path: Path | None = None
        plot_error: str | None = None
        if bool(config.qc.plots_enabled) and vt.shape[0] >= 2:
            try:
                from .plots import save_pca_scatter_png

                plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_dir / f"admixed_pca_cline_{pop_name}_chr{chrom}.png"
                donor_scores = np.vstack(
                    [
                        np.column_stack([score_a, score2_a]),
                        np.column_stack([score_b, score2_b]),
                    ]
                )
                donor_labels = [source_a] * int(score_a.size) + [source_b] * int(score_b.size)
                synth_scores = np.column_stack([score_s, score2_s])
                synth_labels = [pop_name] * int(score_s.size)
                save_pca_scatter_png(
                    plot_path,
                    donor_scores=donor_scores,
                    donor_labels=donor_labels,
                    synthetic_scores=synth_scores,
                    synthetic_labels=synth_labels,
                    title=f"Admixed PCA cline chr{chrom} {pop_name}",
                )
            except Exception as exc:
                plot_error = str(exc)
                plot_path = None
        out[pop_name] = {
            "status": "ok",
            "sources": [source_a, source_b],
            "source_pc1_means": {source_a: mean_a, source_b: mean_b},
            "synthetic_pc1_mean": mean_s,
            "synthetic_pc1_std": float(np.std(score_s)),
            "synthetic_pc1_min": float(np.min(score_s)),
            "synthetic_pc1_max": float(np.max(score_s)),
            "source_pc1_separation": float(source_sep),
            "expected_weighted_midpoint_pc1": float(expected_mid),
            "midpoint_offset_abs": float(abs(mean_s - expected_mid)),
            "midpoint_offset_rel_to_source_sep": (
                float(abs(mean_s - expected_mid) / max(source_sep, 1e-12))
            ),
            "spread_rel_to_source_sep": float((np.max(score_s) - np.min(score_s)) / max(source_sep, 1e-12)),
            "n_source_a": int(score_a.size),
            "n_source_b": int(score_b.size),
            "n_synthetic": int(score_s.size),
            "plot_path": str(plot_path) if plot_path is not None else None,
            "plot_error": plot_error,
        }
    return {"enabled": True, "status": "ok", "population_summary": out}


def _compute_admixed_king_diagnostic(
    *,
    config: RunConfig,
    king_payload: dict[str, Any] | None,
    pop_names: list[str],
) -> dict[str, Any]:
    if not bool(config.qc.admixed_diag_king_enabled):
        return {"enabled": False, "status": "disabled_by_config", "population_summary": {}}
    if not isinstance(king_payload, dict) or king_payload.get("status") != "ok":
        return {"enabled": True, "status": "king_metric_unavailable", "population_summary": {}}
    summary_path = king_payload.get("summary_path")
    if summary_path is None or not Path(summary_path).exists():
        return {"enabled": True, "status": "missing_king_summary_path", "population_summary": {}}
    frame = pd.read_csv(Path(summary_path), sep="\t")
    if frame.empty:
        return {"enabled": True, "status": "empty_king_summary", "population_summary": {}}
    out: dict[str, Any] = {}
    for pop_name in pop_names:
        rows = frame[(frame["bucket"] == "within_synthetic") & (frame["population"] == pop_name)]
        if rows.empty:
            out[pop_name] = {"status": "missing_population_row"}
            continue
        row = rows.iloc[0]
        out[pop_name] = {
            "status": "ok",
            "n_pairs": int(row.get("n_pairs", 0)),
            "mean": float(row.get("mean", float("nan"))),
            "median": float(row.get("median", float("nan"))),
            "p05": float(row.get("p05", float("nan"))),
            "p95": float(row.get("p95", float("nan"))),
        }
    return {"enabled": True, "status": "ok", "population_summary": out}


def _resolve_panel_vcf_for_population(config: RunConfig, chrom: int, pop: str) -> Path:
    return config.resolve_panel_vcf_for_population(
        chrom=int(chrom),
        population=str(pop),
    )


def _resolve_donor_panel_source_for_population(
    config: RunConfig,
    chrom: int,
    pop: str,
) -> tuple[str, Path]:
    return config.resolve_donor_panel_source(
        chrom=int(chrom),
        population=str(pop),
    )


def _resolve_map_path_for_population(config: RunConfig, chrom: int, pop: str) -> Path:
    pattern = config.input.genetic_map_pattern
    candidates: list[Path] = []

    if "{population}" in pattern:
        candidates.append(Path(pattern.format(chrom=chrom, population=pop)))
        candidates.append(Path(pattern.format(chrom=chrom, population="")))
    else:
        candidates.append(Path(pattern.format(chrom=chrom)))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _subsample_ids(sample_ids: list[str], target_n: int, seed: int) -> list[str]:
    if target_n <= 0:
        raise ValueError("target_n must be > 0 for subsampling")
    unique_ids = sorted({str(x) for x in sample_ids})
    if len(unique_ids) <= target_n:
        return unique_ids
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(unique_ids), size=target_n, replace=False))
    return [unique_ids[int(i)] for i in idx.tolist()]


def _sample_variant_ids_for_ld(vcf_path: Path, chrom: int, max_variants: int, seed: int) -> list[str]:
    if max_variants <= 0:
        raise ValueError("max_variants must be > 0")

    seen: set[str] = set()
    variant_ids: list[str] = []
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        for rec in _iter_chrom_records(vcf, chrom):
            if len(rec.alleles) != 2:
                continue
            if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue
            vid = str(rec.id) if rec.id is not None else ""
            if vid in {"", "."}:
                continue
            if vid in seen:
                continue
            seen.add(vid)
            variant_ids.append(vid)

    if not variant_ids:
        raise ValueError(f"No usable variant IDs found for LD sampling in {vcf_path}")

    if len(variant_ids) <= max_variants:
        return variant_ids

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(variant_ids), size=max_variants, replace=False))
    return [variant_ids[int(i)] for i in idx.tolist()]


def _write_extract_ids(path: Path, variant_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for vid in variant_ids:
            handle.write(f"{vid}\n")


def _plink_ld_pairs_for_group(
    *,
    vcf_path: Path,
    out_prefix: Path,
    plink1_path: str,
    threads: int,
    extract_ids_path: Path,
    keep_samples: list[str] | None,
    min_maf: float,
    max_distance_cm: float,
) -> pd.DataFrame:
    cmd_base = [
        plink1_path,
        "--vcf",
        str(vcf_path),
        "--double-id",
        "--allow-extra-chr",
        "--maf",
        str(min_maf),
        "--extract",
        str(extract_ids_path),
        "--r2",
        "gz",
        "yes-really",
        "--ld-window-r2",
        "0",
        "--ld-window",
        "999999",
        "--threads",
        str(max(1, int(threads))),
        "--out",
        str(out_prefix),
    ]
    if keep_samples is not None:
        keep_path = _append_suffix(out_prefix, ".keep.txt")
        _write_keep_file(keep_path, keep_samples)
        cmd_base.extend(["--keep", str(keep_path)])

    cmd = [*cmd_base, "--ld-window-cm", str(float(max_distance_cm))]
    try:
        _run_cmd(cmd)
    except Exception as exc:
        if "--ld-window-cm" not in str(exc):
            raise
        kb = max(1, int(np.ceil(float(max_distance_cm) * 1000.0)))
        logger.warning(
            "plink1.9 --ld-window-cm unsupported; falling back to --ld-window-kb=%s",
            kb,
        )
        _run_cmd([*cmd_base, "--ld-window-kb", str(kb)])

    ld_path_gz = _append_suffix(out_prefix, ".ld.gz")
    ld_path = _append_suffix(out_prefix, ".ld")
    if ld_path_gz.exists():
        ld_path_to_read = ld_path_gz
    elif ld_path.exists():
        ld_path_to_read = ld_path
    else:
        return pd.DataFrame(columns=["BP_A", "BP_B", "R2"])

    df = pd.read_csv(ld_path_to_read, sep=r"\s+", engine="python")
    required = {"BP_A", "BP_B", "R2"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"plink LD output missing required columns {required}: {ld_path_to_read}")

    out = df[["BP_A", "BP_B", "R2"]].copy()
    out["BP_A"] = pd.to_numeric(out["BP_A"], errors="coerce")
    out["BP_B"] = pd.to_numeric(out["BP_B"], errors="coerce")
    out["R2"] = pd.to_numeric(out["R2"], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def _ld_curve_from_pairs(
    *,
    pairs_df: pd.DataFrame,
    map_bp: np.ndarray,
    map_cm: np.ndarray,
    max_distance_cm: float,
    bin_width_cm: float,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(
            columns=["bin_idx", "bin_start_cm", "bin_end_cm", "bin_mid_cm", "n_pairs", "mean_r2", "median_r2"]
        )

    bp_a = pairs_df["BP_A"].to_numpy(dtype=np.float64, copy=False)
    bp_b = pairs_df["BP_B"].to_numpy(dtype=np.float64, copy=False)
    r2 = pairs_df["R2"].to_numpy(dtype=np.float64, copy=False)
    cm_a = interpolate_cm(bp_a, map_bp, map_cm)
    cm_b = interpolate_cm(bp_b, map_bp, map_cm)
    dist_cm = np.abs(cm_b - cm_a)

    valid = np.isfinite(dist_cm) & np.isfinite(r2) & (dist_cm >= 0.0) & (dist_cm <= float(max_distance_cm))
    if not np.any(valid):
        return pd.DataFrame(
            columns=["bin_idx", "bin_start_cm", "bin_end_cm", "bin_mid_cm", "n_pairs", "mean_r2", "median_r2"]
        )

    dist_cm = dist_cm[valid]
    r2 = np.clip(r2[valid], 0.0, 1.0)
    max_bins = int(np.ceil(float(max_distance_cm) / float(bin_width_cm)))
    bin_idx = np.floor(dist_cm / float(bin_width_cm)).astype(np.int64)
    keep = (bin_idx >= 0) & (bin_idx < max_bins)
    if not np.any(keep):
        return pd.DataFrame(
            columns=["bin_idx", "bin_start_cm", "bin_end_cm", "bin_mid_cm", "n_pairs", "mean_r2", "median_r2"]
        )

    frame = pd.DataFrame(
        {
            "bin_idx": bin_idx[keep],
            "r2": r2[keep],
        }
    )
    agg = (
        frame.groupby("bin_idx", as_index=False)
        .agg(
            n_pairs=("r2", "size"),
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
        )
        .sort_values("bin_idx")
        .reset_index(drop=True)
    )
    agg["bin_start_cm"] = agg["bin_idx"].astype(float) * float(bin_width_cm)
    agg["bin_end_cm"] = (agg["bin_idx"].astype(float) + 1.0) * float(bin_width_cm)
    agg["bin_mid_cm"] = 0.5 * (agg["bin_start_cm"] + agg["bin_end_cm"])
    return agg[["bin_idx", "bin_start_cm", "bin_end_cm", "bin_mid_cm", "n_pairs", "mean_r2", "median_r2"]]


def _ld_decay_metrics_from_curves(
    *,
    donor_curve: pd.DataFrame,
    synthetic_curve: pd.DataFrame,
    min_pairs_per_bin: int,
    half_decay_r2_threshold: float,
) -> dict[str, float | int | str]:
    # Compatibility wrapper retained for existing tests/import sites.
    return utils_ld_decay_metrics_from_curves(
        donor_curve=donor_curve,
        synthetic_curve=synthetic_curve,
        min_pairs_per_bin=min_pairs_per_bin,
        half_decay_r2_threshold=half_decay_r2_threshold,
    )


def _ld_curve_from_genotypes_python(
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
    max_neighbors_per_anchor: int,
    numba_threads: int | None,
) -> pd.DataFrame:
    if local_ld_curve_from_genotypes is not None:
        return local_ld_curve_from_genotypes(
            genotypes=genotypes,
            variant_positions_bp=variant_positions_bp,
            map_bp=map_bp,
            map_cm=map_cm,
            max_distance_cm=max_distance_cm,
            bin_width_cm=bin_width_cm,
            min_maf=min_maf,
            seed=seed,
            max_pairs_per_bin=max_pairs_per_bin,
            max_neighbors_per_anchor=max_neighbors_per_anchor,
            numba_threads=numba_threads,
        )
    if utils_ld_curve_from_genotypes is None:
        raise ImportError("No LD curve implementation is available for python LD-decay QC.")
    return utils_ld_curve_from_genotypes(
        genotypes=genotypes,
        variant_positions_bp=variant_positions_bp,
        map_bp=map_bp,
        map_cm=map_cm,
        max_distance_cm=max_distance_cm,
        bin_width_cm=bin_width_cm,
        min_maf=min_maf,
        seed=seed,
        max_pairs_per_bin=max_pairs_per_bin,
        max_neighbors_per_anchor=max_neighbors_per_anchor,
    )


def _haplotypes_to_genotypes(haplotypes: np.ndarray) -> np.ndarray:
    h = np.asarray(haplotypes, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"haplotypes must be 2D, got shape={h.shape}")
    if h.shape[0] % 2 != 0:
        raise ValueError(f"haplotype row count must be even, got shape={h.shape}")
    return h[0::2, :] + h[1::2, :]


def _load_sample_haplotypes_from_panel(
    *,
    config: RunConfig,
    vcf_path: Path,
    chrom: int,
    sample_ids: list[str],
    source_kind: str = "vcf",
) -> tuple[np.ndarray, np.ndarray]:
    panel, _, _ = load_donor_haplotypes_cached(
        source_path=vcf_path,
        source_kind=source_kind,
        chrom=chrom,
        donor_ids=sample_ids,
        cache_dir=config.donor_cache_root,
        force_refresh=False,
    )
    return panel.variants.pos.astype(np.int64, copy=False), panel.donor_haps


def _load_synthetic_population_haplotypes(
    *,
    config: RunConfig,
    inputs: _QcInputs,
    chrom: int,
    pop: str,
    sample_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if inputs.combined_hapstore is not None:
        hs = HapStore(inputs.combined_hapstore, mode="r")
        all_ids = [str(x) for x in hs.sample_ids(chrom)]
        idx_by_id = {sid: idx for idx, sid in enumerate(all_ids)}
        missing = [sid for sid in sample_ids if sid not in idx_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} synthetic samples for {pop} missing in combined hapstore; first={missing[:5]}"
            )
        requested_idx = np.asarray([idx_by_id[sid] for sid in sample_ids], dtype=np.int64)
        mask = np.zeros(len(all_ids), dtype=bool)
        mask[requested_idx] = True
        haps = hs.haplotypes(chrom, sample_mask=mask)

        # HapStore returns selected samples in store-index order; reorder to match `sample_ids`.
        selected_idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        pos = np.searchsorted(selected_idx, requested_idx)
        if pos.size != requested_idx.size or not np.array_equal(selected_idx[pos], requested_idx):
            raise RuntimeError("Unable to reorder combined hapstore haplotypes to requested sample order")
        hap_rows = np.empty(pos.size * 2, dtype=np.int64)
        hap_rows[0::2] = 2 * pos
        hap_rows[1::2] = (2 * pos) + 1
        return hs.variant_positions(chrom), haps[hap_rows, :]

    if pop in inputs.population_hapstores:
        hs = HapStore(inputs.population_hapstores[pop], mode="r")
        all_ids = [str(x) for x in hs.sample_ids(chrom)]
        idx_by_id = {sid: idx for idx, sid in enumerate(all_ids)}
        missing = [sid for sid in sample_ids if sid not in idx_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} synthetic samples for {pop} missing in population hapstore; first={missing[:5]}"
            )
        requested_idx = np.asarray([idx_by_id[sid] for sid in sample_ids], dtype=np.int64)
        mask = np.zeros(len(all_ids), dtype=bool)
        mask[requested_idx] = True
        haps = hs.haplotypes(chrom, sample_mask=mask)

        selected_idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        pos = np.searchsorted(selected_idx, requested_idx)
        if pos.size != requested_idx.size or not np.array_equal(selected_idx[pos], requested_idx):
            raise RuntimeError("Unable to reorder population hapstore haplotypes to requested sample order")
        hap_rows = np.empty(pos.size * 2, dtype=np.int64)
        hap_rows[0::2] = 2 * pos
        hap_rows[1::2] = (2 * pos) + 1
        return hs.variant_positions(chrom), haps[hap_rows, :]

    if inputs.combined_vcf is not None:
        return _load_sample_haplotypes_from_panel(
            config=config,
            vcf_path=inputs.combined_vcf,
            chrom=chrom,
            sample_ids=sample_ids,
        )

    if pop not in inputs.population_vcfs:
        raise ValueError(f"Synthetic population {pop!r} not available in QC inputs")
    return _load_sample_haplotypes_from_panel(
        config=config,
        vcf_path=inputs.population_vcfs[pop],
        chrom=chrom,
        sample_ids=sample_ids,
    )


def _align_haplotype_axes_by_position(
    donor_pos: np.ndarray,
    donor_haps: np.ndarray,
    synth_pos: np.ndarray,
    synth_haps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor_pos = np.asarray(donor_pos, dtype=np.int64)
    synth_pos = np.asarray(synth_pos, dtype=np.int64)
    if donor_haps.shape[1] != donor_pos.shape[0]:
        raise ValueError(
            "donor haplotype axis mismatch: "
            f"haps={donor_haps.shape}, n_positions={donor_pos.shape[0]}"
        )
    if synth_haps.shape[1] != synth_pos.shape[0]:
        raise ValueError(
            "synthetic haplotype axis mismatch: "
            f"haps={synth_haps.shape}, n_positions={synth_pos.shape[0]}"
        )

    if donor_pos.shape == synth_pos.shape and np.array_equal(donor_pos, synth_pos):
        return donor_pos, donor_haps, synth_haps

    common, idx_donor, idx_synth = np.intersect1d(
        donor_pos,
        synth_pos,
        assume_unique=False,
        return_indices=True,
    )
    if common.shape[0] < 2:
        raise ValueError(
            "Unable to align donor/synthetic variant axes by position; "
            f"common_positions={common.shape[0]}"
        )
    return (
        common.astype(np.int64, copy=False),
        donor_haps[:, idx_donor],
        synth_haps[:, idx_synth],
    )


def _compute_ld_decay_qc_python(
    *,
    config: RunConfig,
    chrom: int,
    inputs: _QcInputs,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
) -> tuple[dict[str, Any], Path]:
    from .metrics.ld_heatmap import ld_heatmap_binned_from_genotypes
    from .plots import save_ld_heatmap_pair_png

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
        donor_ids = _subsample_ids(
            donor_ids_all,
            target_n=donor_n,
            seed=derived_seed(int(config.qc.seed), "ld_donor", pop, chrom),
        )
        synth_ids = _subsample_ids(
            synth_ids_all,
            target_n=synth_n,
            seed=derived_seed(int(config.qc.seed), "ld_synth", pop, chrom),
        )

        donor_source_kind, donor_source_path = _resolve_donor_panel_source_for_population(
            config,
            chrom=chrom,
            pop=pop,
        )
        map_path = _resolve_map_path_for_population(config, chrom=chrom, pop=pop)
        map_bp, map_cm = load_genetic_map(map_path)

        donor_pos, donor_haps = _load_sample_haplotypes_from_panel(
            config=config,
            vcf_path=donor_source_path,
            chrom=chrom,
            sample_ids=donor_ids,
            source_kind=donor_source_kind,
        )
        synth_pos, synth_haps = _load_synthetic_population_haplotypes(
            config=config,
            inputs=inputs,
            chrom=chrom,
            pop=pop,
            sample_ids=synth_ids,
        )
        aligned_pos, donor_haps_aligned, synth_haps_aligned = _align_haplotype_axes_by_position(
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
        donor_curve = _ld_curve_from_genotypes_python(
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
        synth_curve = _ld_curve_from_genotypes_python(
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

        metrics = _ld_decay_metrics_from_curves(
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

    gate_pass: bool | None
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


def _compute_ld_decay_qc_plink(
    *,
    config: RunConfig,
    chrom: int,
    inputs: _QcInputs,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
) -> tuple[dict[str, Any], Path]:
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

    ld_sample_size = int(config.qc.ext_max_diploids_per_population)
    ld_max_variants = int(config.qc.ext_max_variants)

    if inputs.combined_vcf is not None:
        variant_sampling_vcf = inputs.combined_vcf
    elif inputs.population_vcfs:
        variant_sampling_vcf = inputs.population_vcfs[pop_names[0]]
    else:
        raise ValueError(
            "qc.ld_decay_backend=plink requires synthetic VCF outputs; "
            "enable outputs.write_phased_vcf for smoke comparison"
        )

    variant_ids = _sample_variant_ids_for_ld(
        vcf_path=variant_sampling_vcf,
        chrom=chrom,
        max_variants=int(ld_max_variants),
        seed=derived_seed(int(config.qc.seed), "ld_variants", chrom),
    )

    table_rows: list[dict[str, Any]] = []
    pop_summary: dict[str, dict[str, Any]] = {}
    n_fail = 0
    n_eval = 0

    with tempfile.TemporaryDirectory(prefix=f"popamp_ld_qc_chr{chrom}_", dir=str(config.run_root)) as tmp_root:
        tmp_dir = Path(tmp_root)
        extract_path = tmp_dir / f"ld_extract_chr{chrom}.txt"
        _write_extract_ids(extract_path, variant_ids)

        for pop in pop_names:
            if pop not in donors_by_population:
                pop_summary[pop] = {"status": "missing_donor_population", "pass": None}
                continue

            donor_ids_all = donors_by_population[pop]
            synth_ids_all = groups[pop]
            donor_n = min(len(donor_ids_all), int(ld_sample_size))
            synth_n = min(len(synth_ids_all), int(ld_sample_size))
            donor_ids = _subsample_ids(
                donor_ids_all,
                target_n=donor_n,
                seed=derived_seed(int(config.qc.seed), "ld_donor", pop, chrom),
            )
            synth_ids = _subsample_ids(
                synth_ids_all,
                target_n=synth_n,
                seed=derived_seed(int(config.qc.seed), "ld_synth", pop, chrom),
            )

            donor_vcf = _resolve_panel_vcf_for_population(config, chrom=chrom, pop=pop)
            if inputs.combined_vcf is not None:
                synthetic_vcf = inputs.combined_vcf
            else:
                if pop not in inputs.population_vcfs:
                    raise ValueError(
                        "qc.ld_decay_backend=plink requires synthetic population VCFs; "
                        f"population {pop!r} is missing"
                    )
                synthetic_vcf = inputs.population_vcfs[pop]
            map_path = _resolve_map_path_for_population(config, chrom=chrom, pop=pop)
            map_bp, map_cm = load_genetic_map(map_path)

            donor_pairs = _plink_ld_pairs_for_group(
                vcf_path=donor_vcf,
                out_prefix=tmp_dir / f"{pop}.donor",
                plink1_path=config.outputs.plink1_path,
                threads=config.qc.plink_threads,
                extract_ids_path=extract_path,
                keep_samples=donor_ids,
                min_maf=float(config.qc.ld_decay_min_maf),
                max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
            )
            synth_pairs = _plink_ld_pairs_for_group(
                vcf_path=synthetic_vcf,
                out_prefix=tmp_dir / f"{pop}.synthetic",
                plink1_path=config.outputs.plink1_path,
                threads=config.qc.plink_threads,
                extract_ids_path=extract_path,
                keep_samples=synth_ids,
                min_maf=float(config.qc.ld_decay_min_maf),
                max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
            )

            donor_curve = _ld_curve_from_pairs(
                pairs_df=donor_pairs,
                map_bp=map_bp,
                map_cm=map_cm,
                max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
                bin_width_cm=float(config.qc.ld_decay_bin_width_cm),
            )
            synth_curve = _ld_curve_from_pairs(
                pairs_df=synth_pairs,
                map_bp=map_bp,
                map_cm=map_cm,
                max_distance_cm=float(config.qc.ld_decay_max_distance_cm),
                bin_width_cm=float(config.qc.ld_decay_bin_width_cm),
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

            metrics = _ld_decay_metrics_from_curves(
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
                **metrics_no_auc,
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

    gate_pass: bool | None
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
            "source": str(variant_sampling_vcf),
            "n_variant_ids": int(len(variant_ids)),
            "max_variants": int(ld_max_variants),
            "min_maf": float(config.qc.ld_decay_min_maf),
            "max_distance_cm": float(config.qc.ld_decay_max_distance_cm),
            "bin_width_cm": float(config.qc.ld_decay_bin_width_cm),
            "min_pairs_per_bin": int(config.qc.ld_decay_min_pairs_per_bin),
            "seed": int(config.qc.seed),
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


def _compute_ld_decay_qc(
    *,
    config: RunConfig,
    chrom: int,
    inputs: _QcInputs,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
) -> tuple[dict[str, Any], Path]:
    if config.qc.ld_decay_backend == "plink":
        return _compute_ld_decay_qc_plink(
            config=config,
            chrom=chrom,
            inputs=inputs,
            groups=groups,
            diagnostics_dir=diagnostics_dir,
        )
    return _compute_ld_decay_qc_python(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=numba_threads,
    )


def _even_folds(sample_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(sample_ids) < n_folds:
        raise ValueError(f"n_folds={n_folds} exceeds available donors={len(sample_ids)}")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sample_ids, dtype=object)
    rng.shuffle(shuffled)
    return [chunk.tolist() for chunk in np.array_split(shuffled, n_folds) if len(chunk) > 0]


def _sample_groups_from_manifest(sample_manifest_path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(sample_manifest_path, sep="\t", dtype=str)
    if "population" not in df.columns or "sample_id" not in df.columns:
        raise ValueError(f"sample manifest missing population/sample_id columns: {sample_manifest_path}")
    groups: dict[str, list[str]] = {}
    for pop, group in df.groupby("population"):
        groups[str(pop)] = sorted(group["sample_id"].astype(str).unique().tolist())
    return groups


def _iter_chrom_records(vcf: pysam.VariantFile, chrom: int | str):
    chrom_s = str(chrom)
    aliases = [chrom_s]
    if chrom_s.startswith("chr"):
        aliases.append(chrom_s[3:])
    else:
        aliases.append(f"chr{chrom_s}")
    aliases = [a for i, a in enumerate(aliases) if a and a not in aliases[:i]]

    for alias in aliases:
        try:
            found = False
            for rec in vcf.fetch(alias):
                found = True
                yield rec
            if found:
                return
        except (ValueError, OSError):
            continue

    alias_set = set(aliases)
    for rec in vcf:
        if rec.contig in alias_set:
            yield rec


def _hudson_fst_from_vcf(vcf_path: Path, chrom: int, groups: dict[str, list[str]], pairs: list[tuple[str, str]]) -> dict[str, float]:
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        header_samples = set(vcf.header.samples)
        for pop, ids in groups.items():
            missing = [sid for sid in ids if sid not in header_samples]
            if missing:
                raise ValueError(f"{len(missing)} {pop} samples missing in {vcf_path}; first={missing[:5]}")

        numer = {pair: 0.0 for pair in pairs}
        denom = {pair: 0.0 for pair in pairs}

        for rec in _iter_chrom_records(vcf, chrom):
            if len(rec.alleles) != 2:
                continue
            if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue

            pop_stats: dict[str, tuple[float, float]] = {}
            for pop, ids in groups.items():
                alt_count = 0.0
                allele_count = 0.0
                for sid in ids:
                    gt = rec.samples[sid].get("GT")
                    if not gt:
                        continue
                    for allele in gt[:2]:
                        if allele is None:
                            continue
                        allele_count += 1.0
                        if allele > 0:
                            alt_count += 1.0
                pop_stats[pop] = (alt_count, allele_count)

            for pair in pairs:
                a, b = pair
                ac_a, an_a = pop_stats[a]
                ac_b, an_b = pop_stats[b]
                if an_a < 4 or an_b < 4:
                    continue

                # Match PLINK Hudson finite-sample correction: n is chromosome count (AN).
                n1 = an_a
                n2 = an_b
                p1 = ac_a / an_a
                p2 = ac_b / an_b
                if n1 <= 1 or n2 <= 1:
                    continue

                num = (p1 - p2) ** 2 - (p1 * (1.0 - p1) / (n1 - 1.0)) - (p2 * (1.0 - p2) / (n2 - 1.0))
                den = p1 * (1.0 - p2) + p2 * (1.0 - p1)
                if den <= 0:
                    continue
                numer[pair] += num
                denom[pair] += den

    out: dict[str, float] = {}
    for pair in pairs:
        key = _pair_key(*pair)
        if denom[pair] == 0:
            out[key] = float("nan")
        else:
            out[key] = float(numer[pair] / denom[pair])
    return out


def _population_allele_stats_from_vcf(
    vcf_path: Path,
    chrom: int,
) -> tuple[list[tuple[str, int, str, str]], np.ndarray, np.ndarray]:
    keys: list[tuple[str, int, str, str]] = []
    ac: list[float] = []
    an: list[float] = []

    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        sample_ids = list(vcf.header.samples)
        for rec in _iter_chrom_records(vcf, chrom):
            if len(rec.alleles) != 2:
                continue
            if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue

            alt_count = 0.0
            allele_count = 0.0
            for sid in sample_ids:
                gt = rec.samples[sid].get("GT")
                if not gt:
                    continue
                for allele in gt[:2]:
                    if allele is None:
                        continue
                    allele_count += 1.0
                    if allele > 0:
                        alt_count += 1.0

            keys.append((str(rec.contig), int(rec.pos), rec.ref, rec.alts[0]))
            ac.append(alt_count)
            an.append(allele_count)

    if not keys:
        raise ValueError(f"No eligible biallelic SNPs found in population VCF {vcf_path}")

    return keys, np.asarray(ac, dtype=np.float64), np.asarray(an, dtype=np.float64)


def _subset_allele_stats_from_vcf(
    vcf_path: Path,
    chrom: int,
    sample_ids: list[str],
) -> tuple[list[tuple[str, int, str, str]], np.ndarray, np.ndarray]:
    if not sample_ids:
        raise ValueError("sample_ids cannot be empty for subset allele stats")

    keys: list[tuple[str, int, str, str]] = []
    ac: list[float] = []
    an: list[float] = []

    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        header_samples = set(vcf.header.samples)
        missing = [sid for sid in sample_ids if sid not in header_samples]
        if missing:
            raise ValueError(
                f"{len(missing)} samples missing in {vcf_path}; first={missing[:5]}"
            )

        for rec in _iter_chrom_records(vcf, chrom):
            if len(rec.alleles) != 2:
                continue
            if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue

            alt_count = 0.0
            allele_count = 0.0
            for sid in sample_ids:
                gt = rec.samples[sid].get("GT")
                if not gt:
                    continue
                for allele in gt[:2]:
                    if allele is None:
                        continue
                    allele_count += 1.0
                    if allele > 0:
                        alt_count += 1.0

            keys.append((str(rec.contig), int(rec.pos), rec.ref, rec.alts[0]))
            ac.append(alt_count)
            an.append(allele_count)

    if not keys:
        raise ValueError(f"No eligible biallelic SNPs found in subset VCF read {vcf_path}")

    return keys, np.asarray(ac, dtype=np.float64), np.asarray(an, dtype=np.float64)


def _hudson_fst_from_counts(ac_a: np.ndarray, an_a: np.ndarray, ac_b: np.ndarray, an_b: np.ndarray) -> float:
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


def _first_marker_mismatch(
    a: list[tuple[str, int, str, str]],
    b: list[tuple[str, int, str, str]],
) -> tuple[tuple[str, int, str, str], tuple[str, int, str, str]] | None:
    for ka, kb in zip(a, b):
        if ka != kb:
            return ka, kb
    return None


def _run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _write_keep_file(path: Path, sample_ids: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for sid in sample_ids:
            sid_s = str(sid)
            handle.write(f"{sid_s}\t{sid_s}\n")


def _write_pheno_file(path: Path, sample_to_group: dict[str, str], column_name: str = "POP") -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"FID\tIID\t{column_name}\n")
        for sid in sorted(sample_to_group.keys()):
            sid_s = str(sid)
            group = str(sample_to_group[sid])
            handle.write(f"{sid_s}\t{sid_s}\t{group}\n")


def _write_pair_file(path: Path, pairs: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for a, b in pairs:
            handle.write(f"{a}\t{b}\n")


def _parse_alt_counts(series: pd.Series) -> np.ndarray:
    # plink2 ALT_CTS can be a comma-separated string; for biallelic sites take first ALT count.
    def _one(value: object) -> float:
        token = str(value).split(",")[0]
        return float(token)

    return series.map(_one).to_numpy(dtype=np.float64)


def _load_plink_acount(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=r"\s+", engine="python")

    chrom_col = "#CHROM" if "#CHROM" in df.columns else "CHROM"
    if chrom_col not in df.columns:
        raise ValueError(f"plink .acount missing CHROM column: {path}")
    if "ID" not in df.columns:
        raise ValueError(f"plink .acount missing ID column: {path}")

    if "ALT_CTS" in df.columns:
        ac = _parse_alt_counts(df["ALT_CTS"])
    elif "ALT_CT" in df.columns:
        ac = pd.to_numeric(df["ALT_CT"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"plink .acount missing ALT count column (ALT_CTS/ALT_CT): {path}")

    if "OBS_CT" in df.columns:
        an = pd.to_numeric(df["OBS_CT"], errors="coerce").to_numpy(dtype=np.float64)
    elif "NCHROBS" in df.columns:
        an = pd.to_numeric(df["NCHROBS"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"plink .acount missing allele-count denominator (OBS_CT/NCHROBS): {path}")

    keys = (df[chrom_col].astype(str) + ":" + df["ID"].astype(str)).to_numpy(dtype=object)
    return keys, ac, an


def _load_plink_fst_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Expected plink2 .fst.summary output not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", engine="python")
    if df.empty:
        return {}

    pop1_col = "POP1"
    pop2_col = "POP2"
    if "#POP1" in df.columns:
        pop1_col = "#POP1"
    if "#POP2" in df.columns:
        pop2_col = "#POP2"
    if pop1_col not in df.columns or pop2_col not in df.columns:
        raise ValueError(f"plink .fst.summary missing POP1/POP2 columns: {path}")

    fst_col = None
    for candidate in ("HUDSON_FST", "WC_FST", "WEIR_COCKERHAM_FST", "FST"):
        if candidate in df.columns:
            fst_col = candidate
            break
    if fst_col is None:
        raise ValueError(f"plink .fst.summary missing FST metric column: {path}")

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        key = _pair_key(str(row[pop1_col]), str(row[pop2_col]))
        out[key] = float(pd.to_numeric(row[fst_col], errors="coerce"))
    return out


def _hudson_fst_from_acount(
    keys_a: np.ndarray,
    ac_a: np.ndarray,
    an_a: np.ndarray,
    keys_b: np.ndarray,
    ac_b: np.ndarray,
    an_b: np.ndarray,
) -> float:
    idx_a = {str(key): i for i, key in enumerate(keys_a.tolist())}
    common_idx_a: list[int] = []
    common_idx_b: list[int] = []
    for j, key in enumerate(keys_b.tolist()):
        i = idx_a.get(str(key))
        if i is None:
            continue
        common_idx_a.append(i)
        common_idx_b.append(j)

    if not common_idx_a:
        return float("nan")

    a_idx = np.asarray(common_idx_a, dtype=np.int64)
    b_idx = np.asarray(common_idx_b, dtype=np.int64)
    return _hudson_fst_from_counts(
        ac_a[a_idx],
        an_a[a_idx],
        ac_b[b_idx],
        an_b[b_idx],
    )


def _plink_acount_for_group(
    *,
    vcf_path: Path,
    out_prefix: Path,
    plink2_path: str,
    threads: int,
    keep_samples: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cmd = [
        plink2_path,
        "--vcf",
        str(vcf_path),
        "--double-id",
        "--allow-extra-chr",
        "--freq",
        "counts",
        "--threads",
        str(threads),
        "--out",
        str(out_prefix),
    ]

    keep_path = None
    if keep_samples is not None:
        keep_path = _append_suffix(out_prefix, ".keep.txt")
        _write_keep_file(keep_path, keep_samples)
        cmd.extend(["--keep", str(keep_path)])

    _run_cmd(cmd)
    acount_path = _append_suffix(out_prefix, ".acount")
    if not acount_path.exists():
        raise FileNotFoundError(f"Expected plink2 .acount output not found: {acount_path}")
    return _load_plink_acount(acount_path)


def _merge_population_vcfs_to_temp(
    *,
    config: RunConfig,
    population_vcfs: dict[str, Path],
    output_vcf: Path,
) -> Path:
    vcf_paths = [population_vcfs[pop] for pop in sorted(population_vcfs.keys())]
    if not vcf_paths:
        raise ValueError("population_vcfs cannot be empty")

    if len(vcf_paths) == 1:
        shutil.copy2(vcf_paths[0], output_vcf)
        src_tbi = Path(f"{vcf_paths[0]}.tbi")
        if src_tbi.exists():
            shutil.copy2(src_tbi, Path(f"{output_vcf}.tbi"))
        return output_vcf

    use_bcftools = bool(shutil.which(config.outputs.bcftools_path))
    if use_bcftools:
        try:
            merge_shard_vcfs_bcftools(
                shard_vcfs=vcf_paths,
                output_vcf_gz=output_vcf,
                bcftools_path=config.outputs.bcftools_path,
                threads=max(1, config.qc.plink_threads),
                no_index=config.outputs.bcftools_no_index,
            )
            return output_vcf
        except Exception:
            logger.warning("bcftools merge failed during temp QC merge; falling back to python merge")

    merge_shard_vcfs(shard_vcfs=vcf_paths, output_vcf_gz=output_vcf)
    return output_vcf


def _plink_fst_for_groups(
    *,
    vcf_path: Path,
    out_prefix: Path,
    plink2_path: str,
    threads: int,
    sample_to_group: dict[str, str],
    pairs: list[tuple[str, str]] | None = None,
    keep_samples: list[str] | None = None,
) -> dict[str, float]:
    if not sample_to_group:
        raise ValueError("sample_to_group cannot be empty for plink fst")

    pheno_path = _append_suffix(out_prefix, ".pheno.tsv")
    _write_pheno_file(pheno_path, sample_to_group, column_name="POP")

    requested_pairs: list[tuple[str, str]] | None = None
    fst_args: list[str] = ["--fst", "POP", "method=hudson"]
    if pairs:
        requested_pairs = [(str(a), str(b)) for a, b in pairs]
        pair_path = _append_suffix(out_prefix, ".pairs.txt")
        _write_pair_file(pair_path, requested_pairs)
        fst_args.append(f"file={pair_path}")

    cmd = [
        plink2_path,
        "--vcf",
        str(vcf_path),
        "--double-id",
        "--allow-extra-chr",
        "--pheno",
        str(pheno_path),
        "--pheno-name",
        "POP",
        *fst_args,
        "--threads",
        str(threads),
        "--out",
        str(out_prefix),
    ]

    if keep_samples is not None:
        keep_path = _append_suffix(out_prefix, ".keep.txt")
        _write_keep_file(keep_path, keep_samples)
        cmd.extend(["--keep", str(keep_path)])

    _run_cmd(cmd)
    parsed = _load_plink_fst_summary(_append_suffix(out_prefix, ".fst.summary"))
    if not requested_pairs:
        return parsed

    out: dict[str, float] = {}
    for a, b in requested_pairs:
        key = _pair_key(a, b)
        out[key] = parsed.get(key, float("nan"))
    return out


def _plink_fst_for_groups_maybe_pair_parallel(
    *,
    vcf_path: Path,
    out_prefix: Path,
    plink2_path: str,
    total_threads: int,
    sample_to_group: dict[str, str],
    pairs: list[tuple[str, str]],
    keep_samples: list[str] | None,
    pair_parallel: bool,
    min_threads_per_match: int | None,
) -> dict[str, float]:
    if not pairs:
        return {}

    budget_threads = max(1, int(total_threads))
    if (not pair_parallel) or min_threads_per_match is None:
        return _plink_fst_for_groups(
            vcf_path=vcf_path,
            out_prefix=out_prefix,
            plink2_path=plink2_path,
            threads=budget_threads,
            sample_to_group=sample_to_group,
            pairs=pairs,
            keep_samples=keep_samples,
        )

    threads_per_match = max(1, int(min_threads_per_match))
    threads_per_match = min(threads_per_match, budget_threads)
    max_parallel_matches = max(1, budget_threads // threads_per_match)

    def _run_one(idx_pair: tuple[int, tuple[str, str]]) -> tuple[str, float]:
        idx, pair = idx_pair
        a, b = pair
        pair_key = _pair_key(a, b)
        pair_prefix = out_prefix.parent / f"{out_prefix.name}.pair_{idx:04d}_{pair_key}"
        pair_out = _plink_fst_for_groups(
            vcf_path=vcf_path,
            out_prefix=pair_prefix,
            plink2_path=plink2_path,
            threads=threads_per_match,
            sample_to_group=sample_to_group,
            pairs=[pair],
            keep_samples=keep_samples,
        )
        return pair_key, float(pair_out.get(pair_key, float("nan")))

    out: dict[str, float] = {}
    if max_parallel_matches == 1 or len(pairs) == 1:
        for idx_pair in enumerate(pairs):
            key, value = _run_one(idx_pair)
            out[key] = value
        return out

    with ThreadPoolExecutor(max_workers=min(max_parallel_matches, len(pairs))) as pool:
        futures = [pool.submit(_run_one, x) for x in enumerate(pairs)]
        for fut in as_completed(futures):
            key, value = fut.result()
            out[key] = value
    return out


def _run_qc_plink_native_fst(
    config: RunConfig,
    chrom: int,
    inputs: _QcInputs,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    within_checks: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    with tempfile.TemporaryDirectory(prefix=f"popamp_qc_native_fst_chr{chrom}_", dir=str(config.run_root)) as tmp_root:
        tmp_dir = Path(tmp_root)

        if inputs.combined_vcf is not None:
            pairwise_vcf = inputs.combined_vcf
        else:
            pairwise_vcf = tmp_dir / f"synthetic_pairwise_qc_chr{chrom}.vcf.gz"
            _merge_population_vcfs_to_temp(
                config=config,
                population_vcfs=inputs.population_vcfs,
                output_vcf=pairwise_vcf,
            )

        sample_to_pop = {sid: pop for pop, ids in groups.items() for sid in ids}
        pairwise = _plink_fst_for_groups(
            vcf_path=pairwise_vcf,
            out_prefix=tmp_dir / "pairwise",
            plink2_path=config.outputs.plink2_path,
            threads=config.qc.plink_threads,
            sample_to_group=sample_to_pop,
            pairs=pairs,
            keep_samples=sorted(sample_to_pop.keys()),
        )

        within: dict[str, float] = {}
        for pop in within_checks:
            sample_ids = groups[pop]
            if len(sample_ids) < 4:
                within[pop] = float("nan")
                continue

            rng = np.random.default_rng(42)
            shuffled = np.array(sample_ids, dtype=object)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            split_a = shuffled[:half].tolist()
            split_b = shuffled[half:].tolist()

            split_groups = {sid: "A" for sid in split_a}
            split_groups.update({sid: "B" for sid in split_b})
            split_out = _plink_fst_for_groups(
                vcf_path=pairwise_vcf,
                out_prefix=tmp_dir / f"within_{pop}",
                plink2_path=config.outputs.plink2_path,
                threads=config.qc.plink_threads,
                sample_to_group=split_groups,
                pairs=[("A", "B")],
                keep_samples=sorted(split_groups.keys()),
            )
            within[pop] = split_out.get(_pair_key("A", "B"), float("nan"))

        return pairwise, within


def _run_qc_plink_counts(
    config: RunConfig,
    chrom: int,
    inputs: _QcInputs,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    within_checks: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    with tempfile.TemporaryDirectory(prefix=f"popamp_qc_chr{chrom}_", dir=str(config.run_root)) as tmp_root:
        tmp_dir = Path(tmp_root)

        population_counts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if inputs.combined_vcf is not None:
            # Combined merged VCF: each population is a sample subset of the same file.
            for pop, ids in groups.items():
                out_prefix = tmp_dir / f"combined_{pop}"
                population_counts[pop] = _plink_acount_for_group(
                    vcf_path=inputs.combined_vcf,
                    out_prefix=out_prefix,
                    plink2_path=config.outputs.plink2_path,
                    threads=config.qc.plink_threads,
                    keep_samples=ids,
                )
        else:
            # Population-merged VCFs: each population has its own merged VCF.
            for pop in sorted(groups.keys()):
                if pop not in inputs.population_vcfs:
                    raise ValueError(f"Population {pop} missing in population_vcf_paths for plink QC")
                out_prefix = tmp_dir / f"pop_{pop}"
                population_counts[pop] = _plink_acount_for_group(
                    vcf_path=inputs.population_vcfs[pop],
                    out_prefix=out_prefix,
                    plink2_path=config.outputs.plink2_path,
                    threads=config.qc.plink_threads,
                    keep_samples=None,
                )

        pairwise: dict[str, float] = {}
        for a, b in pairs:
            key_a, ac_a, an_a = population_counts[a]
            key_b, ac_b, an_b = population_counts[b]
            pairwise[_pair_key(a, b)] = _hudson_fst_from_acount(
                key_a,
                ac_a,
                an_a,
                key_b,
                ac_b,
                an_b,
            )

        within: dict[str, float] = {}
        for pop in within_checks:
            sample_ids = groups[pop]
            if len(sample_ids) < 4:
                within[pop] = float("nan")
                continue
            rng = np.random.default_rng(42)
            shuffled = np.array(sample_ids, dtype=object)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            split_a = shuffled[:half].tolist()
            split_b = shuffled[half:].tolist()

            if inputs.combined_vcf is not None:
                vcf_path = inputs.combined_vcf
            else:
                if pop not in inputs.population_vcfs:
                    raise ValueError(f"within_pop_split_checks requested unknown population {pop}")
                vcf_path = inputs.population_vcfs[pop]

            out_a = tmp_dir / f"within_{pop}_A"
            out_b = tmp_dir / f"within_{pop}_B"
            key_a, ac_a, an_a = _plink_acount_for_group(
                vcf_path=vcf_path,
                out_prefix=out_a,
                plink2_path=config.outputs.plink2_path,
                threads=config.qc.plink_threads,
                keep_samples=split_a,
            )
            key_b, ac_b, an_b = _plink_acount_for_group(
                vcf_path=vcf_path,
                out_prefix=out_b,
                plink2_path=config.outputs.plink2_path,
                threads=config.qc.plink_threads,
                keep_samples=split_b,
            )
            within[pop] = _hudson_fst_from_acount(
                key_a,
                ac_a,
                an_a,
                key_b,
                ac_b,
                an_b,
            )

        return pairwise, within


def _hudson_fst_from_population_vcfs(
    population_vcfs: dict[str, Path],
    chrom: int,
    pairs: list[tuple[str, str]],
) -> dict[str, float]:
    needed_pops = sorted({pop for pair in pairs for pop in pair})
    stats: dict[str, tuple[list[tuple[str, int, str, str]], np.ndarray, np.ndarray]] = {}
    for pop in needed_pops:
        if pop not in population_vcfs:
            raise ValueError(f"Population {pop} missing from merged population VCF inputs")
        stats[pop] = _population_allele_stats_from_vcf(population_vcfs[pop], chrom=chrom)

    out: dict[str, float] = {}
    for a, b in pairs:
        key_a, ac_a, an_a = stats[a]
        key_b, ac_b, an_b = stats[b]
        if len(key_a) != len(key_b):
            raise ValueError(
                f"Marker count mismatch between populations {a} ({len(key_a)}) and {b} ({len(key_b)})"
            )
        if key_a != key_b:
            mismatch = _first_marker_mismatch(key_a, key_b)
            if mismatch is None:
                raise ValueError(f"Marker ordering mismatch between populations {a} and {b}")
            raise ValueError(
                f"Marker mismatch between populations {a} and {b}: {mismatch[0]} != {mismatch[1]}"
            )

        out[_pair_key(a, b)] = _hudson_fst_from_counts(ac_a, an_a, ac_b, an_b)

    return out


def _within_split_fst(vcf_path: Path, chrom: int, pop: str, sample_ids: list[str], seed: int = 42) -> float:
    if len(sample_ids) < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    shuffled = np.array(sample_ids, dtype=object)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    split_a = shuffled[:half].tolist()
    split_b = shuffled[half:].tolist()
    pair = (f"{pop}_A", f"{pop}_B")
    fst = _hudson_fst_from_vcf(
        vcf_path=vcf_path,
        chrom=chrom,
        groups={pair[0]: split_a, pair[1]: split_b},
        pairs=[pair],
    )
    return fst[_pair_key(*pair)]


def _resolve_synthetic_outputs(config: RunConfig, chrom: int, n_shards: int) -> _QcInputs:
    merged_manifest = config.run_root / "manifests" / f"merged_samples_chr{chrom}.tsv"
    merged_vcf = config.run_root / "merged" / f"chr{chrom}" / f"combined.chr{chrom}.merged.vcf.gz"
    merged_hapstore = config.run_root / "merged" / f"chr{chrom}" / f"combined.chr{chrom}.merged.hapstore.zarr"
    if merged_hapstore.exists() and merged_manifest.exists():
        return _QcInputs(
            mode="combined_merged_hapstore",
            sample_manifest=merged_manifest,
            combined_vcf=merged_vcf if merged_vcf.exists() else None,
            combined_hapstore=merged_hapstore,
            population_vcfs={},
            population_hapstores={},
        )

    if merged_vcf.exists() and merged_manifest.exists():
        return _QcInputs(
            mode="combined_merged",
            sample_manifest=merged_manifest,
            combined_vcf=merged_vcf,
            combined_hapstore=None,
            population_vcfs={},
            population_hapstores={},
        )

    pop_dir = config.run_root / "merged" / f"chr{chrom}" / "populations"
    population_vcfs: dict[str, Path] = {
        pop.name: pop_dir / f"{pop.name}.chr{chrom}.merged.vcf.gz" for pop in config.populations
    }
    available_population_vcfs: dict[str, Path] = {
        pop: path for pop, path in population_vcfs.items() if path.exists()
    }
    population_hapstores: dict[str, Path] = {
        pop.name: pop_dir / f"{pop.name}.chr{chrom}.merged.hapstore.zarr" for pop in config.populations
    }
    if merged_manifest.exists() and population_hapstores and all(path.exists() for path in population_hapstores.values()):
        return _QcInputs(
            mode="population_merged_hapstore",
            sample_manifest=merged_manifest,
            combined_vcf=None,
            combined_hapstore=None,
            population_vcfs=available_population_vcfs,
            population_hapstores=population_hapstores,
        )

    if merged_manifest.exists() and population_vcfs and all(path.exists() for path in population_vcfs.values()):
        return _QcInputs(
            mode="population_merged",
            sample_manifest=merged_manifest,
            combined_vcf=None,
            population_vcfs=population_vcfs,
            combined_hapstore=None,
            population_hapstores={},
        )

    if n_shards != 1:
        raise FileNotFoundError(
            "Merged outputs not found for multi-shard QC; run merge first. "
            "Supported QC inputs are either combined merged VCF or merged population VCFs plus merged manifest."
        )

    single_vcf = config.run_root / "synthetic" / f"chr{chrom}" / "shard_0_of_1" / f"combined.chr{chrom}.vcf.gz"
    single_hapstore = (
        config.run_root / "synthetic" / f"chr{chrom}" / "shard_0_of_1" / f"combined.chr{chrom}.hapstore.zarr"
    )
    single_manifest = config.run_root / "manifests" / f"synthetic_samples_chr{chrom}_shard_0_of_1.tsv"
    if single_hapstore.exists() and single_manifest.exists():
        return _QcInputs(
            mode="single_shard_combined_hapstore",
            sample_manifest=single_manifest,
            combined_vcf=single_vcf if single_vcf.exists() else None,
            combined_hapstore=single_hapstore,
            population_vcfs={},
            population_hapstores={},
        )

    if not single_vcf.exists() or not single_manifest.exists():
        raise FileNotFoundError(
            f"Expected synthetic outputs not found for chr{chrom}: {single_vcf} / {single_manifest}"
        )
    return _QcInputs(
        mode="single_shard_combined",
        sample_manifest=single_manifest,
        combined_vcf=single_vcf,
        combined_hapstore=None,
        population_vcfs={},
        population_hapstores={},
    )


def _validate_population_variant_axes(stores: dict[str, HapStore], chrom: int) -> int:
    pops = sorted(stores.keys())
    if not pops:
        raise ValueError("population_hapstores cannot be empty")
    ref_pop = pops[0]
    ref_pos = stores[ref_pop].variant_positions(chrom)
    for pop in pops[1:]:
        cur = stores[pop].variant_positions(chrom)
        if cur.shape != ref_pos.shape or not np.array_equal(cur, ref_pos):
            raise ValueError(
                f"variant axis mismatch between populations {ref_pop!r} and {pop!r} "
                f"in HapStore QC for chr{chrom}"
            )
    return int(ref_pos.shape[0])


def _run_qc_matrix_hapstore(
    config: RunConfig,
    chrom: int,
    inputs: _QcInputs,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    within_checks: list[str],
    numba_threads: int | None,
) -> tuple[dict[str, float], dict[str, float]]:
    sampled = config.qc.main_pairwise_mode == "sampled"
    max_dip = int(config.qc.sampled_max_diploids_per_population) if sampled else None
    variant_block = int(config.qc.matrix_variant_block_size)
    pop_seed_base = int(derived_seed(int(config.qc.seed), "main_qc_pop", chrom))

    per_pop_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_pop_indices: dict[str, np.ndarray] = {}

    def _subselect_counts(
        ac_full: np.ndarray,
        an_full: np.ndarray,
        variant_idx: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if variant_idx is None:
            return ac_full, an_full
        return ac_full[variant_idx], an_full[variant_idx]

    if inputs.combined_hapstore is not None:
        hs = HapStore(inputs.combined_hapstore, mode="r")
        n_variants = hs.n_variants(chrom)
        variant_indices = sample_variant_indices(
            n_variants=n_variants,
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "main_qc_variants", chrom)),
        )
        all_sample_ids = hs.sample_ids(chrom)
        pop_indices = sample_population_indices(
            all_sample_ids=all_sample_ids,
            groups=groups,
            max_diploids_per_population=max_dip,
            seed=pop_seed_base,
        )
        per_pop_indices.update(pop_indices)
        selected_variant_count = int(variant_indices.size) if variant_indices is not None else int(n_variants)
        logger.info(
            "main QC matrix count-build strategy mode=combined_hapstore chrom=%s strategy=%s total_variants=%s selected_variants=%s",
            chrom,
            "full_then_subselect",
            int(n_variants),
            selected_variant_count,
        )

        def _count_pop(pop: str) -> tuple[str, np.ndarray, np.ndarray]:
            ac_full, an_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=pop_indices[pop],
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac, an = _subselect_counts(ac_full, an_full, variant_indices)
            return pop, ac, an

        max_workers = min(max(1, int(config.qc.matrix_workers)), max(1, len(pop_indices)))
        if max_workers > 1 and len(pop_indices) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_count_pop, pop) for pop in sorted(pop_indices.keys())]
                for fut in as_completed(futures):
                    pop, ac, an = fut.result()
                    per_pop_counts[pop] = (ac, an)
        else:
            for pop in sorted(pop_indices.keys()):
                pop, ac, an = _count_pop(pop)
                per_pop_counts[pop] = (ac, an)
    elif inputs.population_hapstores:
        stores = {pop: HapStore(path, mode="r") for pop, path in inputs.population_hapstores.items()}
        n_variants = _validate_population_variant_axes(stores, chrom=chrom)
        variant_indices = sample_variant_indices(
            n_variants=n_variants,
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "main_qc_variants", chrom)),
        )
        selected_variant_count = int(variant_indices.size) if variant_indices is not None else int(n_variants)
        logger.info(
            "main QC matrix count-build strategy mode=population_hapstores chrom=%s strategy=%s total_variants=%s selected_variants=%s",
            chrom,
            "full_then_subselect",
            int(n_variants),
            selected_variant_count,
        )

        def _count_pop(pop: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
            hs = stores[pop]
            all_ids = hs.sample_ids(chrom)
            pop_idx = sample_population_indices(
                all_sample_ids=all_ids,
                groups={pop: groups[pop]},
                max_diploids_per_population=max_dip,
                seed=int(derived_seed(pop_seed_base, pop)),
            )[pop]
            ac_full, an_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=pop_idx,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac, an = _subselect_counts(ac_full, an_full, variant_indices)
            return pop, ac, an, pop_idx

        pops = sorted(stores.keys())
        max_workers = min(max(1, int(config.qc.matrix_workers)), max(1, len(pops)))
        if max_workers > 1 and len(pops) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_count_pop, pop) for pop in pops]
                for fut in as_completed(futures):
                    pop, ac, an, pop_idx = fut.result()
                    per_pop_counts[pop] = (ac, an)
                    per_pop_indices[pop] = pop_idx
        else:
            for pop in pops:
                pop, ac, an, pop_idx = _count_pop(pop)
                per_pop_counts[pop] = (ac, an)
                per_pop_indices[pop] = pop_idx
    else:
        raise ValueError("matrix QC requested but no HapStore inputs are available")

    pairwise: dict[str, float] = {}
    for a, b in pairs:
        if a not in per_pop_counts or b not in per_pop_counts:
            raise ValueError(f"pairwise populations not available in HapStore counts: ({a}, {b})")
        ac_a, an_a = per_pop_counts[a]
        ac_b, an_b = per_pop_counts[b]
        pairwise[_pair_key(a, b)] = matrix_hudson_fst_from_counts(
            ac_a,
            an_a,
            ac_b,
            an_b,
            numba_threads=numba_threads,
        )

    within: dict[str, float] = {}
    if inputs.combined_hapstore is not None:
        hs = HapStore(inputs.combined_hapstore, mode="r")
        variant_indices_within = sample_variant_indices(
            n_variants=hs.n_variants(chrom),
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "within_variants", chrom)),
        )
        for pop in within_checks:
            indices = per_pop_indices.get(pop)
            if indices is None or len(indices) < 4:
                within[pop] = float("nan")
                continue
            rng = np.random.default_rng(int(derived_seed(int(config.qc.seed), "within", chrom, pop)))
            shuffled = np.asarray(indices, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            split_a = shuffled[:half]
            split_b = shuffled[half:]
            ac_a_full, an_a_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_a,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_b_full, an_b_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_b,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_a, an_a = _subselect_counts(ac_a_full, an_a_full, variant_indices_within)
            ac_b, an_b = _subselect_counts(ac_b_full, an_b_full, variant_indices_within)
            within[pop] = matrix_hudson_fst_from_counts(
                ac_a,
                an_a,
                ac_b,
                an_b,
                numba_threads=numba_threads,
            )
    else:
        stores = {pop: HapStore(path, mode="r") for pop, path in inputs.population_hapstores.items()}
        variant_indices_within = sample_variant_indices(
            n_variants=_validate_population_variant_axes(stores, chrom=chrom),
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "within_variants", chrom)),
        )
        for pop in within_checks:
            hs = stores.get(pop)
            indices = per_pop_indices.get(pop)
            if hs is None or indices is None or len(indices) < 4:
                within[pop] = float("nan")
                continue
            rng = np.random.default_rng(int(derived_seed(int(config.qc.seed), "within", chrom, pop)))
            shuffled = np.asarray(indices, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            split_a = shuffled[:half]
            split_b = shuffled[half:]
            ac_a_full, an_a_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_a,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_b_full, an_b_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_b,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_a, an_a = _subselect_counts(ac_a_full, an_a_full, variant_indices_within)
            ac_b, an_b = _subselect_counts(ac_b_full, an_b_full, variant_indices_within)
            within[pop] = matrix_hudson_fst_from_counts(
                ac_a,
                an_a,
                ac_b,
                an_b,
                numba_threads=numba_threads,
            )

    return pairwise, within


def donor_baseline_fst(
    config: RunConfig,
    chrom: int,
    *,
    pairs: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    manifest = load_manifest(config)
    donors = manifest.donors_by_population
    pops = sorted(donors.keys())
    if pairs is None:
        pairs = _resolve_pairwise_sets(config, pops)
    else:
        pairs = [(str(a), str(b)) for a, b in pairs]
    if not pairs:
        return {}

    if config.qc.backend == "plink_counts":
        panel_pattern = config.input.panel_vcf_pattern
        if panel_pattern is None:
            raise ValueError(
                "qc.backend=plink_counts requires input.panel_vcf_pattern (VCF donor panel input)"
            )
        panel_vcf = config.resolve_panel_vcf_for_population(
            chrom=int(chrom),
            population="",
        )
        with tempfile.TemporaryDirectory(prefix=f"popamp_donor_qc_chr{chrom}_", dir=str(config.run_root)) as tmp_root:
            tmp_dir = Path(tmp_root)
            counts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for pop in sorted({x for pair in pairs for x in pair}):
                out_prefix = tmp_dir / f"donor_{pop}"
                counts[pop] = _plink_acount_for_group(
                    vcf_path=panel_vcf,
                    out_prefix=out_prefix,
                    plink2_path=config.outputs.plink2_path,
                    threads=config.qc.plink_threads,
                    keep_samples=donors[pop],
                )
            out: dict[str, float] = {}
            for a, b in pairs:
                key_a, ac_a, an_a = counts[a]
                key_b, ac_b, an_b = counts[b]
                out[_pair_key(a, b)] = _hudson_fst_from_acount(
                    key_a,
                    ac_a,
                    an_a,
                    key_b,
                    ac_b,
                    an_b,
                )
            return out

    if config.qc.backend == "plink_native_fst":
        panel_pattern = config.input.panel_vcf_pattern
        if panel_pattern is None:
            raise ValueError(
                "qc.backend=plink_native_fst requires input.panel_vcf_pattern (VCF donor panel input)"
            )
        with tempfile.TemporaryDirectory(prefix=f"popamp_donor_qc_native_fst_chr{chrom}_", dir=str(config.run_root)) as tmp_root:
            tmp_dir = Path(tmp_root)
            donor_vcf = config.resolve_panel_vcf_for_population(
                chrom=int(chrom),
                population="",
            )
            if "{population}" in panel_pattern and not donor_vcf.exists():
                by_pop: dict[str, Path] = {}
                for pop in pops:
                    pop_vcf = config.resolve_panel_vcf_for_population(
                        chrom=int(chrom),
                        population=str(pop),
                        require_exists=True,
                    )
                    if not pop_vcf.exists():
                        raise FileNotFoundError(f"Missing donor panel VCF for population {pop}: {pop_vcf}")
                    by_pop[pop] = pop_vcf
                donor_vcf = tmp_dir / f"donor_combined_chr{chrom}.vcf.gz"
                _merge_population_vcfs_to_temp(
                    config=config,
                    population_vcfs=by_pop,
                    output_vcf=donor_vcf,
                )

            sample_to_pop = {sid: pop for pop, ids in donors.items() for sid in ids}
            return _plink_fst_for_groups(
                vcf_path=donor_vcf,
                out_prefix=tmp_dir / "donor_pairwise",
                plink2_path=config.outputs.plink2_path,
                threads=config.qc.plink_threads,
                sample_to_group=sample_to_pop,
                pairs=pairs,
                keep_samples=sorted(sample_to_pop.keys()),
            )

    needed_pops = sorted({pop for pair in pairs for pop in pair})
    per_pop_pos: dict[str, np.ndarray] = {}
    per_pop_genotypes: dict[str, np.ndarray] = {}
    for pop in needed_pops:
        source_kind, source_path = _resolve_donor_panel_source_for_population(
            config,
            chrom=chrom,
            pop=pop,
        )
        donor_pos, donor_haps = _load_sample_haplotypes_from_panel(
            config=config,
            vcf_path=source_path,
            chrom=chrom,
            sample_ids=donors[pop],
            source_kind=source_kind,
        )
        per_pop_pos[pop] = np.asarray(donor_pos, dtype=np.int64)
        per_pop_genotypes[pop] = (
            donor_haps[0::2, :].astype(np.uint8, copy=False) + donor_haps[1::2, :].astype(np.uint8, copy=False)
        )

    common_pos = per_pop_pos[needed_pops[0]]
    for pop in needed_pops[1:]:
        common_pos = np.intersect1d(common_pos, per_pop_pos[pop], assume_unique=False)

    if common_pos.size < 2:
        return {_pair_key(a, b): float("nan") for a, b in pairs}

    counts_by_pop: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_variants = int(common_pos.size)
    for pop in needed_pops:
        pop_pos = per_pop_pos[pop]
        idx = np.searchsorted(pop_pos, common_pos)
        if idx.size != common_pos.size or not np.array_equal(pop_pos[idx], common_pos):
            raise RuntimeError(f"donor baseline variant alignment mismatch for population {pop!r}")
        geno = np.asarray(per_pop_genotypes[pop][:, idx], dtype=np.uint8, copy=False)
        counts_by_pop[pop] = (
            np.sum(geno, axis=0, dtype=np.float64),
            np.full(n_variants, 2.0 * float(geno.shape[0]), dtype=np.float64),
        )

    return _pairwise_fst_from_counts_map(
        pairs=pairs,
        counts_by_population=counts_by_pop,
        numba_threads=_resolve_python_qc_numba_threads(config),
    )


def _pairwise_fst_from_counts_map(
    *,
    pairs: list[tuple[str, str]],
    counts_by_population: dict[str, tuple[np.ndarray, np.ndarray]],
    numba_threads: int | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for a, b in pairs:
        key = _pair_key(a, b)
        if a not in counts_by_population or b not in counts_by_population:
            out[key] = float("nan")
            continue
        ac_a, an_a = counts_by_population[a]
        ac_b, an_b = counts_by_population[b]
        out[key] = matrix_hudson_fst_from_counts(
            ac_a,
            an_a,
            ac_b,
            an_b,
            numba_threads=numba_threads,
        )
    return out


def _resolve_fst_stochastic_gate_activation(
    *,
    config: RunConfig,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    donors_by_population: dict[str, list[str]],
) -> dict[str, Any]:
    mode = str(config.qc.fst_stochastic_gate_mode).strip().lower()
    threshold = float(config.qc.fst_stochastic_small_pop_fraction_threshold)
    pop_names = sorted({str(pop) for pair in pairs for pop in pair})
    sampled_cap = int(config.qc.sampled_max_diploids_per_population)

    donor_sizes: dict[str, int] = {}
    sampled_diploids: dict[str, int] = {}
    sampled_fraction_by_population: dict[str, float] = {}
    for pop in pop_names:
        donor_n = len({str(x) for x in donors_by_population.get(pop, [])})
        synth_n = len({str(x) for x in groups.get(pop, [])})
        sampled_n = min(int(synth_n), int(sampled_cap))
        donor_sizes[pop] = int(donor_n)
        sampled_diploids[pop] = int(sampled_n)
        sampled_fraction_by_population[pop] = (
            float(sampled_n) / float(donor_n) if donor_n > 0 else float("nan")
        )

    finite_fracs = [x for x in sampled_fraction_by_population.values() if np.isfinite(x)]
    auto_triggered = bool(any(x >= threshold for x in finite_fracs))
    enabled = False
    reason = "disabled_by_mode"
    if mode == "on":
        enabled = True
        reason = "forced_on"
    elif mode == "auto":
        enabled = auto_triggered
        reason = "auto_small_population_triggered" if enabled else "auto_not_triggered"

    return {
        "mode": mode,
        "enabled": bool(enabled),
        "reason": reason,
        "small_pop_fraction_threshold": float(threshold),
        "auto_triggered": bool(auto_triggered),
        "population_order": pop_names,
        "donor_sizes": donor_sizes,
        "sampled_diploids_by_population": sampled_diploids,
        "sampled_fraction_by_population": sampled_fraction_by_population,
    }


def _compute_stochastic_donor_fst_gate(
    *,
    config: RunConfig,
    chrom: int,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    pairwise_fst: dict[str, float],
    donors_by_population: dict[str, list[str]],
    sampled_diploids_by_population: dict[str, int],
    numba_threads: int | None,
) -> dict[str, Any]:
    reps = int(config.qc.fst_stochastic_reps)
    replace = bool(config.qc.fst_stochastic_replace)
    min_p_ge = float(config.qc.fst_stochastic_min_p_ge)
    seed = int(derived_seed(int(config.qc.seed), "fst_stochastic_gate", chrom))
    pop_names = sorted({str(pop) for pair in pairs for pop in pair})

    if reps <= 0:
        return {"available": False, "reason": "non_positive_reps", "reps": int(reps)}
    if not pop_names:
        return {"available": False, "reason": "no_populations_for_pairs"}

    donor_sizes: dict[str, int] = {}
    n_target_by_pop: dict[str, int] = {}
    per_pop_pos: dict[str, np.ndarray] = {}
    per_pop_genotypes: dict[str, np.ndarray] = {}
    for pop in pop_names:
        donor_ids_all = sorted({str(x) for x in donors_by_population.get(pop, [])})
        donor_n = len(donor_ids_all)
        donor_sizes[pop] = int(donor_n)
        if donor_n == 0:
            return {
                "available": False,
                "reason": f"missing_donors_for_population:{pop}",
                "donor_sizes": donor_sizes,
            }

        n_target = int(sampled_diploids_by_population.get(pop, 0))
        if n_target <= 0:
            synth_n = len({str(x) for x in groups.get(pop, [])})
            n_target = min(
                int(synth_n),
                int(config.qc.sampled_max_diploids_per_population),
                donor_n,
            )
        n_target = min(max(1, int(n_target)), donor_n)
        if n_target < 2:
            return {
                "available": False,
                "reason": f"insufficient_donors_for_population:{pop}",
                "donor_sizes": donor_sizes,
                "sampled_diploids_by_population": n_target_by_pop,
            }
        if (not replace) and donor_n < n_target:
            return {
                "available": False,
                "reason": f"replace_false_requires_more_donors:{pop}",
                "donor_sizes": donor_sizes,
                "sampled_diploids_by_population": n_target_by_pop,
            }
        n_target_by_pop[pop] = int(n_target)

        source_kind, source_path = _resolve_donor_panel_source_for_population(
            config,
            chrom=chrom,
            pop=pop,
        )
        donor_pos, donor_haps = _load_sample_haplotypes_from_panel(
            config=config,
            vcf_path=source_path,
            chrom=chrom,
            sample_ids=donor_ids_all,
            source_kind=source_kind,
        )
        if donor_haps.shape[0] < 4 or donor_haps.shape[1] < 2:
            return {
                "available": False,
                "reason": f"insufficient_donor_haplotypes:{pop}",
                "donor_sizes": donor_sizes,
            }
        donor_genotypes = (
            donor_haps[0::2, :].astype(np.uint8, copy=False) + donor_haps[1::2, :].astype(np.uint8, copy=False)
        )
        per_pop_pos[pop] = np.asarray(donor_pos, dtype=np.int64)
        per_pop_genotypes[pop] = np.asarray(donor_genotypes, dtype=np.uint8, copy=False)

    common_pos = per_pop_pos[pop_names[0]]
    for pop in pop_names[1:]:
        common_pos = np.intersect1d(common_pos, per_pop_pos[pop], assume_unique=False)
    if common_pos.size < 2:
        return {
            "available": False,
            "reason": "insufficient_common_variants",
            "common_variant_count": int(common_pos.size),
            "donor_sizes": donor_sizes,
            "sampled_diploids_by_population": n_target_by_pop,
        }

    variant_idx = sample_variant_indices(
        n_variants=int(common_pos.size),
        max_variants=int(config.qc.sampled_max_variants),
        seed=int(derived_seed(int(config.qc.seed), "main_qc_variants", chrom)),
    )
    sampled_pos = common_pos if variant_idx is None else common_pos[variant_idx]

    per_pop_genotypes_common: dict[str, np.ndarray] = {}
    for pop in pop_names:
        pop_pos = per_pop_pos[pop]
        idx = np.searchsorted(pop_pos, sampled_pos)
        if idx.size != sampled_pos.size or not np.array_equal(pop_pos[idx], sampled_pos):
            return {
                "available": False,
                "reason": f"failed_variant_axis_alignment:{pop}",
                "variant_count": int(sampled_pos.size),
                "donor_sizes": donor_sizes,
                "sampled_diploids_by_population": n_target_by_pop,
            }
        per_pop_genotypes_common[pop] = np.asarray(per_pop_genotypes[pop][:, idx], dtype=np.uint8, copy=False)

    n_variants = int(sampled_pos.size)
    baseline_counts_by_pop: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pop in pop_names:
        geno = per_pop_genotypes_common[pop]
        baseline_counts_by_pop[pop] = (
            np.sum(geno, axis=0, dtype=np.float64),
            np.full(n_variants, 2.0 * float(geno.shape[0]), dtype=np.float64),
        )
    baseline_pairwise_fst = _pairwise_fst_from_counts_map(
        pairs=pairs,
        counts_by_population=baseline_counts_by_pop,
        numba_threads=numba_threads,
    )

    observed_pair_abs_diff: dict[str, float] = {}
    for a, b in pairs:
        key = _pair_key(a, b)
        obs = pairwise_fst.get(key, float("nan"))
        base = baseline_pairwise_fst.get(key, float("nan"))
        if np.isfinite(obs) and np.isfinite(base):
            observed_pair_abs_diff[key] = float(abs(obs - base))
    if not observed_pair_abs_diff:
        return {
            "available": False,
            "reason": "no_finite_observed_pairs",
            "variant_count": int(n_variants),
            "donor_sizes": donor_sizes,
            "sampled_diploids_by_population": n_target_by_pop,
            "baseline_pairwise_fst": baseline_pairwise_fst,
        }

    null_per_pair: dict[str, list[float]] = {key: [] for key in observed_pair_abs_diff}
    null_rep_means: list[float] = []
    rng = np.random.default_rng(int(seed))
    for _ in range(reps):
        rep_counts_by_pop: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for pop in pop_names:
            geno = per_pop_genotypes_common[pop]
            n_target = n_target_by_pop[pop]
            sampled_rows = rng.choice(geno.shape[0], size=int(n_target), replace=bool(replace))
            sub = geno[sampled_rows, :]
            rep_counts_by_pop[pop] = (
                np.sum(sub, axis=0, dtype=np.float64),
                np.full(n_variants, 2.0 * float(n_target), dtype=np.float64),
            )

        rep_pairwise_fst = _pairwise_fst_from_counts_map(
            pairs=pairs,
            counts_by_population=rep_counts_by_pop,
            numba_threads=numba_threads,
        )
        rep_abs_diffs: list[float] = []
        for key in observed_pair_abs_diff:
            base = baseline_pairwise_fst.get(key, float("nan"))
            rep = rep_pairwise_fst.get(key, float("nan"))
            delta = float(abs(rep - base)) if np.isfinite(rep) and np.isfinite(base) else float("nan")
            null_per_pair[key].append(delta)
            if np.isfinite(delta):
                rep_abs_diffs.append(delta)
        null_rep_means.append(float(np.mean(rep_abs_diffs)) if rep_abs_diffs else float("nan"))

    obs_mean = float(np.mean(np.asarray(list(observed_pair_abs_diff.values()), dtype=np.float64)))
    null_arr = np.asarray([x for x in null_rep_means if np.isfinite(x)], dtype=np.float64)
    p_ge_mean = float((np.sum(null_arr >= obs_mean) + 1.0) / (null_arr.size + 1.0)) if null_arr.size else float("nan")
    gate_pass = bool(p_ge_mean >= min_p_ge) if np.isfinite(p_ge_mean) else None

    pair_summary: dict[str, dict[str, float | int]] = {}
    for key, obs in observed_pair_abs_diff.items():
        arr = np.asarray([x for x in null_per_pair[key] if np.isfinite(x)], dtype=np.float64)
        if arr.size:
            p_ge = float((np.sum(arr >= obs) + 1.0) / (arr.size + 1.0))
            pair_summary[key] = {
                "observed_abs_diff": float(obs),
                "null_mean": float(np.mean(arr)),
                "null_sd": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                "null_min": float(np.min(arr)),
                "null_max": float(np.max(arr)),
                "p_ge_observed": float(p_ge),
                "n_reps_used": int(arr.size),
            }
        else:
            pair_summary[key] = {
                "observed_abs_diff": float(obs),
                "null_mean": float("nan"),
                "null_sd": float("nan"),
                "null_min": float("nan"),
                "null_max": float("nan"),
                "p_ge_observed": float("nan"),
                "n_reps_used": 0,
            }

    sampled_fraction_by_population = {
        pop: (float(n_target_by_pop[pop]) / float(donor_sizes[pop])) if donor_sizes[pop] > 0 else float("nan")
        for pop in pop_names
    }
    return {
        "available": True,
        "gate_pass": gate_pass,
        "criterion": "p_ge_observed_abs_diff_mean >= fst_stochastic_min_p_ge",
        "fst_stochastic_min_p_ge": float(min_p_ge),
        "reps": int(reps),
        "replace": bool(replace),
        "seed": int(seed),
        "variant_count": int(n_variants),
        "donor_sizes": donor_sizes,
        "sampled_diploids_by_population": n_target_by_pop,
        "sampled_fraction_by_population": sampled_fraction_by_population,
        "baseline_pairwise_fst": baseline_pairwise_fst,
        "observed_pair_abs_diff": observed_pair_abs_diff,
        "summary": {
            "observed_abs_diff_mean": float(obs_mean),
            "null_abs_diff_mean_mean": float(np.mean(null_arr)) if null_arr.size else float("nan"),
            "null_abs_diff_mean_sd": float(np.std(null_arr, ddof=1)) if null_arr.size > 1 else float("nan"),
            "null_abs_diff_mean_min": float(np.min(null_arr)) if null_arr.size else float("nan"),
            "null_abs_diff_mean_max": float(np.max(null_arr)) if null_arr.size else float("nan"),
            "p_ge_observed_abs_diff_mean": float(p_ge_mean),
            "n_reps_used": int(null_arr.size),
        },
        "pairwise": pair_summary,
    }


def _summarize_fst_values(values: list[float], n_donors: int, n_folds: int) -> dict[str, float | int]:
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


def _validate_round_robin_n_folds(n_folds: int) -> None:
    if n_folds < 2 or n_folds > 10:
        raise ValueError("n_folds must be in [2, 10]")


def _validate_donor_fold_fraction(fold_fraction: float) -> None:
    if fold_fraction < 0.05 or fold_fraction > 0.5:
        raise ValueError("fold_fraction must be in [0.05, 0.5]")


def _round_robin_folds_from_fraction(fold_fraction: float) -> int:
    _validate_donor_fold_fraction(fold_fraction)
    n_folds = int(round(1.0 / fold_fraction))
    return max(2, min(10, n_folds))


def _round_robin_for_population_chrom(
    *,
    pop: str,
    vcf_path: Path,
    chrom: int,
    sample_ids: list[str],
    n_folds: int,
    seed: int,
    threads: int,
    plink2_path: str,
    out_prefix: Path,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    _validate_round_robin_n_folds(n_folds)
    folds = _even_folds(sample_ids, n_folds=n_folds, seed=seed)
    fold_sizes = [len(x) for x in folds]
    fold_label_pairs: list[tuple[str, str]] = []
    for fold_a, fold_b in combinations(range(len(folds)), 2):
        fold_label_pairs.append((f"F{fold_a}", f"F{fold_b}"))

    sample_to_fold: dict[str, str] = {}
    for fold_idx, fold_ids in enumerate(folds):
        label = f"F{fold_idx}"
        for sid in fold_ids:
            sample_to_fold[sid] = label

    fst_by_pair = _plink_fst_for_groups(
        vcf_path=vcf_path,
        out_prefix=out_prefix,
        plink2_path=plink2_path,
        threads=max(1, threads),
        sample_to_group=sample_to_fold,
        pairs=fold_label_pairs,
        keep_samples=sorted(sample_to_fold.keys()),
    )

    records: list[dict[str, float | int | str]] = []
    for fold_a, fold_b in combinations(range(len(folds)), 2):
        fst = float(fst_by_pair.get(_pair_key(f"F{fold_a}", f"F{fold_b}"), float("nan")))
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

    fst_values = [float(x["fst"]) for x in records]
    summary = _summarize_fst_values(fst_values, n_donors=len(sample_ids), n_folds=len(folds))
    summary["fold_min_size"] = int(min(fold_sizes))
    summary["fold_max_size"] = int(max(fold_sizes))
    return records, summary


def run_donor_within_pop_round_robin(
    config: RunConfig,
    chrom: int,
    fold_fraction: float = 0.2,
    seed: int = 42,
    workers: int = 1,
    populations: list[str] | None = None,
) -> DonorWithinPopRoundRobinReport:
    if workers <= 0:
        raise ValueError("workers must be > 0")
    n_folds = _round_robin_folds_from_fraction(fold_fraction)

    manifest = load_manifest(config)
    donors_by_population = manifest.donors_by_population
    pop_names = _resolve_requested_populations(populations, sorted(donors_by_population.keys()))
    diagnostics_dir = ensure_dir(config.run_root / "diagnostics")

    records: list[dict[str, float | int | str]] = []
    summaries: dict[str, dict[str, float | int]] = {}

    for pop_index, pop in enumerate(pop_names):
        donors = sorted(donors_by_population[pop])
        pop_seed = int(seed + (pop_index + 1) * 1009)
        panel_vcf = _resolve_panel_vcf_for_population(config, chrom=chrom, pop=pop)
        pop_records, pop_summary = _round_robin_for_population_chrom(
            pop=pop,
            vcf_path=panel_vcf,
            chrom=chrom,
            sample_ids=donors,
            n_folds=n_folds,
            seed=pop_seed,
            threads=workers,
            plink2_path=config.outputs.plink2_path,
            out_prefix=diagnostics_dir / f"donor_rr_{pop}_chr{chrom}",
        )
        records.extend(pop_records)
        summaries[pop] = pop_summary

    table_path = diagnostics_dir / f"donor_within_pop_round_robin_chr{chrom}.tsv"
    diagnostics_path = diagnostics_dir / f"donor_within_pop_round_robin_chr{chrom}.json"

    table_df = pd.DataFrame.from_records(records)
    if table_df.empty:
        table_df = pd.DataFrame(columns=["population", "fold_a", "fold_b", "n_a", "n_b", "fst"])
    table_df = table_df.sort_values(["population", "fold_a", "fold_b"]).reset_index(drop=True)
    table_df.to_csv(table_path, sep="\t", index=False)

    payload = {
        "chrom": int(chrom),
        "n_folds": int(n_folds),
        "fold_fraction": float(fold_fraction),
        "seed": int(seed),
        "workers": int(workers),
        "n_records": int(len(records)),
        "table_path": str(table_path),
        "summary_by_population": summaries,
    }
    write_json(diagnostics_path, payload)

    return DonorWithinPopRoundRobinReport(
        chrom=chrom,
        n_folds=n_folds,
        fold_fraction=float(fold_fraction),
        seed=seed,
        workers=workers,
        diagnostics_path=diagnostics_path,
        table_path=table_path,
        summary_by_population=summaries,
        n_records=len(records),
    )


def _synthetic_shard_groups_from_manifests(
    config: RunConfig, chrom: int, n_shards: int
) -> dict[str, dict[int, list[str]]]:
    groups: dict[str, dict[int, set[str]]] = {}
    for shard_id in range(n_shards):
        manifest_path = config.run_root / "manifests" / f"synthetic_samples_chr{chrom}_shard_{shard_id}_of_{n_shards}.tsv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing synthetic shard manifest: {manifest_path}")
        df = pd.read_csv(manifest_path, sep="\t", dtype=str)
        if "population" not in df.columns or "sample_id" not in df.columns:
            raise ValueError(f"Synthetic shard manifest missing required columns: {manifest_path}")
        for pop, grp in df.groupby("population"):
            by_pop = groups.setdefault(str(pop), {})
            by_pop.setdefault(shard_id, set()).update(grp["sample_id"].astype(str).tolist())

    out: dict[str, dict[int, list[str]]] = {}
    for pop, shard_map in groups.items():
        out[pop] = {int(shard_id): sorted(sample_ids) for shard_id, sample_ids in shard_map.items() if sample_ids}
    return out


def _pair_count(n: int) -> int:
    n_i = int(n)
    if n_i <= 1:
        return 0
    return (n_i * (n_i - 1)) // 2


def _min_shards_for_target_pairs(target_pairs: int) -> int:
    """Smallest n such that nC2 >= target_pairs."""
    k = int(target_pairs)
    if k <= 0:
        return 0
    # 2 shards give 1 pair.
    n = 2
    while _pair_count(n) < k:
        n += 1
    return n


def _subset_shards_for_target_pairs(
    *,
    shard_ids: list[int],
    target_pairs: int,
    seed: int,
) -> list[int]:
    """Randomly select a minimal shard subset to support at least target_pairs comparisons.

    This is used to reduce per-shard count building when a max pair target is provided.
    """
    if not shard_ids:
        return []
    k = max(1, int(target_pairs))
    n = min(len(shard_ids), _min_shards_for_target_pairs(k))
    if n >= len(shard_ids):
        return list(shard_ids)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(shard_ids), size=int(n), replace=False))
    return [shard_ids[int(i)] for i in idx.tolist()]


def _synthetic_shard_round_robin_for_population_chrom(
    *,
    pop: str,
    vcf_path: Path,
    chrom: int,
    shard_to_sample_ids: dict[int, list[str]],
    seed: int,
    threads: int,
    min_threads_per_match: int,
    max_pairs_per_population: int | None,
    plink2_path: str,
    out_prefix: Path,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int], int, int]:
    shard_ids = sorted(int(x) for x in shard_to_sample_ids.keys() if shard_to_sample_ids[x])
    shard_sizes = {int(shard_id): len(shard_to_sample_ids[shard_id]) for shard_id in shard_ids}
    total_samples = int(sum(shard_sizes.values()))

    candidate_pairs_total = _pair_count(len(shard_ids))
    shard_ids_used = shard_ids
    if max_pairs_per_population is not None:
        shard_ids_used = _subset_shards_for_target_pairs(
            shard_ids=shard_ids,
            target_pairs=int(max_pairs_per_population),
            seed=int(derived_seed(seed, "rr_shard_subset", pop, chrom)),
        )
        selected_pairs = list(combinations(shard_ids_used, 2))
        target_pairs = int(max_pairs_per_population)
        logger.info(
            "synthetic premerge RR (plink) shard subset pop=%s chrom=%s target_pairs=%s subset_shards=%s selected_pairs=%s candidate_pairs_total=%s",
            pop,
            chrom,
            target_pairs,
            len(shard_ids_used),
            len(selected_pairs),
            candidate_pairs_total,
        )
    else:
        selected_pairs = list(combinations(shard_ids, 2))
    logger.info(
        "synthetic premerge RR (plink) start pop=%s chrom=%s candidate_pairs=%s selected_pairs=%s",
        pop,
        chrom,
        candidate_pairs_total,
        len(selected_pairs),
    )

    sample_to_shard: dict[str, str] = {}
    for shard_id in shard_ids_used:
        label = f"S{shard_id}"
        for sid in shard_to_sample_ids[shard_id]:
            sample_to_shard[sid] = label

    records: list[dict[str, float | int | str]] = []
    fst_values: list[float] = []
    if selected_pairs:
        requested_label_pairs = [(f"S{shard_a}", f"S{shard_b}") for shard_a, shard_b in selected_pairs]
        fst_by_pair = _plink_fst_for_groups_maybe_pair_parallel(
            vcf_path=vcf_path,
            out_prefix=out_prefix,
            plink2_path=plink2_path,
            total_threads=max(1, threads),
            sample_to_group=sample_to_shard,
            pairs=requested_label_pairs,
            keep_samples=sorted(sample_to_shard.keys()),
            pair_parallel=True,
            min_threads_per_match=int(min_threads_per_match),
        )

        for shard_a, shard_b in selected_pairs:
            fst = float(fst_by_pair.get(_pair_key(f"S{shard_a}", f"S{shard_b}"), float("nan")))
            records.append(
                {
                    "population": pop,
                    "chrom": int(chrom),
                    "shard_a": int(shard_a),
                    "shard_b": int(shard_b),
                    "n_a": int(shard_sizes[shard_a]),
                    "n_b": int(shard_sizes[shard_b]),
                    "fst": float(fst),
                }
            )
            fst_values.append(float(fst))
        logger.info(
            "synthetic premerge RR (plink) complete pop=%s chrom=%s selected_pairs=%s fst_values=%s",
            pop,
            chrom,
            len(selected_pairs),
            [float(x) for x in fst_values],
        )

    summary = _summarize_fst_values(fst_values, n_donors=total_samples, n_folds=len(shard_ids))
    summary["n_shards"] = int(len(shard_ids))
    summary["n_candidate_pairs"] = int(candidate_pairs_total)
    summary["n_selected_pairs"] = int(len(selected_pairs))
    if max_pairs_per_population is not None:
        summary["rr_target_pairs"] = int(max_pairs_per_population)
        summary["rr_shard_subset_size"] = int(len(shard_ids_used))
    if shard_sizes:
        summary["shard_min_size"] = int(min(shard_sizes.values()))
        summary["shard_max_size"] = int(max(shard_sizes.values()))
    else:
        summary["shard_min_size"] = 0
        summary["shard_max_size"] = 0
    return records, summary, int(candidate_pairs_total), len(selected_pairs)


def _synthetic_shard_round_robin_for_population_chrom_matrix(
    *,
    pop: str,
    chrom: int,
    shard_to_sample_ids: dict[int, list[str]],
    shard_hapstores: dict[int, Path],
    seed: int,
    sampled: bool,
    max_variants: int,
    variant_block_size: int,
    numba_threads: int | None,
    max_pairs_per_population: int | None,
    total_threads: int,
    min_threads_per_match: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int], int, int]:
    shard_ids = sorted(int(x) for x in shard_to_sample_ids.keys() if shard_to_sample_ids[x])
    shard_sizes = {int(shard_id): len(shard_to_sample_ids[shard_id]) for shard_id in shard_ids}
    total_samples = int(sum(shard_sizes.values()))

    candidate_pairs_total = _pair_count(len(shard_ids))
    if max_pairs_per_population is not None:
        shard_ids_used = _subset_shards_for_target_pairs(
            shard_ids=shard_ids,
            target_pairs=int(max_pairs_per_population),
            seed=int(derived_seed(seed, "rr_shard_subset", pop, chrom)),
        )
        selected_pairs = list(combinations(shard_ids_used, 2))
        target_pairs = int(max_pairs_per_population)
        logger.info(
            "synthetic premerge RR (matrix) shard subset pop=%s chrom=%s target_pairs=%s subset_shards=%s selected_pairs=%s candidate_pairs_total=%s",
            pop,
            chrom,
            target_pairs,
            len(shard_ids_used),
            len(selected_pairs),
            candidate_pairs_total,
        )
    else:
        selected_pairs = list(combinations(shard_ids, 2))
    logger.info(
        "synthetic premerge RR (matrix) start pop=%s chrom=%s sampled=%s candidate_pairs=%s selected_pairs=%s",
        pop,
        chrom,
        sampled,
        candidate_pairs_total,
        len(selected_pairs),
    )

    if not shard_ids:
        summary = _summarize_fst_values([], n_donors=0, n_folds=0)
        summary["n_shards"] = 0
        summary["n_candidate_pairs"] = 0
        summary["n_selected_pairs"] = 0
        summary["shard_min_size"] = 0
        summary["shard_max_size"] = 0
        return [], summary, 0, 0

    if not selected_pairs:
        summary = _summarize_fst_values([], n_donors=total_samples, n_folds=len(shard_ids))
        summary["n_shards"] = int(len(shard_ids))
        summary["n_candidate_pairs"] = int(candidate_pairs_total)
        summary["n_selected_pairs"] = 0
        summary["shard_min_size"] = int(min(shard_sizes.values())) if shard_sizes else 0
        summary["shard_max_size"] = int(max(shard_sizes.values())) if shard_sizes else 0
        return [], summary, int(candidate_pairs_total), 0

    needed_shards = sorted({int(sid) for pair in selected_pairs for sid in pair})
    counts_by_shard: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    total_threads_i = max(1, int(total_threads))
    ingest_workers = min(len(needed_shards), total_threads_i)
    ingest_threads_per_worker = max(1, total_threads_i // max(1, ingest_workers))
    with telemetry_span(
        logger,
        "qc.syn_rr.build_counts",
        chrom=int(chrom),
        pop=str(pop),
        sampled=bool(sampled),
        selected_pairs=int(len(selected_pairs)),
        needed_shards=int(len(needed_shards)),
        ingest_workers=int(ingest_workers),
        ingest_threads_per_worker=int(ingest_threads_per_worker),
        count_strategy="full_then_subselect",
    ):
        shard_store_paths = {
            sid: path for sid, path in shard_hapstores.items() if sid in needed_shards
        }
        if len(shard_store_paths) != len(needed_shards):
            missing = sorted(set(needed_shards) - set(shard_store_paths.keys()))
            raise FileNotFoundError(f"Missing HapStore inputs for population {pop}: shards={missing}")

        ref_chrom = f"chr{chrom}"
        shard_stores = {
            sid: HapStore(path, mode="r") for sid, path in shard_store_paths.items()
        }
        n_variants = _validate_population_variant_axes(
            {f"S{sid}": hs for sid, hs in shard_stores.items()}, chrom=chrom
        )
        variant_indices = sample_variant_indices(
            n_variants=n_variants,
            max_variants=(int(max_variants) if sampled else None),
            seed=int(derived_seed(seed, "rr_variants", pop, chrom)),
        )
        selected_variant_count = int(variant_indices.size) if variant_indices is not None else int(n_variants)
        logger.info(
            "synthetic premerge RR (matrix) count-build strategy pop=%s chrom=%s strategy=%s total_variants=%s selected_variants=%s",
            pop,
            chrom,
            "full_then_subselect",
            n_variants,
            selected_variant_count,
        )

        rr_count_progress = RateLimitedProgress(
            logger=logger,
            name="qc.syn_rr.build_counts",
            every_sec=30.0,
        )

        def _compute_counts_for_shard(sid: int) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
            hs = HapStore(shard_store_paths[sid], mode="r")
            if not hs.has_chromosome(ref_chrom):
                raise KeyError(f"population {pop} shard {sid} HapStore missing chromosome {ref_chrom!r}")
            # Build AC/AN over the full contiguous variant axis, then subselect sampled
            # variants in-memory. This avoids thousands of tiny random zarr reads.
            ac_full, an_full = hapstore_counts_for_samples(
                store=hs,
                chrom=ref_chrom,
                sample_indices=None,
                variant_indices=None,
                variant_block_size=variant_block_size,
                numba_threads=ingest_threads_per_worker,
            )
            if variant_indices is None:
                ac, an = ac_full, an_full
            else:
                ac = ac_full[variant_indices]
                an = an_full[variant_indices]
            return sid, (ac, an)

        if ingest_workers > 1:
            with ThreadPoolExecutor(max_workers=ingest_workers) as pool:
                futures = [pool.submit(_compute_counts_for_shard, sid) for sid in needed_shards]
                done = 0
                for fut in as_completed(futures):
                    sid_out, counts = fut.result()
                    counts_by_shard[sid_out] = counts
                    done += 1
                    rr_count_progress.log(
                        pop=str(pop),
                        chrom=int(chrom),
                        done=int(done),
                        total=int(len(needed_shards)),
                        last_shard=int(sid_out),
                    )
        else:
            for sid in needed_shards:
                sid_out, counts = _compute_counts_for_shard(sid)
                counts_by_shard[sid_out] = counts
                rr_count_progress.log(
                    pop=str(pop),
                    chrom=int(chrom),
                    done=int(len(counts_by_shard)),
                    total=int(len(needed_shards)),
                    last_shard=int(sid_out),
                )

    records: list[dict[str, float | int | str]] = []
    fst_values: list[float] = []
    rr_threads_per_match = max(1, int(min_threads_per_match))
    rr_threads_per_match = min(rr_threads_per_match, max(1, int(total_threads)))
    rr_max_parallel_matches = max(1, int(total_threads) // rr_threads_per_match)

    def _compute_one(shard_pair: tuple[int, int]) -> dict[str, float | int | str]:
        shard_a, shard_b = shard_pair
        ac_a, an_a = counts_by_shard[shard_a]
        ac_b, an_b = counts_by_shard[shard_b]
        fst = matrix_hudson_fst_from_counts(
            ac_a,
            an_a,
            ac_b,
            an_b,
            numba_threads=rr_threads_per_match,
        )
        return {
            "population": pop,
            "chrom": int(chrom),
            "shard_a": int(shard_a),
            "shard_b": int(shard_b),
            "n_a": int(shard_sizes[shard_a]),
            "n_b": int(shard_sizes[shard_b]),
            "fst": float(fst),
        }

    rr_progress = RateLimitedProgress(logger=logger, name="qc.syn_rr.matrix_pairs", every_sec=30.0)
    with telemetry_span(
        logger,
        "qc.syn_rr.compute_pairs",
        chrom=int(chrom),
        pop=str(pop),
        selected_pairs=int(len(selected_pairs)),
        threads_per_match=int(rr_threads_per_match),
        max_parallel=int(rr_max_parallel_matches),
    ):
        t_pairs = perf_counter()
        if rr_max_parallel_matches > 1 and len(selected_pairs) > 1:
            with ThreadPoolExecutor(max_workers=min(rr_max_parallel_matches, len(selected_pairs))) as pool:
                futures = [pool.submit(_compute_one, pair) for pair in selected_pairs]
                done = 0
                for fut in as_completed(futures):
                    rec = fut.result()
                    records.append(rec)
                    fst_values.append(float(rec["fst"]))
                    done += 1
                    elapsed = perf_counter() - t_pairs
                    rr_progress.log(
                        pop=str(pop),
                        chrom=int(chrom),
                        done=int(done),
                        total=int(len(selected_pairs)),
                        rate_per_min=float((done / elapsed) * 60.0) if elapsed > 0 else float("nan"),
                        last_pair=f"S{rec['shard_a']}__S{rec['shard_b']}",
                        last_fst=float(rec["fst"]),
                    )
            records.sort(key=lambda x: (int(x["shard_a"]), int(x["shard_b"])))
        else:
            done = 0
            for shard_pair in selected_pairs:
                rec = _compute_one(shard_pair)
                records.append(rec)
                fst_values.append(float(rec["fst"]))
                done += 1
                elapsed = perf_counter() - t_pairs
                rr_progress.log(
                    pop=str(pop),
                    chrom=int(chrom),
                    done=int(done),
                    total=int(len(selected_pairs)),
                    rate_per_min=float((done / elapsed) * 60.0) if elapsed > 0 else float("nan"),
                    last_pair=f"S{rec['shard_a']}__S{rec['shard_b']}",
                    last_fst=float(rec["fst"]),
                )

    logger.info(
        "synthetic premerge RR (matrix) complete pop=%s chrom=%s selected_pairs=%s fst_values=%s",
        pop,
        chrom,
        len(selected_pairs),
        [float(x) for x in fst_values],
    )

    summary = _summarize_fst_values(fst_values, n_donors=total_samples, n_folds=len(shard_ids))
    summary["n_shards"] = int(len(shard_ids))
    summary["n_candidate_pairs"] = int(candidate_pairs_total)
    summary["n_selected_pairs"] = int(len(selected_pairs))
    if max_pairs_per_population is not None:
        summary["rr_target_pairs"] = int(max_pairs_per_population)
        # selected_pairs are only drawn from shard_ids_used so this is an upper bound.
        summary["rr_shard_subset_size"] = int(len(sorted({sid for pair in selected_pairs for sid in pair})))
    summary["shard_min_size"] = int(min(shard_sizes.values())) if shard_sizes else 0
    summary["shard_max_size"] = int(max(shard_sizes.values())) if shard_sizes else 0
    return records, summary, int(candidate_pairs_total), len(selected_pairs)


def _merge_synthetic_population_shards_temp(
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pop: str,
    output_vcf: Path,
) -> Path:
    shard_vcfs = [
        config.run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}" / f"{pop}.chr{chrom}.vcf.gz"
        for shard_id in range(n_shards)
    ]
    missing = [path for path in shard_vcfs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing synthetic shard VCFs for {pop}; first missing={missing[0]}")

    if n_shards == 1:
        shutil.copy2(shard_vcfs[0], output_vcf)
        src_tbi = Path(f"{shard_vcfs[0]}.tbi")
        if src_tbi.exists():
            shutil.copy2(src_tbi, Path(f"{output_vcf}.tbi"))
        return output_vcf

    try_bcftools = bool(shutil.which(config.outputs.bcftools_path))
    if try_bcftools:
        try:
            merge_shard_vcfs_bcftools(
                shard_vcfs=shard_vcfs,
                output_vcf_gz=output_vcf,
                bcftools_path=config.outputs.bcftools_path,
                threads=max(1, config.qc.plink_threads),
                no_index=config.outputs.bcftools_no_index,
            )
            return output_vcf
        except Exception:
            logger.warning("bcftools temp merge failed for %s; falling back to python merge", pop)

    merge_shard_vcfs(shard_vcfs=shard_vcfs, output_vcf_gz=output_vcf)
    return output_vcf


def run_synthetic_premerge_within_pop_round_robin(
    config: RunConfig,
    chrom: int,
    n_shards: int,
    n_folds: int | None = None,
    seed: int = 42,
    workers: int = 1,
    populations: list[str] | None = None,
) -> SyntheticPremergeWithinPopRoundRobinReport:
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")
    if workers <= 0:
        raise ValueError("workers must be > 0")
    if n_folds is not None:
        logger.warning("n_folds is ignored for synthetic premerge RR; using shard-vs-shard pairing instead")

    with telemetry_span(logger, "qc.syn_rr.load_shard_manifests", chrom=int(chrom), n_shards=int(n_shards)):
        sample_groups_by_shard = _synthetic_shard_groups_from_manifests(config, chrom=chrom, n_shards=n_shards)
    pop_names = _resolve_requested_populations(populations, sorted(sample_groups_by_shard.keys()))

    records: list[dict[str, float | int | str]] = []
    summaries: dict[str, dict[str, float | int]] = {}
    total_candidate_pairs = 0
    total_selected_pairs = 0
    diagnostics_dir = ensure_dir(config.run_root / "diagnostics")
    sampled_rr = config.qc.rr_pairwise_mode == "sampled"
    sampled_rr_uses_plink = sampled_rr and config.qc.backend != "python"
    python_qc_numba_threads = _resolve_python_qc_numba_threads(config)
    rr_min_threads = max(1, int(config.qc.synthetic_rr_min_cpus_per_match))
    rr_population_workers = min(len(pop_names), max(1, int(workers) // rr_min_threads)) if pop_names else 1
    rr_threads_budget_per_population = max(1, int(workers) // max(1, rr_population_workers))
    logger.info(
        "synthetic premerge RR dispatch chrom=%s backend=%s rr_pairwise_mode=%s sampled_rr_uses_plink=%s workers=%s n_shards=%s rr_population_workers=%s rr_threads_budget_per_population=%s",
        chrom,
        config.qc.backend,
        config.qc.rr_pairwise_mode,
        sampled_rr_uses_plink,
        workers,
        n_shards,
        rr_population_workers,
        rr_threads_budget_per_population,
    )

    if sampled_rr_uses_plink:
        with telemetry_span(
            logger,
            "qc.syn_rr.within_pop_plink",
            chrom=int(chrom),
            n_shards=int(n_shards),
            workers=int(workers),
            n_populations=int(len(pop_names)),
        ):
            with tempfile.TemporaryDirectory(prefix=f"popamp_syn_premerge_rr_chr{chrom}_", dir=str(config.run_root)) as tmp_root:
                tmp_dir = Path(tmp_root)
                merged_tmp_by_pop: dict[str, Path] = {}
                for pop_index, pop in enumerate(pop_names):
                    merged_tmp_vcf = tmp_dir / f"{pop}.chr{chrom}.tmp.merged.vcf.gz"
                    _merge_synthetic_population_shards_temp(
                        config=config,
                        chrom=chrom,
                        n_shards=n_shards,
                        pop=pop,
                        output_vcf=merged_tmp_vcf,
                    )
                    merged_tmp_by_pop[pop] = merged_tmp_vcf

                def _run_pop_rr(
                    pop_index: int, pop: str
                ) -> tuple[str, list[dict[str, float | int | str]], dict[str, float | int], int, int]:
                    pop_seed = int(seed + (pop_index + 1) * 1009)
                    pop_records, pop_summary, pop_candidate_pairs, pop_selected_pairs = (
                        _synthetic_shard_round_robin_for_population_chrom(
                            pop=pop,
                            vcf_path=merged_tmp_by_pop[pop],
                            chrom=chrom,
                            shard_to_sample_ids=sample_groups_by_shard[pop],
                            seed=pop_seed,
                            threads=rr_threads_budget_per_population,
                            min_threads_per_match=int(config.qc.synthetic_rr_min_cpus_per_match),
                            max_pairs_per_population=config.qc.synthetic_round_robin_max_pairs_per_population,
                            plink2_path=config.outputs.plink2_path,
                            out_prefix=tmp_dir / f"synthetic_rr_{pop}_chr{chrom}",
                        )
                    )
                    return pop, pop_records, pop_summary, int(pop_candidate_pairs), int(pop_selected_pairs)

                if rr_population_workers > 1 and len(pop_names) > 1:
                    with ThreadPoolExecutor(max_workers=rr_population_workers) as pool:
                        futures = [pool.submit(_run_pop_rr, pop_index, pop) for pop_index, pop in enumerate(pop_names)]
                        for future in as_completed(futures):
                            pop, pop_records, pop_summary, pop_candidate_pairs, pop_selected_pairs = future.result()
                            records.extend(pop_records)
                            summaries[pop] = pop_summary
                            total_candidate_pairs += int(pop_candidate_pairs)
                            total_selected_pairs += int(pop_selected_pairs)
                            logger.info(
                                "synthetic premerge RR pop complete pop=%s selected_pairs=%s candidate_pairs=%s",
                                pop,
                                pop_selected_pairs,
                                pop_candidate_pairs,
                            )
                else:
                    for pop_index, pop in enumerate(pop_names):
                        pop, pop_records, pop_summary, pop_candidate_pairs, pop_selected_pairs = _run_pop_rr(pop_index, pop)
                        records.extend(pop_records)
                        summaries[pop] = pop_summary
                        total_candidate_pairs += int(pop_candidate_pairs)
                        total_selected_pairs += int(pop_selected_pairs)
                        logger.info(
                            "synthetic premerge RR pop complete pop=%s selected_pairs=%s candidate_pairs=%s",
                            pop,
                            pop_selected_pairs,
                            pop_candidate_pairs,
                        )

                # Cross-pop comparisons are intentionally not performed pre-merge.
    else:
        sampled_matrix = sampled_rr
        pop_hapstore_paths: dict[str, dict[int, Path]] = {}
        for pop in pop_names:
            shard_ids = sorted(sample_groups_by_shard[pop].keys())
            shard_hapstores = {
                int(shard_id): (
                    config.run_root
                    / "synthetic"
                    / f"chr{chrom}"
                    / f"shard_{int(shard_id)}_of_{n_shards}"
                    / f"{pop}.chr{chrom}.hapstore.zarr"
                )
                for shard_id in shard_ids
            }
            missing = [path for path in shard_hapstores.values() if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing synthetic shard HapStores for population {pop}; first missing={missing[0]}"
                )
            pop_hapstore_paths[pop] = shard_hapstores

        # rr_population_workers and rr_threads_budget_per_population are computed once
        # from the caller-provided worker budget and qc.synthetic_rr_min_cpus_per_match.

        def _run_pop_rr_matrix(pop_index: int, pop: str) -> tuple[str, list[dict[str, float | int | str]], dict[str, float | int], int, int]:
            pop_seed = int(seed + (pop_index + 1) * 1009)
            pop_records, pop_summary, pop_candidate_pairs, pop_selected_pairs = _synthetic_shard_round_robin_for_population_chrom_matrix(
                pop=pop,
                chrom=chrom,
                shard_to_sample_ids=sample_groups_by_shard[pop],
                shard_hapstores=pop_hapstore_paths[pop],
                seed=pop_seed,
                sampled=sampled_matrix,
                max_variants=int(config.qc.sampled_max_variants),
                variant_block_size=int(config.qc.matrix_variant_block_size),
                numba_threads=python_qc_numba_threads,
                max_pairs_per_population=config.qc.synthetic_round_robin_max_pairs_per_population,
                total_threads=rr_threads_budget_per_population,
                min_threads_per_match=int(config.qc.synthetic_rr_min_cpus_per_match),
            )
            return pop, pop_records, pop_summary, int(pop_candidate_pairs), int(pop_selected_pairs)

        if rr_population_workers > 1:
            with ThreadPoolExecutor(max_workers=rr_population_workers) as pool:
                future_to_pop = {
                    pool.submit(_run_pop_rr_matrix, pop_index, pop): pop
                    for pop_index, pop in enumerate(pop_names)
                }
                for future in as_completed(future_to_pop):
                    pop, pop_records, pop_summary, pop_candidate_pairs, pop_selected_pairs = future.result()
                    records.extend(pop_records)
                    summaries[pop] = pop_summary
                    total_candidate_pairs += int(pop_candidate_pairs)
                    total_selected_pairs += int(pop_selected_pairs)
                    logger.info(
                        "synthetic premerge RR pop complete pop=%s selected_pairs=%s candidate_pairs=%s",
                        pop,
                        pop_selected_pairs,
                        pop_candidate_pairs,
                    )
        else:
            for pop_index, pop in enumerate(pop_names):
                pop, pop_records, pop_summary, pop_candidate_pairs, pop_selected_pairs = _run_pop_rr_matrix(pop_index, pop)
                records.extend(pop_records)
                summaries[pop] = pop_summary
                total_candidate_pairs += int(pop_candidate_pairs)
                total_selected_pairs += int(pop_selected_pairs)
                logger.info(
                    "synthetic premerge RR pop complete pop=%s selected_pairs=%s candidate_pairs=%s",
                    pop,
                    pop_selected_pairs,
                    pop_candidate_pairs,
                )

        # Cross-pop comparisons are intentionally not performed pre-merge.

    with telemetry_span(logger, "qc.syn_rr.write_outputs", chrom=int(chrom)):
        table_path = diagnostics_dir / f"synthetic_premerge_within_pop_round_robin_chr{chrom}.tsv"
        diagnostics_path = diagnostics_dir / f"synthetic_premerge_within_pop_round_robin_chr{chrom}.json"
        table_df = pd.DataFrame.from_records(records)
        if table_df.empty:
            table_df = pd.DataFrame(columns=["population", "chrom", "shard_a", "shard_b", "n_a", "n_b", "fst"])
        table_df = table_df.sort_values(["population", "shard_a", "shard_b"]).reset_index(drop=True)
        table_df.to_csv(table_path, sep="\t", index=False)

        payload = {
            "chrom": int(chrom),
            "n_shards": int(n_shards),
            "rr_pairwise_mode": str(config.qc.rr_pairwise_mode),
            "rr_sampled_implementation": (
                "plink"
                if sampled_rr_uses_plink
                else ("matrix_sampled" if sampled_rr else "matrix_exact")
            ),
            "synthetic_rr_min_cpus_per_match": int(config.qc.synthetic_rr_min_cpus_per_match),
            "seed": int(seed),
            "workers": int(workers),
            "rr_population_workers": int(rr_population_workers),
            "rr_threads_budget_per_population": int(rr_threads_budget_per_population),
            "synthetic_round_robin_max_pairs_per_population": (
                int(config.qc.synthetic_round_robin_max_pairs_per_population)
                if config.qc.synthetic_round_robin_max_pairs_per_population is not None
                else None
            ),
            "n_candidate_pairs": int(total_candidate_pairs),
            "n_selected_pairs": int(total_selected_pairs),
            "n_records": int(len(records)),
            "table_path": str(table_path),
            "summary_by_population": summaries,
        }
        write_json(diagnostics_path, payload)
        logger.info(
            "synthetic premerge RR complete chrom=%s n_records=%s n_selected_pairs=%s diagnostics=%s",
            chrom,
            len(records),
            total_selected_pairs,
            diagnostics_path,
        )

    return SyntheticPremergeWithinPopRoundRobinReport(
        chrom=chrom,
        n_shards=n_shards,
        seed=seed,
        workers=workers,
        diagnostics_path=diagnostics_path,
        table_path=table_path,
        summary_by_population=summaries,
        n_candidate_pairs=int(total_candidate_pairs),
        n_selected_pairs=int(total_selected_pairs),
        n_records=len(records),
    )


def run_qc(config: RunConfig, chrom: int, n_shards: int = 1) -> QcReport:
    t0 = perf_counter()
    stage_timings: dict[str, float] = {}
    with telemetry_span(logger, "qc.resolve_inputs", chrom=int(chrom), n_shards=int(n_shards)):
        inputs = _resolve_synthetic_outputs(config, chrom=chrom, n_shards=n_shards)
        groups = _sample_groups_from_manifest(inputs.sample_manifest)
        pop_names = sorted(groups.keys())
        pairs = _resolve_pairwise_sets(config, pop_names)
        within_checks = _resolve_within_split_checks(config, pop_names)
        python_qc_numba_threads = _resolve_python_qc_numba_threads(config)
    population_spec_by_name = {str(pop.name): pop for pop in config.populations}
    population_engine = {
        pop_name: (
            str(population_spec_by_name[pop_name].engine)
            if pop_name in population_spec_by_name
            else "unknown"
        )
        for pop_name in pop_names
    }
    population_msprime_is_admixed = {
        pop_name: bool(
            pop_name in population_spec_by_name
            and population_spec_by_name[pop_name].engine == "msprime_tracts"
            and population_spec_by_name[pop_name].msprime is not None
            and len(population_spec_by_name[pop_name].msprime.sources) > 1
        )
        for pop_name in pop_names
    }
    admixed_pop_names = [pop for pop in pop_names if bool(population_msprime_is_admixed.get(pop, False))]
    admixed_pop_set = set(admixed_pop_names)
    admixed_qc_enabled = _is_admixed_qc_enabled(config, population_msprime_is_admixed)
    source_source_pairs = [pair for pair in pairs if not _is_admixed_involved_pair(pair, population_msprime_is_admixed)]
    admixed_involved_pairs = [pair for pair in pairs if _is_admixed_involved_pair(pair, population_msprime_is_admixed)]
    pair_key_to_pair = {_pair_key(a, b): (a, b) for a, b in pairs}

    with telemetry_span(
        logger,
        "qc.compute_fst",
        chrom=int(chrom),
        backend=str(config.qc.backend),
        mode=str(inputs.mode),
        n_pairs=int(len(pairs)),
        n_within=int(len(within_checks)),
    ):
        t_pairwise = perf_counter()
        if config.qc.backend == "plink_counts":
            pairwise, within = _run_qc_plink_counts(
                config=config,
                chrom=chrom,
                inputs=inputs,
                groups=groups,
                pairs=pairs,
                within_checks=within_checks,
            )
        elif config.qc.backend == "plink_native_fst":
            pairwise, within = _run_qc_plink_native_fst(
                config=config,
                chrom=chrom,
                inputs=inputs,
                groups=groups,
                pairs=pairs,
                within_checks=within_checks,
            )
        else:
            if inputs.combined_hapstore is not None or inputs.population_hapstores:
                pairwise, within = _run_qc_matrix_hapstore(
                    config=config,
                    chrom=chrom,
                    inputs=inputs,
                    groups=groups,
                    pairs=pairs,
                    within_checks=within_checks,
                    numba_threads=python_qc_numba_threads,
                )
            elif inputs.combined_vcf is not None:
                pairwise = _hudson_fst_from_vcf(
                    vcf_path=inputs.combined_vcf,
                    chrom=chrom,
                    groups=groups,
                    pairs=pairs,
                )
                within = {
                    pop: _within_split_fst(
                        vcf_path=inputs.combined_vcf,
                        chrom=chrom,
                        pop=pop,
                        sample_ids=groups[pop],
                        seed=42,
                    )
                    for pop in within_checks
                }
            else:
                pairwise = _hudson_fst_from_population_vcfs(
                    population_vcfs=inputs.population_vcfs,
                    chrom=chrom,
                    pairs=pairs,
                )
                within = {}
                for pop in within_checks:
                    if pop not in inputs.population_vcfs:
                        raise ValueError(f"within_pop_split_checks requested unknown population {pop}")
                    within[pop] = _within_split_fst(
                        vcf_path=inputs.population_vcfs[pop],
                        chrom=chrom,
                        pop=pop,
                        sample_ids=groups[pop],
                        seed=42,
                    )
        stage_timings["compute_pairwise_within_sec"] = perf_counter() - t_pairwise
    logger.info(
        "main QC pairwise complete chrom=%s backend=%s mode=%s pairwise_fst=%s within_split_fst=%s",
        chrom,
        config.qc.backend,
        inputs.mode,
        pairwise,
        within,
    )

    gate_pass = True
    donor_pairwise: dict[str, float] | None = None
    fst_gate_standard_pass: bool | None = None
    fst_gate_effective_pass: bool | None = None
    fst_gate_strategy_used = "standard_relative_tolerance"
    fst_stochastic_gate_activation: dict[str, Any] | None = None
    fst_stochastic_gate: dict[str, Any] | None = None
    fst_pairwise_gate_status: dict[str, Any] = {}
    within_split_gate_status: dict[str, Any] = {}
    msprime_tract_composition_gate: dict[str, Any] | None = None
    admixed_tract_length_gate: dict[str, Any] | None = None
    admixed_ancestry_variance_gate: dict[str, Any] | None = None
    admixed_optional_diagnostics: dict[str, Any] = {}
    admixed_qc: dict[str, Any] | None = None
    ld_decay: dict[str, Any] | None = None
    ld_decay_path: Path | None = None
    fst_gate_tolerance = float(config.qc.fst_tolerance_relative)
    if (
        config.qc.backend == "python"
        and config.qc.main_pairwise_mode == "sampled"
        and (inputs.combined_hapstore is not None or inputs.population_hapstores)
    ):
        fst_gate_tolerance = float(config.qc.sampled_gate_tolerance_relative)
    diagnostics_dir = ensure_dir(config.run_root / "diagnostics")
    with telemetry_span(logger, "qc.donor_gate", chrom=int(chrom)):
        t_gate = perf_counter()
        try:
            t_donor_pairs = perf_counter()
            donor_pairs_for_baseline = pairs
            if bool(config.qc.admixed_skip_donor_baseline_pairs) and admixed_qc_enabled:
                donor_pairs_for_baseline = source_source_pairs
            donor_pairwise = donor_baseline_fst(
                config,
                chrom=chrom,
                pairs=donor_pairs_for_baseline,
            )
            stage_timings["donor_gate_source_source_sec"] = perf_counter() - t_donor_pairs
            if bool(config.qc.admixed_skip_donor_baseline_pairs) and admixed_qc_enabled:
                logger.info(
                    "main QC donor baseline pair filtering chrom=%s total_pairs=%s source_source_pairs=%s skipped_admixed_pairs=%s",
                    chrom,
                    len(pairs),
                    len(source_source_pairs),
                    len(admixed_involved_pairs),
                )
            n_fst_pairs_gated = 0
            n_fst_pairs_failed = 0
            for key, obs in pairwise.items():
                base = donor_pairwise.get(key)
                pair = pair_key_to_pair.get(key)
                gated = True
                gate_reason = "standard"
                is_admixed_pair = bool(pair is not None and _is_admixed_involved_pair(pair, population_msprime_is_admixed))
                if (
                    is_admixed_pair
                    and bool(config.qc.admixed_skip_donor_baseline_pairs)
                    and admixed_qc_enabled
                ):
                    gated = False
                    gate_reason = "skipped_admixed_pair_policy"
                elif (
                    bool(config.qc.engine_aware_gating)
                    and is_admixed_pair
                ):
                    gated = False
                    gate_reason = "advisory_admixed_msprime_population"

                rel: float | None = None
                pair_pass: bool | None = None
                if gate_reason == "skipped_admixed_pair_policy":
                    pair_pass = None
                    rel = None
                elif base is None or np.isnan(base) or np.isnan(obs):
                    gate_reason = "missing_baseline_or_observed"
                else:
                    if abs(base) > 1e-12:
                        rel = abs(obs - base) / abs(base)
                    else:
                        rel = abs(obs - base)
                    pair_pass = bool(rel <= fst_gate_tolerance)
                    if gated:
                        n_fst_pairs_gated += 1
                        if pair_pass is False:
                            n_fst_pairs_failed += 1
                            gate_reason = "failed_relative_tolerance"
                        else:
                            gate_reason = "passed_relative_tolerance"
                    else:
                        gate_reason = "advisory_admixed_msprime_population"
                fst_pairwise_gate_status[key] = {
                    "pair": [pair[0], pair[1]] if pair is not None else None,
                    "gated": bool(gated),
                    "reason": str(gate_reason),
                    "observed_fst": (float(obs) if np.isfinite(obs) else None),
                    "donor_baseline_fst": (float(base) if (base is not None and np.isfinite(base)) else None),
                    "relative_delta": (float(rel) if (rel is not None and np.isfinite(rel)) else None),
                    "threshold": float(fst_gate_tolerance),
                    "pair_pass": pair_pass,
                }

            if n_fst_pairs_gated > 0:
                fst_gate_standard_pass = bool(n_fst_pairs_failed == 0)
            else:
                fst_gate_standard_pass = None

            fst_gate_effective_pass = fst_gate_standard_pass
            fst_gate_strategy_used = "standard_relative_tolerance"
            try:
                manifest = load_manifest(config)
                if admixed_qc_enabled:
                    fst_stochastic_gate_activation = {
                        "mode": str(config.qc.fst_stochastic_gate_mode),
                        "enabled": False,
                        "reason": "disabled_admixed_qc_policy",
                    }
                else:
                    fst_stochastic_gate_activation = _resolve_fst_stochastic_gate_activation(
                        config=config,
                        groups=groups,
                        pairs=pairs,
                        donors_by_population=manifest.donors_by_population,
                    )
                    if bool(fst_stochastic_gate_activation.get("enabled")):
                        try:
                            fst_stochastic_gate = _compute_stochastic_donor_fst_gate(
                                config=config,
                                chrom=chrom,
                                groups=groups,
                                pairs=pairs,
                                pairwise_fst=pairwise,
                                donors_by_population=manifest.donors_by_population,
                                sampled_diploids_by_population=(
                                    fst_stochastic_gate_activation.get("sampled_diploids_by_population", {})
                                    if isinstance(fst_stochastic_gate_activation, dict)
                                    else {}
                                ),
                                numba_threads=python_qc_numba_threads,
                            )
                        except Exception as exc:
                            fst_stochastic_gate = {
                                "available": False,
                                "reason": "error",
                                "error": str(exc),
                            }
                            logger.warning(
                                "Stochastic donor-baseline FST gate failed; falling back to standard gate: %s", exc
                            )
                        if (
                            isinstance(fst_stochastic_gate, dict)
                            and bool(fst_stochastic_gate.get("available"))
                            and isinstance(fst_stochastic_gate.get("gate_pass"), bool)
                        ):
                            fst_gate_effective_pass = bool(fst_stochastic_gate.get("gate_pass"))
                            fst_gate_strategy_used = "stochastic_donor_baseline"
            except Exception as exc:
                fst_stochastic_gate_activation = {
                    "mode": str(config.qc.fst_stochastic_gate_mode),
                    "enabled": False,
                    "reason": "activation_error",
                    "error": str(exc),
                }
                logger.warning("Skipping stochastic donor-baseline FST gate activation due to error: %s", exc)
            if fst_gate_effective_pass is False:
                gate_pass = False
            for pop, value in within.items():
                gated = True
                gate_reason = "standard"
                if admixed_qc_enabled and bool(population_msprime_is_admixed.get(pop, False)):
                    gated = False
                    gate_reason = "advisory_admixed_msprime_population"
                within_pass: bool | None = None
                if np.isnan(value):
                    gate_reason = "missing_observed"
                else:
                    within_pass = bool(abs(value) <= config.qc.within_split_abs_max)
                    if gated:
                        gate_reason = (
                            "passed_within_split_abs_max" if within_pass else "failed_within_split_abs_max"
                        )
                    else:
                        gate_reason = "advisory_admixed_msprime_population"
                    if gated and within_pass is False:
                        gate_pass = False
                within_split_gate_status[str(pop)] = {
                    "gated": bool(gated),
                    "reason": str(gate_reason),
                    "observed_within_split_fst": (float(value) if np.isfinite(value) else None),
                    "threshold": float(config.qc.within_split_abs_max),
                    "within_pass": within_pass,
                }

        except Exception as exc:
            logger.warning("Skipping donor-baseline gate due to error: %s", exc)
        if bool(config.qc.engine_aware_gating) or admixed_qc_enabled:
            try:
                msprime_tract_composition_gate = _compute_msprime_tract_composition_gate(
                    config=config,
                    chrom=int(chrom),
                    n_shards=int(n_shards),
                    pop_names=pop_names,
                )
                if (
                    isinstance(msprime_tract_composition_gate, dict)
                    and bool(msprime_tract_composition_gate.get("enabled"))
                    and msprime_tract_composition_gate.get("gate_pass") is False
                ):
                    gate_pass = False
            except Exception as exc:
                msprime_tract_composition_gate = {
                    "enabled": True,
                    "status": "error",
                    "gate_pass": False,
                    "error": str(exc),
                }
                gate_pass = False
                logger.warning("msprime tract-composition gate failed: %s", exc)
        if admixed_qc_enabled:
            t_admixed_tract_length = perf_counter()
            try:
                admixed_tract_length_gate = _compute_admixed_tract_length_gate(
                    config=config,
                    chrom=int(chrom),
                    n_shards=int(n_shards),
                    pop_names=pop_names,
                    diagnostics_dir=diagnostics_dir,
                )
                if (
                    isinstance(admixed_tract_length_gate, dict)
                    and bool(admixed_tract_length_gate.get("enabled"))
                    and admixed_tract_length_gate.get("gate_pass") is False
                ):
                    gate_pass = False
            except Exception as exc:
                admixed_tract_length_gate = {
                    "enabled": True,
                    "status": "error",
                    "gate_pass": False,
                    "error": str(exc),
                }
                gate_pass = False
                logger.warning("admixed tract-length gate failed: %s", exc)
            stage_timings["admixed_tract_length_gate_sec"] = perf_counter() - t_admixed_tract_length

            t_admixed_var = perf_counter()
            try:
                admixed_ancestry_variance_gate = _compute_admixed_ancestry_variance_gate(
                    config=config,
                    chrom=int(chrom),
                    n_shards=int(n_shards),
                    pop_names=pop_names,
                )
                if (
                    isinstance(admixed_ancestry_variance_gate, dict)
                    and bool(admixed_ancestry_variance_gate.get("enabled"))
                    and admixed_ancestry_variance_gate.get("gate_pass") is False
                ):
                    gate_pass = False
            except Exception as exc:
                admixed_ancestry_variance_gate = {
                    "enabled": True,
                    "status": "error",
                    "gate_pass": False,
                    "error": str(exc),
                }
                gate_pass = False
                logger.warning("admixed ancestry-variance gate failed: %s", exc)
            stage_timings["admixed_ancestry_variance_gate_sec"] = perf_counter() - t_admixed_var
        if donor_pairwise is not None:
            logger.info("main QC donor baseline pairwise chrom=%s donor_pairwise_fst=%s", chrom, donor_pairwise)
        stage_timings["donor_gate_sec"] = perf_counter() - t_gate

    donors_by_population: dict[str, list[str]] = {}
    donor_manifest_error: str | None = None
    if any(
        (
            bool(config.qc.king_enabled),
            bool(config.qc.afs_enabled),
            bool(config.qc.pca_enabled),
            bool(config.qc.roh_enabled),
            bool(config.qc.hwe_enabled),
            bool(config.qc.inbreeding_f_enabled),
            bool(config.qc.plink2_parity_enabled),
        )
    ):
        try:
            manifest = load_manifest(config)
            donors_by_population = manifest.donors_by_population
        except Exception as exc:
            donor_manifest_error = str(exc)
            logger.warning("Failed to load donor manifest for extended QC metrics: %s", exc)

    ext_data = None
    ext_data_error: str | None = None
    if any(
        (
            bool(config.qc.king_enabled),
            bool(config.qc.afs_enabled),
            bool(config.qc.pca_enabled),
            bool(config.qc.roh_enabled),
            bool(config.qc.hwe_enabled),
            bool(config.qc.inbreeding_f_enabled),
        )
    ):
        try:
            if donor_manifest_error is not None:
                raise RuntimeError(f"donor manifest unavailable: {donor_manifest_error}")
            from .context import build_ext_aligned_data

            ext_data = build_ext_aligned_data(
                chrom=int(chrom),
                seed=int(config.qc.seed),
                ext_max_diploids_per_population=int(config.qc.ext_max_diploids_per_population),
                ext_max_variants=int(config.qc.ext_max_variants),
                groups=groups,
                donors_by_population=donors_by_population,
                inputs=inputs,
                subsample_ids=lambda ids, target_n, s: _subsample_ids(
                    list(ids), target_n=int(target_n), seed=int(s)
                ),
                resolve_panel_source_for_population=lambda c, pop: _resolve_donor_panel_source_for_population(
                    config, chrom=int(c), pop=str(pop)
                ),
                load_sample_haplotypes_from_panel_source=lambda kind, source_path, c, sample_ids: _load_sample_haplotypes_from_panel(
                    config=config,
                    vcf_path=Path(source_path),
                    chrom=int(c),
                    sample_ids=list(sample_ids),
                    source_kind=str(kind),
                ),
                load_synthetic_population_haplotypes=lambda inps, c, pop, sample_ids: _load_synthetic_population_haplotypes(
                    config=config,
                    inputs=inps,
                    chrom=int(c),
                    pop=str(pop),
                    sample_ids=list(sample_ids),
                ),
            )
        except Exception as exc:
            ext_data_error = str(exc)
            logger.warning("Failed to build extended-QC shared sampling cache: %s", exc)

    metric_payloads: dict[str, dict[str, Any] | None] = {
        "ld_decay": None,
        "king": None,
        "afs": None,
        "hwe": None,
        "inbreeding_f": None,
        "pca": None,
        "roh": None,
        "plink2_parity": None,
    }
    for spec in METRIC_REGISTRY:
        if not spec.enabled(config):
            continue
        with telemetry_span(logger, f"qc.{spec.name}", chrom=int(chrom)):
            t_metric = perf_counter()
            try:
                if donor_manifest_error is not None and spec.payload_key != "ld_decay":
                    raise RuntimeError(f"donor manifest unavailable: {donor_manifest_error}")
                metric_payload = spec.runner(
                    config=config,
                    chrom=chrom,
                    inputs=inputs,
                    groups=groups,
                    donors_by_population=donors_by_population,
                    diagnostics_dir=diagnostics_dir,
                    numba_threads=python_qc_numba_threads,
                    ext_data=ext_data,
                    ext_data_error=ext_data_error,
                )
            except Exception as exc:
                logger.warning("%s QC failed and will be recorded as error: %s", spec.name, exc)
                metric_payload = {
                    "enabled": True,
                    "chrom": int(chrom),
                    "status": "error",
                    "error": str(exc),
                }
                if spec.contributes_gate:
                    metric_payload["gate_pass"] = None
            if (
                admixed_qc_enabled
                and spec.payload_key in {"hwe", "inbreeding_f"}
                and str(config.qc.admixed_hwe_inbreeding_mode).strip().lower() == "non_gating"
            ):
                metric_payload = _apply_admixed_non_gating_to_metric(
                    metric_payload,
                    admixed_pop_names=admixed_pop_set,
                    reason="non_gating_admixed_policy",
                )
            metric_payloads[spec.payload_key] = metric_payload
            if spec.contributes_gate and isinstance(metric_payload, dict):
                metric_gate = metric_payload.get("gate_pass")
                if metric_gate is False:
                    gate_pass = False
            stage_timings[f"{spec.payload_key}_sec"] = perf_counter() - t_metric

    ld_decay = metric_payloads["ld_decay"]
    king = metric_payloads["king"]
    afs = metric_payloads["afs"]
    hwe = metric_payloads["hwe"]
    inbreeding_f = metric_payloads["inbreeding_f"]
    pca = metric_payloads["pca"]
    roh = metric_payloads["roh"]
    plink2_parity = metric_payloads["plink2_parity"]
    if isinstance(ld_decay, dict) and ld_decay.get("status") != "error":
        ld_decay_path = diagnostics_dir / f"ld_decay_chr{chrom}.json"
    if not bool(config.qc.ld_decay_enabled):
        stage_timings.setdefault("ld_decay_sec", 0.0)

    if admixed_qc_enabled:
        t_diag_pca = perf_counter()
        try:
            admixed_optional_diagnostics["pca_cline"] = _compute_admixed_pca_cline_diagnostic(
                config=config,
                chrom=int(chrom),
                ext_data=ext_data,
                inputs=inputs,
                groups=groups,
                pop_names=admixed_pop_names,
                diagnostics_dir=diagnostics_dir,
            )
        except Exception as exc:
            admixed_optional_diagnostics["pca_cline"] = {
                "enabled": bool(config.qc.admixed_diag_pca_cline_enabled),
                "status": "error",
                "error": str(exc),
            }
            logger.warning("admixed PCA cline diagnostic failed: %s", exc)
        stage_timings["admixed_diag_pca_cline_sec"] = perf_counter() - t_diag_pca

        t_diag_king = perf_counter()
        try:
            admixed_optional_diagnostics["king"] = _compute_admixed_king_diagnostic(
                config=config,
                king_payload=king,
                pop_names=admixed_pop_names,
            )
        except Exception as exc:
            admixed_optional_diagnostics["king"] = {
                "enabled": bool(config.qc.admixed_diag_king_enabled),
                "status": "error",
                "error": str(exc),
            }
            logger.warning("admixed KING diagnostic failed: %s", exc)
        stage_timings["admixed_diag_king_sec"] = perf_counter() - t_diag_king

        if bool(config.qc.admixed_diag_stratified_ld_enabled):
            admixed_optional_diagnostics["stratified_ld"] = {
                "enabled": True,
                "status": "not_implemented",
                "gate_pass": None,
            }
        else:
            admixed_optional_diagnostics["stratified_ld"] = {
                "enabled": False,
                "status": "disabled_by_config",
                "gate_pass": None,
            }
    else:
        admixed_optional_diagnostics["pca_cline"] = {
            "enabled": bool(config.qc.admixed_diag_pca_cline_enabled),
            "status": "disabled_admixed_qc_policy",
        }
        admixed_optional_diagnostics["king"] = {
            "enabled": bool(config.qc.admixed_diag_king_enabled),
            "status": "disabled_admixed_qc_policy",
        }
        admixed_optional_diagnostics["stratified_ld"] = {
            "enabled": bool(config.qc.admixed_diag_stratified_ld_enabled),
            "status": "disabled_admixed_qc_policy",
            "gate_pass": None,
        }

    population_status: dict[str, dict[str, Any]] = {}
    for pop_name in pop_names:
        pop_spec = population_spec_by_name.get(pop_name)
        source_count = (
            len(pop_spec.msprime.sources)
            if pop_spec is not None and pop_spec.msprime is not None
            else None
        )
        if bool(population_msprime_is_admixed.get(pop_name, False)):
            classification = "admixed_msprime_tracts"
        elif pop_spec is not None and pop_spec.engine == "msprime_tracts":
            classification = "single_source_msprime_tracts"
        else:
            classification = "mosaic"
        population_status[pop_name] = {
            "engine": population_engine.get(pop_name, "unknown"),
            "classification": classification,
            "msprime_source_count": source_count,
        }

    hard_gate_components = {
        "tract_composition": (
            msprime_tract_composition_gate
            if isinstance(msprime_tract_composition_gate, dict)
            else {"enabled": False, "status": "not_available", "gate_pass": None}
        ),
        "tract_length_distribution": (
            admixed_tract_length_gate
            if isinstance(admixed_tract_length_gate, dict)
            else {"enabled": False, "status": "not_available", "gate_pass": None}
        ),
        "ancestry_variance": (
            admixed_ancestry_variance_gate
            if isinstance(admixed_ancestry_variance_gate, dict)
            else {"enabled": False, "status": "not_available", "gate_pass": None}
        ),
    }
    skipped_components: dict[str, Any] = {}
    if bool(config.qc.admixed_skip_donor_baseline_pairs) and admixed_qc_enabled and admixed_involved_pairs:
        skipped_components["donor_baseline_pairs"] = {
            "status": "skipped_by_policy",
            "reason": "skipped_admixed_pair_policy",
            "n_pairs_skipped": int(len(admixed_involved_pairs)),
            "n_pairs_total": int(len(pairs)),
            "n_pairs_source_source_computed": int(len(source_source_pairs)),
        }
    if admixed_qc_enabled and admixed_pop_names:
        skipped_components["within_split_fst"] = {
            "status": "non_gating_for_admixed",
            "reason": "advisory_admixed_msprime_population",
            "n_admixed_populations": int(len(admixed_pop_names)),
        }
    if (
        admixed_qc_enabled
        and str(config.qc.admixed_hwe_inbreeding_mode).strip().lower() == "non_gating"
        and admixed_pop_names
    ):
        skipped_components["hwe_inbreeding"] = {
            "status": "non_gating_for_admixed",
            "reason": "non_gating_admixed_policy",
            "n_admixed_populations": int(len(admixed_pop_names)),
        }

    component_gate_values = [
        component.get("gate_pass")
        for component in hard_gate_components.values()
        if isinstance(component, dict) and component.get("enabled") and isinstance(component.get("gate_pass"), bool)
    ]
    hard_gate_pass: bool | None
    if str(config.qc.admixed_qc_mode).strip().lower() == "strict" and admixed_qc_enabled:
        strict_values = []
        for component in hard_gate_components.values():
            if isinstance(component, dict) and component.get("enabled"):
                strict_values.append(component.get("gate_pass") is True)
        hard_gate_pass = bool(strict_values and all(strict_values))
    elif component_gate_values:
        hard_gate_pass = bool(all(component_gate_values))
    else:
        hard_gate_pass = None
    if hard_gate_pass is False:
        gate_pass = False

    admixed_qc = {
        "enabled": bool(admixed_qc_enabled),
        "mode": str(config.qc.admixed_qc_mode),
        "population_status": population_status,
        "hard_gate_components": hard_gate_components,
        "hard_gate_pass": hard_gate_pass,
        "skipped_components": skipped_components,
        "optional_diagnostics": admixed_optional_diagnostics,
    }

    diagnostics_path = diagnostics_dir / f"qc_chr{chrom}.json"
    with telemetry_span(logger, "qc.write_diagnostics", chrom=int(chrom)):
        t_write = perf_counter()
        payload = {
            "chrom": chrom,
            "backend": config.qc.backend,
            "ld_decay_backend": config.qc.ld_decay_backend,
            "main_pairwise_mode": config.qc.main_pairwise_mode,
            "rr_pairwise_mode": config.qc.rr_pairwise_mode,
            "python_qc_numba_threads": int(python_qc_numba_threads),
            "fst_gate_tolerance_relative_used": fst_gate_tolerance,
            "fst_gate_strategy_used": fst_gate_strategy_used,
            "fst_gate_standard_pass": fst_gate_standard_pass,
            "fst_gate_effective_pass": fst_gate_effective_pass,
            "fst_stochastic_gate_activation": fst_stochastic_gate_activation,
            "fst_stochastic_gate": fst_stochastic_gate,
            "fst_pairwise_gate_status": fst_pairwise_gate_status,
            "within_split_gate_status": within_split_gate_status,
            "engine_aware_gating": bool(config.qc.engine_aware_gating),
            "population_engine": population_engine,
            "population_msprime_is_admixed": population_msprime_is_admixed,
            "msprime_tract_composition_gate": msprime_tract_composition_gate,
            "admixed_tract_length_gate": admixed_tract_length_gate,
            "admixed_ancestry_variance_gate": admixed_ancestry_variance_gate,
            "admixed_optional_diagnostics": admixed_optional_diagnostics,
            "admixed_qc": admixed_qc,
            "mode": inputs.mode,
            "vcf_path": str(inputs.combined_vcf) if inputs.combined_vcf is not None else None,
            "hapstore_path": str(inputs.combined_hapstore) if inputs.combined_hapstore is not None else None,
            "population_vcf_paths": {k: str(v) for k, v in inputs.population_vcfs.items()},
            "population_hapstore_paths": {k: str(v) for k, v in inputs.population_hapstores.items()},
            "sample_manifest_path": str(inputs.sample_manifest),
            "pairwise_fst": pairwise,
            "donor_pairwise_fst": donor_pairwise,
            "within_split_fst": within,
            "ld_decay": ld_decay,
            "ld_decay_path": str(ld_decay_path) if ld_decay_path is not None else None,
            "king": king,
            "afs": afs,
            "hwe": hwe,
            "inbreeding_f": inbreeding_f,
            "pca": pca,
            "roh": roh,
            "plink2_parity": plink2_parity,
            "gate_pass": gate_pass,
        }
        elapsed = perf_counter() - t0
        stage_timings["write_diagnostics_sec"] = perf_counter() - t_write
        stage_timings["total_sec"] = elapsed
        payload["elapsed_sec"] = elapsed
        payload["stage_timings"] = stage_timings
        write_json(diagnostics_path, payload)
    logger.info(
        "QC complete chrom=%s backend=%s ld_backend=%s gate_pass=%s elapsed_sec=%.3f stage_timings=%s",
        chrom,
        config.qc.backend,
        config.qc.ld_decay_backend,
        gate_pass,
        elapsed,
        stage_timings,
    )

    return QcReport(
        chrom=chrom,
        vcf_path=inputs.combined_vcf,
        sample_manifest_path=inputs.sample_manifest,
        pairwise_fst=pairwise,
        within_split_fst=within,
        gate_pass=gate_pass,
        diagnostics_path=diagnostics_path,
        mode=inputs.mode,
        population_vcf_paths=inputs.population_vcfs,
        hapstore_path=inputs.combined_hapstore,
        population_hapstore_paths=inputs.population_hapstores,
        donor_pairwise_fst=donor_pairwise,
        ld_decay=ld_decay,
        ld_decay_table_path=(Path(str(ld_decay["table_path"])) if isinstance(ld_decay, dict) and "table_path" in ld_decay else None),
        elapsed_seconds=elapsed,
        stage_timings=stage_timings,
    )
