"""Admixed QC gates and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import RunConfig
from ..donors import load_manifest
from ..maps import load_genetic_map
from ..utils import derived_seed
from .types import QcInputs


def compute_msprime_tract_composition_gate(
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


def is_admixed_qc_enabled(
    config: RunConfig,
    population_msprime_is_admixed: dict[str, bool],
) -> bool:
    mode = str(config.qc.admixed_qc_mode).strip().lower()
    if mode == "off":
        return False
    return bool(any(bool(x) for x in population_msprime_is_admixed.values()))


def is_admixed_involved_pair(
    pair: tuple[str, str],
    population_msprime_is_admixed: dict[str, bool],
) -> bool:
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


def compute_admixed_tract_length_gate(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pop_names: list[str],
    diagnostics_dir: Path,
    resolve_map_path_fn: Callable[[RunConfig, int, str], Path],
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
                        map_path = resolve_map_path_fn(config, int(chrom), str(pop_name))
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


def compute_admixed_ancestry_variance_gate(
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
            exp_var = float((float(p) * (1.0 - float(p))) / (float(t_gen) * float(chrom_map_morgans)))
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
                "expected_model": "pulse_uniform_recomb_approx",
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


def apply_admixed_non_gating_to_metric(
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


def compute_admixed_pca_cline_diagnostic(
    *,
    config: RunConfig,
    chrom: int,
    ext_data: Any | None,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    pop_names: list[str],
    diagnostics_dir: Path,
    resolve_panel_source_fn: Callable[[RunConfig, int, str], tuple[str, Path]],
    load_sample_haplotypes_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    load_synthetic_population_haplotypes_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    subsample_ids_fn: Callable[[list[str], int, int], list[str]],
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
                donor_ids_by_source[src] = subsample_ids_fn(
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

            synth_ids = subsample_ids_fn(
                synth_ids_all,
                target_n=min(len(synth_ids_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_synth", pop_name, chrom)),
            )

            per_src_pos: dict[str, np.ndarray] = {}
            per_src_haps: dict[str, np.ndarray] = {}
            try:
                for src in sources:
                    src_kind, src_path = resolve_panel_source_fn(
                        config,
                        int(chrom),
                        str(src),
                    )
                    pos_src, haps_src = load_sample_haplotypes_fn(
                        config=config,
                        source_path=src_path,
                        chrom=int(chrom),
                        sample_ids=donor_ids_by_source[src],
                        source_kind=src_kind,
                    )
                    per_src_pos[src] = np.asarray(pos_src, dtype=np.int64)
                    per_src_haps[src] = np.asarray(haps_src, dtype=np.uint8, copy=False)

                pos_s, synth_raw = load_synthetic_population_haplotypes_fn(
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

            donor_labels: list[str] = []
            src_means_pc1: dict[str, float] = {}
            src_means_pc2: dict[str, float] = {}
            n_donors_by_source: dict[str, int] = {}
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

            _, _, vt = np.linalg.svd(g_all_std, full_matrices=False)
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
            donor_ids_a = subsample_ids_fn(
                donor_ids_a_all,
                target_n=min(len(donor_ids_a_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_source_donor", source_a, chrom)),
            )
            donor_ids_b = subsample_ids_fn(
                donor_ids_b_all,
                target_n=min(len(donor_ids_b_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_source_donor", source_b, chrom)),
            )
            synth_ids = subsample_ids_fn(
                synth_ids_all,
                target_n=min(len(synth_ids_all), cap),
                seed=int(derived_seed(int(config.qc.seed), "admixed_pca_synth", pop_name, chrom)),
            )
            try:
                source_kind_a, source_path_a = resolve_panel_source_fn(config, int(chrom), str(source_a))
                source_kind_b, source_path_b = resolve_panel_source_fn(config, int(chrom), str(source_b))
                pos_a, donor_a_raw = load_sample_haplotypes_fn(
                    config=config,
                    source_path=source_path_a,
                    chrom=int(chrom),
                    sample_ids=donor_ids_a,
                    source_kind=source_kind_a,
                )
                pos_b, donor_b_raw = load_sample_haplotypes_fn(
                    config=config,
                    source_path=source_path_b,
                    chrom=int(chrom),
                    sample_ids=donor_ids_b,
                    source_kind=source_kind_b,
                )
                pos_s, synth_raw = load_synthetic_population_haplotypes_fn(
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
        _, _, vt = np.linalg.svd(g_all_std, full_matrices=False)
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


def compute_admixed_king_diagnostic(
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
