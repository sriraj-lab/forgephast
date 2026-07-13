"""Shared extended-metric context and registry dispatch for QC."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .context import build_ext_aligned_data


@dataclass(frozen=True)
class ExtendedMetricContext:
    donors_by_population: dict[str, list[str]]
    donor_manifest_error: str | None
    ext_data: Any | None
    ext_data_error: str | None


@dataclass(frozen=True)
class MetricEvaluation:
    payloads: dict[str, dict[str, Any] | None]
    gate_pass: bool
    ld_decay_path: Path | None
    stage_timings: dict[str, float]


def _extended_metrics_requested(config: Any) -> bool:
    return any(
        (
            bool(config.qc.king_enabled),
            bool(config.qc.afs_enabled),
            bool(config.qc.pca_enabled),
            bool(config.qc.roh_enabled),
            bool(config.qc.hwe_enabled),
            bool(config.qc.inbreeding_f_enabled),
        )
    )


def prepare_extended_metric_context(
    *,
    config: Any,
    chrom: int,
    groups: dict[str, list[str]],
    inputs: Any,
    load_manifest_fn: Callable[[Any], Any],
    subsample_ids_fn: Callable[[list[str], int, int], list[str]],
    resolve_panel_source_for_population_fn: Callable[[int, str], tuple[str, Any]],
    load_sample_haplotypes_from_panel_source_fn: Callable[[str, Any, int, list[str]], tuple[Any, Any]],
    load_synthetic_population_haplotypes_fn: Callable[[Any, int, str, list[str]], tuple[Any, Any]],
    logger: logging.Logger,
) -> ExtendedMetricContext:
    donors_by_population: dict[str, list[str]] = {}
    donor_manifest_error: str | None = None
    if _extended_metrics_requested(config):
        try:
            manifest = load_manifest_fn(config)
            donors_by_population = manifest.donors_by_population
        except Exception as exc:
            donor_manifest_error = str(exc)
            logger.warning("Failed to load donor manifest for extended QC metrics: %s", exc)

    ext_data = None
    ext_data_error: str | None = None
    if _extended_metrics_requested(config):
        try:
            if donor_manifest_error is not None:
                raise RuntimeError(f"donor manifest unavailable: {donor_manifest_error}")
            ext_data = build_ext_aligned_data(
                chrom=int(chrom),
                seed=int(config.qc.seed),
                ext_max_diploids_per_population=int(config.qc.ext_max_diploids_per_population),
                ext_max_variants=int(config.qc.ext_max_variants),
                groups=groups,
                donors_by_population=donors_by_population,
                inputs=inputs,
                subsample_ids=lambda ids, target_n, s: subsample_ids_fn(
                    list(ids), int(target_n), int(s)
                ),
                resolve_panel_source_for_population=lambda c, pop: resolve_panel_source_for_population_fn(
                    int(c), str(pop)
                ),
                load_sample_haplotypes_from_panel_source=lambda kind, source_path, c, sample_ids: (
                    load_sample_haplotypes_from_panel_source_fn(
                        str(kind),
                        source_path,
                        int(c),
                        list(sample_ids),
                    )
                ),
                load_synthetic_population_haplotypes=lambda inps, c, pop, sample_ids: (
                    load_synthetic_population_haplotypes_fn(
                        inps,
                        int(c),
                        str(pop),
                        list(sample_ids),
                    )
                ),
            )
        except Exception as exc:
            ext_data_error = str(exc)
            logger.warning("Failed to build extended-QC shared sampling cache: %s", exc)

    return ExtendedMetricContext(
        donors_by_population=donors_by_population,
        donor_manifest_error=donor_manifest_error,
        ext_data=ext_data,
        ext_data_error=ext_data_error,
    )


def run_registered_metrics(
    *,
    config: Any,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    metric_registry: tuple[Any, ...],
    ext_context: ExtendedMetricContext,
    admixed_qc_enabled: bool,
    admixed_pop_set: set[str],
    apply_admixed_non_gating_fn: Callable[..., dict[str, Any]],
    initial_gate_pass: bool,
    logger: logging.Logger,
) -> MetricEvaluation:
    metric_payloads: dict[str, dict[str, Any] | None] = {
        "synthetic_shard_round_robin": None,
        "ld_decay": None,
        "king": None,
        "afs": None,
        "hwe": None,
        "inbreeding_f": None,
        "pca": None,
        "roh": None,
    }
    gate_pass = bool(initial_gate_pass)
    stage_timings: dict[str, float] = {}

    for spec in metric_registry:
        if not spec.enabled(config):
            continue
        t_metric = perf_counter()
        try:
            if ext_context.donor_manifest_error is not None and spec.payload_key != "ld_decay":
                raise RuntimeError(f"donor manifest unavailable: {ext_context.donor_manifest_error}")
            metric_payload = spec.runner(
                config=config,
                chrom=chrom,
                inputs=inputs,
                groups=groups,
                donors_by_population=ext_context.donors_by_population,
                diagnostics_dir=diagnostics_dir,
                numba_threads=numba_threads,
                ext_data=ext_context.ext_data,
                ext_data_error=ext_context.ext_data_error,
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
                metric_payload["gate_pass"] = False
                gate_pass = False

        if (
            admixed_qc_enabled
            and spec.payload_key in {"hwe", "inbreeding_f"}
            and str(config.qc.admixed_hwe_inbreeding_mode).strip().lower() == "non_gating"
        ):
            metric_payload = apply_admixed_non_gating_fn(
                metric_payload,
                admixed_pop_names=admixed_pop_set,
                reason="non_gating_admixed_policy",
            )

        metric_payloads[spec.payload_key] = metric_payload
        if spec.contributes_gate and isinstance(metric_payload, dict) and metric_payload.get("gate_pass") is False:
            gate_pass = False
        stage_timings[f"{spec.payload_key}_sec"] = perf_counter() - t_metric

    ld_decay = metric_payloads["ld_decay"]
    ld_decay_path: Path | None = None
    if isinstance(ld_decay, dict) and ld_decay.get("status") != "error":
        ld_decay_path = diagnostics_dir / f"ld_decay_chr{chrom}.json"
    if not bool(config.qc.ld_decay_enabled):
        stage_timings.setdefault("ld_decay_sec", 0.0)

    return MetricEvaluation(
        payloads=metric_payloads,
        gate_pass=gate_pass,
        ld_decay_path=ld_decay_path,
        stage_timings=stage_timings,
    )
