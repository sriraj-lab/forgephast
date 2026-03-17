"""QC diagnostics assembly and report-writing support."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from ..telemetry import telemetry_span
from ..utils import write_json
from .types import QcInputs, QcReport


def compute_admixed_optional_diagnostics(
    *,
    config: Any,
    chrom: int,
    admixed_qc_enabled: bool,
    ext_data: Any | None,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    admixed_pop_names: list[str],
    diagnostics_dir: Path,
    king_payload: dict[str, Any] | None,
    compute_pca_cline_fn: Callable[..., dict[str, Any]],
    compute_king_fn: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    diagnostics: dict[str, Any] = {}
    timings: dict[str, float] = {}
    if admixed_qc_enabled:
        t_diag_pca = perf_counter()
        try:
            diagnostics["pca_cline"] = compute_pca_cline_fn(
                config=config,
                chrom=int(chrom),
                ext_data=ext_data,
                inputs=inputs,
                groups=groups,
                pop_names=admixed_pop_names,
                diagnostics_dir=diagnostics_dir,
            )
        except Exception as exc:
            diagnostics["pca_cline"] = {
                "enabled": bool(config.qc.admixed_diag_pca_cline_enabled),
                "status": "error",
                "error": str(exc),
            }
        timings["admixed_diag_pca_cline_sec"] = perf_counter() - t_diag_pca

        t_diag_king = perf_counter()
        try:
            diagnostics["king"] = compute_king_fn(
                config=config,
                king_payload=king_payload,
                pop_names=admixed_pop_names,
            )
        except Exception as exc:
            diagnostics["king"] = {
                "enabled": bool(config.qc.admixed_diag_king_enabled),
                "status": "error",
                "error": str(exc),
            }
        timings["admixed_diag_king_sec"] = perf_counter() - t_diag_king

        if bool(config.qc.admixed_diag_stratified_ld_enabled):
            diagnostics["stratified_ld"] = {
                "enabled": True,
                "status": "not_implemented",
                "gate_pass": None,
            }
        else:
            diagnostics["stratified_ld"] = {
                "enabled": False,
                "status": "disabled_by_config",
                "gate_pass": None,
            }
    else:
        diagnostics["pca_cline"] = {
            "enabled": bool(config.qc.admixed_diag_pca_cline_enabled),
            "status": "disabled_admixed_qc_policy",
        }
        diagnostics["king"] = {
            "enabled": bool(config.qc.admixed_diag_king_enabled),
            "status": "disabled_admixed_qc_policy",
        }
        diagnostics["stratified_ld"] = {
            "enabled": bool(config.qc.admixed_diag_stratified_ld_enabled),
            "status": "disabled_admixed_qc_policy",
            "gate_pass": None,
        }
    return diagnostics, timings


def build_population_status(
    *,
    pop_names: list[str],
    population_spec_by_name: dict[str, Any],
    population_engine: dict[str, str],
    population_msprime_is_admixed: dict[str, bool],
) -> dict[str, dict[str, Any]]:
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
    return population_status


def build_admixed_qc_summary(
    *,
    config: Any,
    admixed_qc_enabled: bool,
    admixed_pop_names: list[str],
    pairs: list[tuple[str, str]],
    source_source_pairs: list[tuple[str, str]],
    admixed_involved_pairs: list[tuple[str, str]],
    population_status: dict[str, dict[str, Any]],
    msprime_tract_composition_gate: dict[str, Any] | None,
    admixed_tract_length_gate: dict[str, Any] | None,
    admixed_ancestry_variance_gate: dict[str, Any] | None,
    admixed_optional_diagnostics: dict[str, Any],
) -> tuple[dict[str, Any], bool | None]:
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

    return {
        "enabled": bool(admixed_qc_enabled),
        "mode": str(config.qc.admixed_qc_mode),
        "population_status": population_status,
        "hard_gate_components": hard_gate_components,
        "hard_gate_pass": hard_gate_pass,
        "skipped_components": skipped_components,
        "optional_diagnostics": admixed_optional_diagnostics,
    }, hard_gate_pass


def build_qc_payload(
    *,
    chrom: int,
    config: Any,
    inputs: QcInputs,
    python_qc_numba_threads: int,
    fst_gate_tolerance: float,
    fst_gate_strategy_used: str,
    fst_gate_standard_pass: bool | None,
    fst_gate_effective_pass: bool | None,
    fst_stochastic_gate_activation: dict[str, Any] | None,
    fst_stochastic_gate: dict[str, Any] | None,
    fst_pairwise_gate_status: dict[str, Any],
    within_split_gate_status: dict[str, Any],
    population_engine: dict[str, str],
    population_msprime_is_admixed: dict[str, bool],
    msprime_tract_composition_gate: dict[str, Any] | None,
    admixed_tract_length_gate: dict[str, Any] | None,
    admixed_ancestry_variance_gate: dict[str, Any] | None,
    admixed_optional_diagnostics: dict[str, Any],
    admixed_qc: dict[str, Any] | None,
    pairwise: dict[str, float],
    donor_pairwise: dict[str, float] | None,
    within: dict[str, float],
    synthetic_shard_round_robin: dict[str, Any] | None,
    ld_decay: dict[str, Any] | None,
    ld_decay_path: Path | None,
    king: dict[str, Any] | None,
    afs: dict[str, Any] | None,
    hwe: dict[str, Any] | None,
    inbreeding_f: dict[str, Any] | None,
    pca: dict[str, Any] | None,
    roh: dict[str, Any] | None,
    gate_pass: bool,
    stage_timings: dict[str, float],
    elapsed: float,
) -> dict[str, Any]:
    payload = {
        "chrom": chrom,
        "backend": config.qc.backend,
        "ld_decay_backend": config.qc.ld_decay_backend,
        "main_pairwise_mode": config.qc.main_pairwise_mode,
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
        "population_hapstore_paths": {k: str(v) for k, v in inputs.population_hapstores.items()},
        "sample_manifest_path": str(inputs.sample_manifest),
        "pairwise_fst": pairwise,
        "donor_pairwise_fst": donor_pairwise,
        "within_split_fst": within,
        "synthetic_shard_round_robin": synthetic_shard_round_robin,
        "ld_decay": ld_decay,
        "ld_decay_path": str(ld_decay_path) if ld_decay_path is not None else None,
        "king": king,
        "afs": afs,
        "hwe": hwe,
        "inbreeding_f": inbreeding_f,
        "pca": pca,
        "roh": roh,
        "gate_pass": gate_pass,
        "elapsed_sec": elapsed,
        "stage_timings": stage_timings,
    }
    return payload


def finalize_qc_report(
    *,
    chrom: int,
    inputs: QcInputs,
    pairwise: dict[str, float],
    within: dict[str, float],
    gate_pass: bool,
    diagnostics_path: Path,
    donor_pairwise: dict[str, float] | None,
    ld_decay: dict[str, Any] | None,
    elapsed: float,
    stage_timings: dict[str, float],
) -> QcReport:
    return QcReport(
        chrom=chrom,
        vcf_path=inputs.combined_vcf,
        sample_manifest_path=inputs.sample_manifest,
        pairwise_fst=pairwise,
        within_split_fst=within,
        gate_pass=gate_pass,
        diagnostics_path=diagnostics_path,
        mode=inputs.mode,
        hapstore_path=inputs.combined_hapstore,
        population_hapstore_paths=inputs.population_hapstores,
        donor_pairwise_fst=donor_pairwise,
        ld_decay=ld_decay,
        ld_decay_table_path=(Path(str(ld_decay["table_path"])) if isinstance(ld_decay, dict) and "table_path" in ld_decay else None),
        elapsed_seconds=elapsed,
        stage_timings=stage_timings,
    )


def write_qc_diagnostics_and_finalize(
    *,
    logger: logging.Logger,
    chrom: int,
    config: Any,
    inputs: QcInputs,
    python_qc_numba_threads: int,
    fst_gate_tolerance: float,
    fst_gate_strategy_used: str,
    fst_gate_standard_pass: bool | None,
    fst_gate_effective_pass: bool | None,
    fst_stochastic_gate_activation: dict[str, Any] | None,
    fst_stochastic_gate: dict[str, Any] | None,
    fst_pairwise_gate_status: dict[str, Any],
    within_split_gate_status: dict[str, Any],
    population_engine: dict[str, str],
    population_msprime_is_admixed: dict[str, bool],
    msprime_tract_composition_gate: dict[str, Any] | None,
    admixed_tract_length_gate: dict[str, Any] | None,
    admixed_ancestry_variance_gate: dict[str, Any] | None,
    admixed_optional_diagnostics: dict[str, Any],
    admixed_qc: dict[str, Any] | None,
    pairwise: dict[str, float],
    donor_pairwise: dict[str, float] | None,
    within: dict[str, float],
    synthetic_shard_round_robin: dict[str, Any] | None,
    ld_decay: dict[str, Any] | None,
    ld_decay_path: Path | None,
    king: dict[str, Any] | None,
    afs: dict[str, Any] | None,
    hwe: dict[str, Any] | None,
    inbreeding_f: dict[str, Any] | None,
    pca: dict[str, Any] | None,
    roh: dict[str, Any] | None,
    gate_pass: bool,
    diagnostics_dir: Path,
    stage_timings: dict[str, float],
    t0: float,
) -> QcReport:
    diagnostics_path = diagnostics_dir / f"qc_chr{chrom}.json"
    with telemetry_span(logger, "qc.write_diagnostics", chrom=int(chrom)):
        t_write = perf_counter()
        elapsed = perf_counter() - t0
        payload = build_qc_payload(
            chrom=chrom,
            config=config,
            inputs=inputs,
            python_qc_numba_threads=python_qc_numba_threads,
            fst_gate_tolerance=fst_gate_tolerance,
            fst_gate_strategy_used=fst_gate_strategy_used,
            fst_gate_standard_pass=fst_gate_standard_pass,
            fst_gate_effective_pass=fst_gate_effective_pass,
            fst_stochastic_gate_activation=fst_stochastic_gate_activation,
            fst_stochastic_gate=fst_stochastic_gate,
            fst_pairwise_gate_status=fst_pairwise_gate_status,
            within_split_gate_status=within_split_gate_status,
            population_engine=population_engine,
            population_msprime_is_admixed=population_msprime_is_admixed,
            msprime_tract_composition_gate=msprime_tract_composition_gate,
            admixed_tract_length_gate=admixed_tract_length_gate,
            admixed_ancestry_variance_gate=admixed_ancestry_variance_gate,
            admixed_optional_diagnostics=admixed_optional_diagnostics,
            admixed_qc=admixed_qc,
            pairwise=pairwise,
            donor_pairwise=donor_pairwise,
            within=within,
            synthetic_shard_round_robin=synthetic_shard_round_robin,
            ld_decay=ld_decay,
            ld_decay_path=ld_decay_path,
            king=king,
            afs=afs,
            hwe=hwe,
            inbreeding_f=inbreeding_f,
            pca=pca,
            roh=roh,
            gate_pass=gate_pass,
            stage_timings=stage_timings,
            elapsed=elapsed,
        )
        write_json(diagnostics_path, payload)
        stage_timings["write_diagnostics_sec"] = perf_counter() - t_write
        stage_timings["total_sec"] = elapsed

    logger.info(
        "QC complete chrom=%s backend=%s ld_backend=%s gate_pass=%s elapsed_sec=%.3f stage_timings=%s",
        chrom,
        config.qc.backend,
        config.qc.ld_decay_backend,
        gate_pass,
        elapsed,
        stage_timings,
    )

    return finalize_qc_report(
        chrom=chrom,
        inputs=inputs,
        pairwise=pairwise,
        within=within,
        gate_pass=gate_pass,
        diagnostics_path=diagnostics_path,
        donor_pairwise=donor_pairwise,
        ld_decay=ld_decay,
        elapsed=elapsed,
        stage_timings=stage_timings,
    )
