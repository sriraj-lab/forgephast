"""Main QC orchestration support."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from ..config import RunConfig
from ..donors import load_manifest
from ..telemetry import telemetry_span
from ..utils import ensure_dir
from . import (
    admixed_support,
    data_support,
    fst_gate,
    gate_support,
    matrix_backend,
    metric_support,
    policy_support,
    report_support,
)
from .registry import METRIC_REGISTRY
from .types import QcReport


def donor_baseline_fst(
    config: RunConfig,
    chrom: int,
    *,
    pairs: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    return fst_gate.donor_baseline_fst_impl(
        config=config,
        chrom=int(chrom),
        pairs=pairs,
        load_manifest_fn=load_manifest,
        resolve_pairwise_sets_fn=policy_support.resolve_pairwise_sets,
        resolve_donor_panel_source_for_population_fn=data_support.resolve_donor_panel_source,
        load_sample_haplotypes_from_panel_fn=data_support.load_sample_haplotypes_from_panel,
        resolve_python_qc_numba_threads_fn=policy_support.resolve_python_qc_numba_threads,
    )


def _compute_stochastic_donor_fst_gate(**kwargs: Any) -> dict[str, Any]:
    return fst_gate.compute_stochastic_donor_fst_gate_impl(
        **kwargs,
        resolve_donor_panel_source_for_population_fn=data_support.resolve_donor_panel_source,
        load_sample_haplotypes_from_panel_fn=data_support.load_sample_haplotypes_from_panel,
    )


def _compute_admixed_tract_length_gate(**kwargs: Any) -> dict[str, Any]:
    return admixed_support.compute_admixed_tract_length_gate(
        **kwargs,
        resolve_map_path_fn=data_support.resolve_map_path,
    )


def _compute_admixed_pca_cline_diagnostic(**kwargs: Any) -> dict[str, Any]:
    return admixed_support.compute_admixed_pca_cline_diagnostic(
        **kwargs,
        resolve_panel_source_fn=data_support.resolve_donor_panel_source,
        load_sample_haplotypes_fn=data_support.load_sample_haplotypes_from_panel,
        load_synthetic_population_haplotypes_fn=data_support.load_synthetic_population_haplotypes,
        subsample_ids_fn=data_support.subsample_ids,
    )


def run_qc_impl(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    logger: logging.Logger,
) -> QcReport:
    t0 = perf_counter()
    stage_timings: dict[str, float] = {}
    with telemetry_span(logger, "qc.resolve_inputs", chrom=int(chrom), n_shards=int(n_shards)):
        inputs = data_support.resolve_synthetic_outputs(config, chrom=int(chrom), n_shards=int(n_shards))
        groups = data_support.sample_groups_from_manifest(inputs.sample_manifest)
        pop_names = sorted(groups.keys())
        pairs = policy_support.resolve_pairwise_sets(config, pop_names)
        within_checks = policy_support.resolve_within_split_checks(config, pop_names)
        python_qc_numba_threads = policy_support.resolve_python_qc_numba_threads(config)

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
    admixed_qc_enabled = admixed_support.is_admixed_qc_enabled(config, population_msprime_is_admixed)
    source_source_pairs = [
        pair
        for pair in pairs
        if not admixed_support.is_admixed_involved_pair(pair, population_msprime_is_admixed)
    ]
    admixed_involved_pairs = [
        pair
        for pair in pairs
        if admixed_support.is_admixed_involved_pair(pair, population_msprime_is_admixed)
    ]

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
        pairwise, within = matrix_backend.run_qc_matrix_hapstore_impl(
            config=config,
            chrom=int(chrom),
            inputs=inputs,
            groups=groups,
            pairs=pairs,
            within_checks=within_checks,
            numba_threads=python_qc_numba_threads,
            logger=logger,
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
    fst_gate_tolerance = float(config.qc.fst_tolerance_relative)
    if (
        config.qc.backend == "python"
        and config.qc.main_pairwise_mode == "sampled"
        and (inputs.combined_hapstore is not None or inputs.population_hapstores)
    ):
        fst_gate_tolerance = float(config.qc.sampled_gate_tolerance_relative)
    diagnostics_dir = ensure_dir(config.run_root / "diagnostics")
    pair_key_to_pair = {policy_support.pair_key(a, b): (a, b) for a, b in pairs}
    with telemetry_span(logger, "qc.donor_gate", chrom=int(chrom)):
        gate_eval = gate_support.evaluate_main_qc_gates(
            config=config,
            chrom=int(chrom),
            n_shards=int(n_shards),
            pairwise=pairwise,
            within=within,
            groups=groups,
            pairs=pairs,
            pop_names=pop_names,
            source_source_pairs=source_source_pairs,
            admixed_involved_pairs=admixed_involved_pairs,
            pair_key_to_pair=pair_key_to_pair,
            population_msprime_is_admixed=population_msprime_is_admixed,
            admixed_qc_enabled=admixed_qc_enabled,
            diagnostics_dir=diagnostics_dir,
            fst_gate_tolerance=fst_gate_tolerance,
            python_qc_numba_threads=python_qc_numba_threads,
            donor_baseline_fst_fn=donor_baseline_fst,
            load_manifest_fn=load_manifest,
            stochastic_gate_activation_fn=fst_gate.resolve_fst_stochastic_gate_activation_impl,
            stochastic_gate_fn=_compute_stochastic_donor_fst_gate,
            msprime_tract_composition_gate_fn=admixed_support.compute_msprime_tract_composition_gate,
            admixed_tract_length_gate_fn=_compute_admixed_tract_length_gate,
            admixed_ancestry_variance_gate_fn=admixed_support.compute_admixed_ancestry_variance_gate,
            is_admixed_involved_pair_fn=admixed_support.is_admixed_involved_pair,
            logger=logger,
        )
        gate_pass = gate_eval.gate_pass
        donor_pairwise = gate_eval.donor_pairwise
        fst_gate_standard_pass = gate_eval.fst_gate_standard_pass
        fst_gate_effective_pass = gate_eval.fst_gate_effective_pass
        fst_gate_strategy_used = gate_eval.fst_gate_strategy_used
        fst_stochastic_gate_activation = gate_eval.fst_stochastic_gate_activation
        fst_stochastic_gate = gate_eval.fst_stochastic_gate
        fst_pairwise_gate_status = gate_eval.fst_pairwise_gate_status
        within_split_gate_status = gate_eval.within_split_gate_status
        msprime_tract_composition_gate = gate_eval.msprime_tract_composition_gate
        admixed_tract_length_gate = gate_eval.admixed_tract_length_gate
        admixed_ancestry_variance_gate = gate_eval.admixed_ancestry_variance_gate
        stage_timings.update(gate_eval.stage_timings)

    ext_context = metric_support.prepare_extended_metric_context(
        config=config,
        chrom=int(chrom),
        groups=groups,
        inputs=inputs,
        load_manifest_fn=load_manifest,
        subsample_ids_fn=data_support.subsample_ids,
        resolve_panel_source_for_population_fn=lambda c, pop: data_support.resolve_donor_panel_source(
            config, int(c), str(pop)
        ),
        load_sample_haplotypes_from_panel_source_fn=lambda kind, source_path, c, sample_ids: (
            data_support.load_sample_haplotypes_from_panel(
                config=config,
                source_path=Path(source_path),
                chrom=int(c),
                sample_ids=list(sample_ids),
                source_kind=str(kind),
            )
        ),
        load_synthetic_population_haplotypes_fn=lambda inps, c, pop, sample_ids: (
            data_support.load_synthetic_population_haplotypes(
                inputs=inps,
                chrom=int(c),
                pop=str(pop),
                sample_ids=list(sample_ids),
            )
        ),
        logger=logger,
    )
    metric_eval = metric_support.run_registered_metrics(
        config=config,
        chrom=int(chrom),
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=python_qc_numba_threads,
        metric_registry=METRIC_REGISTRY,
        ext_context=ext_context,
        admixed_qc_enabled=admixed_qc_enabled,
        admixed_pop_set=admixed_pop_set,
        apply_admixed_non_gating_fn=admixed_support.apply_admixed_non_gating_to_metric,
        initial_gate_pass=gate_pass,
        logger=logger,
    )
    metric_payloads = metric_eval.payloads
    gate_pass = metric_eval.gate_pass
    ld_decay_path = metric_eval.ld_decay_path
    stage_timings.update(metric_eval.stage_timings)

    admixed_optional_diagnostics, admixed_diag_timings = report_support.compute_admixed_optional_diagnostics(
        config=config,
        chrom=int(chrom),
        admixed_qc_enabled=admixed_qc_enabled,
        ext_data=ext_context.ext_data,
        inputs=inputs,
        groups=groups,
        admixed_pop_names=admixed_pop_names,
        diagnostics_dir=diagnostics_dir,
        king_payload=metric_payloads["king"],
        compute_pca_cline_fn=_compute_admixed_pca_cline_diagnostic,
        compute_king_fn=admixed_support.compute_admixed_king_diagnostic,
    )
    stage_timings.update(admixed_diag_timings)

    population_status = report_support.build_population_status(
        pop_names=pop_names,
        population_spec_by_name=population_spec_by_name,
        population_engine=population_engine,
        population_msprime_is_admixed=population_msprime_is_admixed,
    )
    admixed_qc, hard_gate_pass = report_support.build_admixed_qc_summary(
        config=config,
        admixed_qc_enabled=admixed_qc_enabled,
        admixed_pop_names=admixed_pop_names,
        pairs=pairs,
        source_source_pairs=source_source_pairs,
        admixed_involved_pairs=admixed_involved_pairs,
        population_status=population_status,
        msprime_tract_composition_gate=msprime_tract_composition_gate,
        admixed_tract_length_gate=admixed_tract_length_gate,
        admixed_ancestry_variance_gate=admixed_ancestry_variance_gate,
        admixed_optional_diagnostics=admixed_optional_diagnostics,
    )
    if hard_gate_pass is False:
        gate_pass = False

    return report_support.write_qc_diagnostics_and_finalize(
        logger=logger,
        chrom=int(chrom),
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
        synthetic_shard_round_robin=metric_payloads["synthetic_shard_round_robin"],
        ld_decay=metric_payloads["ld_decay"],
        ld_decay_path=ld_decay_path,
        king=metric_payloads["king"],
        afs=metric_payloads["afs"],
        hwe=metric_payloads["hwe"],
        inbreeding_f=metric_payloads["inbreeding_f"],
        pca=metric_payloads["pca"],
        roh=metric_payloads["roh"],
        gate_pass=gate_pass,
        diagnostics_dir=diagnostics_dir,
        stage_timings=stage_timings,
        t0=t0,
    )
