"""Main QC orchestration support."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .types import QcReport


def run_qc_impl(
    *,
    config: Any,
    chrom: int,
    n_shards: int,
    logger: logging.Logger,
    t0: float,
    stage_timings: dict[str, float],
    resolve_synthetic_outputs_fn: Callable[[Any, int, int], Any],
    sample_groups_from_manifest_fn: Callable[[Path], dict[str, list[str]]],
    resolve_pairwise_sets_fn: Callable[[Any, list[str]], list[tuple[str, str]]],
    resolve_within_split_checks_fn: Callable[[Any, list[str]], list[str]],
    resolve_python_qc_numba_threads_fn: Callable[[Any], int],
    is_admixed_qc_enabled_fn: Callable[[Any, dict[str, bool]], bool],
    is_admixed_involved_pair_fn: Callable[[tuple[str, str], dict[str, bool]], bool],
    run_qc_matrix_hapstore_fn: Callable[..., tuple[dict[str, float], dict[str, float]]],
    evaluate_main_qc_gates_fn: Callable[..., Any],
    prepare_extended_metric_context_fn: Callable[..., Any],
    run_registered_metrics_fn: Callable[..., Any],
    compute_admixed_optional_diagnostics_fn: Callable[..., tuple[dict[str, Any], dict[str, float]]],
    build_population_status_fn: Callable[..., dict[str, dict[str, Any]]],
    build_admixed_qc_summary_fn: Callable[..., tuple[dict[str, Any], bool | None]],
    write_qc_diagnostics_and_finalize_fn: Callable[..., QcReport],
    load_manifest_fn: Callable[[Any], Any],
    apply_admixed_non_gating_fn: Callable[..., dict[str, Any]],
    resolve_donor_panel_source_for_population_fn: Callable[[int, str], tuple[str, Path]],
    load_sample_haplotypes_from_panel_fn: Callable[..., tuple[Any, Any]],
    load_synthetic_population_haplotypes_fn: Callable[..., tuple[Any, Any]],
    donor_baseline_fst_fn: Callable[..., dict[str, float]],
    resolve_fst_stochastic_gate_activation_fn: Callable[..., dict[str, Any] | None],
    compute_stochastic_donor_fst_gate_fn: Callable[..., dict[str, Any]],
    compute_msprime_tract_composition_gate_fn: Callable[..., dict[str, Any]],
    compute_admixed_tract_length_gate_fn: Callable[..., dict[str, Any]],
    compute_admixed_ancestry_variance_gate_fn: Callable[..., dict[str, Any]],
    compute_admixed_pca_cline_diagnostic_fn: Callable[..., dict[str, Any]],
    compute_admixed_king_diagnostic_fn: Callable[..., dict[str, Any]],
    telemetry_span_fn: Callable[..., Any],
    ensure_dir_fn: Callable[[Path], Path],
    subsample_ids_fn: Callable[[list[str], int, int], list[str]],
    metric_registry: tuple[Any, ...],
) -> QcReport:
    with telemetry_span_fn(logger, "qc.resolve_inputs", chrom=int(chrom), n_shards=int(n_shards)):
        inputs = resolve_synthetic_outputs_fn(config, chrom=int(chrom), n_shards=int(n_shards))
        groups = sample_groups_from_manifest_fn(inputs.sample_manifest)
        pop_names = sorted(groups.keys())
        pairs = resolve_pairwise_sets_fn(config, pop_names)
        within_checks = resolve_within_split_checks_fn(config, pop_names)
        python_qc_numba_threads = resolve_python_qc_numba_threads_fn(config)

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
    admixed_qc_enabled = is_admixed_qc_enabled_fn(config, population_msprime_is_admixed)
    source_source_pairs = [pair for pair in pairs if not is_admixed_involved_pair_fn(pair, population_msprime_is_admixed)]
    admixed_involved_pairs = [pair for pair in pairs if is_admixed_involved_pair_fn(pair, population_msprime_is_admixed)]

    with telemetry_span_fn(
        logger,
        "qc.compute_fst",
        chrom=int(chrom),
        backend=str(config.qc.backend),
        mode=str(inputs.mode),
        n_pairs=int(len(pairs)),
        n_within=int(len(within_checks)),
    ):
        t_pairwise = perf_counter()
        pairwise, within = run_qc_matrix_hapstore_fn(
            config=config,
            chrom=int(chrom),
            inputs=inputs,
            groups=groups,
            pairs=pairs,
            within_checks=within_checks,
            numba_threads=python_qc_numba_threads,
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
    diagnostics_dir = ensure_dir_fn(config.run_root / "diagnostics")
    pair_key_to_pair = {f"{a}__{b}" if a <= b else f"{b}__{a}": (a, b) for a, b in pairs}
    with telemetry_span_fn(logger, "qc.donor_gate", chrom=int(chrom)):
        gate_eval = evaluate_main_qc_gates_fn(
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
            donor_baseline_fst_fn=donor_baseline_fst_fn,
            load_manifest_fn=load_manifest_fn,
            stochastic_gate_activation_fn=resolve_fst_stochastic_gate_activation_fn,
            stochastic_gate_fn=compute_stochastic_donor_fst_gate_fn,
            msprime_tract_composition_gate_fn=compute_msprime_tract_composition_gate_fn,
            admixed_tract_length_gate_fn=compute_admixed_tract_length_gate_fn,
            admixed_ancestry_variance_gate_fn=compute_admixed_ancestry_variance_gate_fn,
            is_admixed_involved_pair_fn=is_admixed_involved_pair_fn,
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

    ext_context = prepare_extended_metric_context_fn(
        config=config,
        chrom=int(chrom),
        groups=groups,
        inputs=inputs,
        load_manifest_fn=load_manifest_fn,
        subsample_ids_fn=lambda ids, target_n, seed: subsample_ids_fn(list(ids), int(target_n), int(seed)),
        resolve_panel_source_for_population_fn=lambda c, pop: resolve_donor_panel_source_for_population_fn(int(c), str(pop)),
        load_sample_haplotypes_from_panel_source_fn=lambda kind, source_path, c, sample_ids: (
            load_sample_haplotypes_from_panel_fn(
                config=config,
                source_path=Path(source_path),
                chrom=int(c),
                sample_ids=list(sample_ids),
                source_kind=str(kind),
            )
        ),
        load_synthetic_population_haplotypes_fn=lambda inps, c, pop, sample_ids: (
            load_synthetic_population_haplotypes_fn(
                config=config,
                inputs=inps,
                chrom=int(c),
                pop=str(pop),
                sample_ids=list(sample_ids),
            )
        ),
        logger=logger,
    )
    metric_eval = run_registered_metrics_fn(
        config=config,
        chrom=int(chrom),
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=python_qc_numba_threads,
        metric_registry=metric_registry,
        ext_context=ext_context,
        admixed_qc_enabled=admixed_qc_enabled,
        admixed_pop_set=admixed_pop_set,
        apply_admixed_non_gating_fn=apply_admixed_non_gating_fn,
        initial_gate_pass=gate_pass,
        logger=logger,
    )
    metric_payloads = metric_eval.payloads
    gate_pass = metric_eval.gate_pass
    ld_decay_path = metric_eval.ld_decay_path
    stage_timings.update(metric_eval.stage_timings)

    admixed_optional_diagnostics, admixed_diag_timings = compute_admixed_optional_diagnostics_fn(
        config=config,
        chrom=int(chrom),
        admixed_qc_enabled=admixed_qc_enabled,
        ext_data=ext_context.ext_data,
        inputs=inputs,
        groups=groups,
        admixed_pop_names=admixed_pop_names,
        diagnostics_dir=diagnostics_dir,
        king_payload=metric_payloads["king"],
        compute_pca_cline_fn=compute_admixed_pca_cline_diagnostic_fn,
        compute_king_fn=compute_admixed_king_diagnostic_fn,
    )
    stage_timings.update(admixed_diag_timings)

    population_status = build_population_status_fn(
        pop_names=pop_names,
        population_spec_by_name=population_spec_by_name,
        population_engine=population_engine,
        population_msprime_is_admixed=population_msprime_is_admixed,
    )
    admixed_qc, hard_gate_pass = build_admixed_qc_summary_fn(
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

    return write_qc_diagnostics_and_finalize_fn(
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
