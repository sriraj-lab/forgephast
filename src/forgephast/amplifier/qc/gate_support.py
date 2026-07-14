"""Main QC gate evaluation support."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Callable

import numpy as np

from ..config import RunConfig


@dataclass
class MainGateEvaluation:
    gate_pass: bool
    donor_pairwise: dict[str, float] | None
    fst_gate_standard_pass: bool | None
    fst_gate_effective_pass: bool | None
    fst_gate_strategy_used: str
    fst_stochastic_gate_activation: dict[str, Any] | None
    fst_stochastic_gate: dict[str, Any] | None
    fst_pairwise_gate_status: dict[str, Any]
    within_split_gate_status: dict[str, Any]
    msprime_tract_composition_gate: dict[str, Any] | None
    admixed_tract_length_gate: dict[str, Any] | None
    admixed_ancestry_variance_gate: dict[str, Any] | None
    stage_timings: dict[str, float]


def evaluate_main_qc_gates(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pairwise: dict[str, float],
    within: dict[str, float],
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    pop_names: list[str],
    source_source_pairs: list[tuple[str, str]],
    admixed_involved_pairs: list[tuple[str, str]],
    pair_key_to_pair: dict[str, tuple[str, str]],
    population_msprime_is_admixed: dict[str, bool],
    admixed_qc_enabled: bool,
    diagnostics_dir: Any,
    fst_gate_tolerance: float,
    python_qc_numba_threads: int,
    donor_baseline_fst_fn: Callable[..., dict[str, float]],
    load_manifest_fn: Callable[[RunConfig], Any],
    stochastic_gate_activation_fn: Callable[..., dict[str, Any]],
    stochastic_gate_fn: Callable[..., dict[str, Any]],
    msprime_tract_composition_gate_fn: Callable[..., dict[str, Any]],
    admixed_tract_length_gate_fn: Callable[..., dict[str, Any]],
    admixed_ancestry_variance_gate_fn: Callable[..., dict[str, Any]],
    is_admixed_involved_pair_fn: Callable[[tuple[str, str], dict[str, bool]], bool],
    logger: logging.Logger,
) -> MainGateEvaluation:
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
    stage_timings: dict[str, float] = {}

    t_gate = perf_counter()
    try:
        t_donor_pairs = perf_counter()
        donor_pairs_for_baseline = pairs
        if bool(config.qc.admixed_skip_donor_baseline_pairs) and admixed_qc_enabled:
            donor_pairs_for_baseline = source_source_pairs
        donor_pairwise = donor_baseline_fst_fn(
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
            is_admixed_pair = bool(pair is not None and is_admixed_involved_pair_fn(pair, population_msprime_is_admixed))
            if (
                is_admixed_pair
                and bool(config.qc.admixed_skip_donor_baseline_pairs)
                and admixed_qc_enabled
            ):
                gated = False
                gate_reason = "skipped_admixed_pair_policy"
            elif bool(config.qc.engine_aware_gating) and is_admixed_pair:
                gated = False
                gate_reason = "advisory_admixed_msprime_population"

            rel: float | None = None
            pair_pass: bool | None = None
            if gate_reason == "skipped_admixed_pair_policy":
                pair_pass = None
            elif base is None or np.isnan(base) or np.isnan(obs):
                gate_reason = "missing_baseline_or_observed"
                if gated:
                    pair_pass = False
                    n_fst_pairs_gated += 1
                    n_fst_pairs_failed += 1
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

        fst_gate_effective_pass = fst_gate_standard_pass
        try:
            manifest = load_manifest_fn(config)
            if admixed_qc_enabled:
                fst_stochastic_gate_activation = {
                    "mode": str(config.qc.fst_stochastic_gate_mode),
                    "enabled": False,
                    "reason": "disabled_admixed_qc_policy",
                }
            else:
                fst_stochastic_gate_activation = stochastic_gate_activation_fn(
                    config=config,
                    groups=groups,
                    pairs=pairs,
                    donors_by_population=manifest.donors_by_population,
                )
        except Exception as exc:
            fst_stochastic_gate_activation = {
                "mode": str(config.qc.fst_stochastic_gate_mode),
                "enabled": False,
                "reason": "activation_error",
                "error": str(exc),
            }
            gate_pass = False
            logger.warning("Stochastic donor-baseline FST gate activation failed: %s", exc)
        if isinstance(fst_stochastic_gate_activation, dict) and bool(fst_stochastic_gate_activation.get("enabled")):
            try:
                manifest = load_manifest_fn(config)
                fst_stochastic_gate = stochastic_gate_fn(
                    config=config,
                    chrom=chrom,
                    groups=groups,
                    pairs=pairs,
                    pairwise_fst=pairwise,
                    donors_by_population=manifest.donors_by_population,
                    donor_weights_by_population=manifest.donor_weights_by_population,
                    sampled_diploids_by_population=fst_stochastic_gate_activation.get("sampled_diploids_by_population", {}),
                    numba_threads=python_qc_numba_threads,
                )
            except Exception as exc:
                fst_stochastic_gate = {
                    "available": False,
                    "reason": "error",
                    "error": str(exc),
                }
                gate_pass = False
                logger.warning("Stochastic donor-baseline FST gate failed: %s", exc)
            if (
                isinstance(fst_stochastic_gate, dict)
                and bool(fst_stochastic_gate.get("available"))
                and isinstance(fst_stochastic_gate.get("gate_pass"), bool)
            ):
                fst_gate_effective_pass = bool(fst_stochastic_gate.get("gate_pass"))
                fst_gate_strategy_used = "stochastic_donor_baseline"
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
                if gated:
                    within_pass = False
                    gate_pass = False
            else:
                within_pass = bool(abs(value) <= config.qc.within_split_abs_max)
                if gated:
                    gate_reason = "passed_within_split_abs_max" if within_pass else "failed_within_split_abs_max"
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
        gate_pass = False
        fst_gate_effective_pass = False
        fst_pairwise_gate_status["__gate_error__"] = {
            "gated": True,
            "reason": "error",
            "error": str(exc),
            "pair_pass": False,
        }
        logger.warning("Donor-baseline gate failed: %s", exc)

    if bool(config.qc.engine_aware_gating) or admixed_qc_enabled:
        try:
            msprime_tract_composition_gate = msprime_tract_composition_gate_fn(
                config=config,
                chrom=int(chrom),
                n_shards=int(n_shards),
                pop_names=pop_names,
            )
            if (
                isinstance(msprime_tract_composition_gate, dict)
                and bool(msprime_tract_composition_gate.get("enabled"))
                and msprime_tract_composition_gate.get("gate_pass") is not True
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
            admixed_tract_length_gate = admixed_tract_length_gate_fn(
                config=config,
                chrom=int(chrom),
                n_shards=int(n_shards),
                pop_names=pop_names,
                diagnostics_dir=diagnostics_dir,
            )
            if (
                isinstance(admixed_tract_length_gate, dict)
                and bool(admixed_tract_length_gate.get("enabled"))
                and admixed_tract_length_gate.get("gate_pass") is not True
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
            admixed_ancestry_variance_gate = admixed_ancestry_variance_gate_fn(
                config=config,
                chrom=int(chrom),
                n_shards=int(n_shards),
                pop_names=pop_names,
            )
            if (
                isinstance(admixed_ancestry_variance_gate, dict)
                and bool(admixed_ancestry_variance_gate.get("enabled"))
                and admixed_ancestry_variance_gate.get("gate_pass") is not True
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

    return MainGateEvaluation(
        gate_pass=gate_pass,
        donor_pairwise=donor_pairwise,
        fst_gate_standard_pass=fst_gate_standard_pass,
        fst_gate_effective_pass=fst_gate_effective_pass,
        fst_gate_strategy_used=fst_gate_strategy_used,
        fst_stochastic_gate_activation=fst_stochastic_gate_activation,
        fst_stochastic_gate=fst_stochastic_gate,
        fst_pairwise_gate_status=fst_pairwise_gate_status,
        within_split_gate_status=within_split_gate_status,
        msprime_tract_composition_gate=msprime_tract_composition_gate,
        admixed_tract_length_gate=admixed_tract_length_gate,
        admixed_ancestry_variance_gate=admixed_ancestry_variance_gate,
        stage_timings=stage_timings,
    )
