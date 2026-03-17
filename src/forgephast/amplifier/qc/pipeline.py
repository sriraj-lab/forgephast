"""QC and population differentiation diagnostics."""

from __future__ import annotations

from itertools import combinations
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from hapstore import HapStore

from .metrics.ld_decay import (
    ld_decay_metrics_from_curves as local_ld_decay_metrics_from_curves,
    compute_ld_decay_qc_impl as _metric_compute_ld_decay_qc_impl,
)
from . import admixed_support
from . import donor_round_robin_support
from . import fst_gate
from . import gate_support
from . import matrix_backend
from . import metric_support
from . import report_support
from . import run_support
from .types import (
    DonorWithinPopRoundRobinReport,
    QcInputs,
    QcReport,
)
from . import round_robin_support as rr_support
from .round_robin_support import (
    round_robin_folds_from_fraction as _round_robin_folds_from_fraction,
    validate_donor_fold_fraction as _validate_donor_fold_fraction,
    validate_round_robin_n_folds as _validate_round_robin_n_folds,
)

from ..config import RunConfig
from ..donors import load_manifest
from ..donor_panels import load_donor_haplotypes_from_source
from .metrics.fst import (
    hudson_fst_from_counts as matrix_hudson_fst_from_counts,
)
from ..telemetry import RateLimitedProgress, telemetry_span
from ..utils import derived_seed, ensure_dir, write_json
from .registry import METRIC_REGISTRY

logger = logging.getLogger(__name__)


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
    return admixed_support.compute_msprime_tract_composition_gate(
        config=config,
        chrom=chrom,
        n_shards=n_shards,
        pop_names=pop_names,
    )


def _is_admixed_qc_enabled(config: RunConfig, population_msprime_is_admixed: dict[str, bool]) -> bool:
    return admixed_support.is_admixed_qc_enabled(config, population_msprime_is_admixed)


def _is_admixed_involved_pair(pair: tuple[str, str], population_msprime_is_admixed: dict[str, bool]) -> bool:
    return admixed_support.is_admixed_involved_pair(pair, population_msprime_is_admixed)


def _compute_admixed_tract_length_gate(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pop_names: list[str],
    diagnostics_dir: Path,
) -> dict[str, Any]:
    return admixed_support.compute_admixed_tract_length_gate(
        config=config,
        chrom=chrom,
        n_shards=n_shards,
        pop_names=pop_names,
        diagnostics_dir=diagnostics_dir,
        resolve_map_path_fn=_resolve_map_path_for_population,
    )


def _compute_admixed_ancestry_variance_gate(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    pop_names: list[str],
) -> dict[str, Any]:
    return admixed_support.compute_admixed_ancestry_variance_gate(
        config=config,
        chrom=chrom,
        n_shards=n_shards,
        pop_names=pop_names,
    )


def _apply_admixed_non_gating_to_metric(
    metric_payload: dict[str, Any],
    *,
    admixed_pop_names: set[str],
    reason: str,
) -> dict[str, Any]:
    return admixed_support.apply_admixed_non_gating_to_metric(
        metric_payload,
        admixed_pop_names=admixed_pop_names,
        reason=reason,
    )


def _compute_admixed_pca_cline_diagnostic(
    *,
    config: RunConfig,
    chrom: int,
    ext_data: Any | None,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    pop_names: list[str],
    diagnostics_dir: Path,
) -> dict[str, Any]:
    return admixed_support.compute_admixed_pca_cline_diagnostic(
        config=config,
        chrom=chrom,
        ext_data=ext_data,
        inputs=inputs,
        groups=groups,
        pop_names=pop_names,
        diagnostics_dir=diagnostics_dir,
        resolve_panel_source_fn=_resolve_donor_panel_source_for_population,
        load_sample_haplotypes_fn=_load_sample_haplotypes_from_panel,
        load_synthetic_population_haplotypes_fn=_load_synthetic_population_haplotypes,
        subsample_ids_fn=_subsample_ids,
    )


def _compute_admixed_king_diagnostic(
    *,
    config: RunConfig,
    king_payload: dict[str, Any] | None,
    pop_names: list[str],
) -> dict[str, Any]:
    return admixed_support.compute_admixed_king_diagnostic(
        config=config,
        king_payload=king_payload,
        pop_names=pop_names,
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


def _ld_decay_metrics_from_curves(
    *,
    donor_curve: pd.DataFrame,
    synthetic_curve: pd.DataFrame,
    min_pairs_per_bin: int,
    half_decay_r2_threshold: float,
) -> dict[str, float | int | str]:
    # Compatibility wrapper retained for existing tests/import sites.
    return local_ld_decay_metrics_from_curves(
        donor_curve=donor_curve,
        synthetic_curve=synthetic_curve,
        min_pairs_per_bin=min_pairs_per_bin,
        half_decay_r2_threshold=half_decay_r2_threshold,
    )


def _load_sample_haplotypes_from_panel(
    *,
    config: RunConfig,
    source_path: Path,
    chrom: int,
    sample_ids: list[str],
    source_kind: str = "vcf",
) -> tuple[np.ndarray, np.ndarray]:
    panel, _, _ = load_donor_haplotypes_from_source(
        source_path=source_path,
        source_kind=source_kind,
        chrom=chrom,
        donor_ids=sample_ids,
        cache_dir=config.donor_cache_root,
        force_refresh=False,
        ingest_workers=max(1, int(config.simulation.donor_io_workers)),
    )
    return panel.variants.pos.astype(np.int64, copy=False), panel.donor_haps


def _load_synthetic_population_haplotypes(
    *,
    config: RunConfig,
    inputs: QcInputs,
    chrom: int,
    pop: str,
    sample_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    def _load_hapstore_subset(store_path: Path, *, missing_context: str) -> tuple[np.ndarray, np.ndarray]:
        hs = HapStore(store_path, mode="r")
        all_ids = [str(x) for x in hs.sample_ids(chrom)]
        idx_by_id = {sid: idx for idx, sid in enumerate(all_ids)}
        missing = [sid for sid in sample_ids if sid not in idx_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} synthetic samples for {pop} missing in {missing_context}; first={missing[:5]}"
            )
        requested_idx = np.asarray([idx_by_id[sid] for sid in sample_ids], dtype=np.int64)
        mask = np.zeros(len(all_ids), dtype=bool)
        mask[requested_idx] = True
        haps = hs.haplotypes(chrom, sample_mask=mask)

        selected_idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        pos = np.searchsorted(selected_idx, requested_idx)
        if pos.size != requested_idx.size or not np.array_equal(selected_idx[pos], requested_idx):
            raise RuntimeError(f"Unable to reorder {missing_context} haplotypes to requested sample order")
        hap_rows = np.empty(pos.size * 2, dtype=np.int64)
        hap_rows[0::2] = 2 * pos
        hap_rows[1::2] = (2 * pos) + 1
        return hs.variant_positions(chrom), haps[hap_rows, :]

    if inputs.combined_hapstore is not None:
        return _load_hapstore_subset(inputs.combined_hapstore, missing_context="combined hapstore")

    if pop in inputs.population_hapstores:
        return _load_hapstore_subset(inputs.population_hapstores[pop], missing_context="population hapstore")

    raise ValueError(
        f"Synthetic population {pop!r} is not available in HapStore QC inputs; "
        "run simulation/merge with outputs.write_hapstore=true"
    )


def _compute_ld_decay_qc_python(
    *,
    config: RunConfig,
    chrom: int,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
) -> tuple[dict[str, Any], Path]:
    return _metric_compute_ld_decay_qc_impl(
        config=config,
        chrom=int(chrom),
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=numba_threads,
        resolve_donor_panel_source_for_population_fn=lambda c, pop: _resolve_donor_panel_source_for_population(
            config,
            chrom=int(c),
            pop=str(pop),
        ),
        resolve_map_path_for_population_fn=lambda c, pop: _resolve_map_path_for_population(
            config,
            chrom=int(c),
            pop=str(pop),
        ),
        load_sample_haplotypes_from_panel_fn=lambda *, source_path, chrom, sample_ids, source_kind: (
            _load_sample_haplotypes_from_panel(
                config=config,
                source_path=Path(source_path),
                chrom=int(chrom),
                sample_ids=list(sample_ids),
                source_kind=str(source_kind),
            )
        ),
        load_synthetic_population_haplotypes_fn=lambda *, inputs, chrom, pop, sample_ids: (
            _load_synthetic_population_haplotypes(
                config=config,
                inputs=inputs,
                chrom=int(chrom),
                pop=str(pop),
                sample_ids=list(sample_ids),
            )
        ),
    )


def _compute_ld_decay_qc(
    *,
    config: RunConfig,
    chrom: int,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
) -> tuple[dict[str, Any], Path]:
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


def _resolve_synthetic_outputs(config: RunConfig, chrom: int, n_shards: int) -> QcInputs:
    merged_manifest = config.run_root / "manifests" / f"merged_samples_chr{chrom}.tsv"
    merged_hapstore = config.run_root / "merged" / f"chr{chrom}" / f"combined.chr{chrom}.merged.hapstore.zarr"
    if merged_hapstore.exists() and merged_manifest.exists():
        return QcInputs(
            mode="combined_merged_hapstore",
            sample_manifest=merged_manifest,
            combined_vcf=None,
            combined_hapstore=merged_hapstore,
            population_hapstores={},
        )

    pop_dir = config.run_root / "merged" / f"chr{chrom}" / "populations"
    population_hapstores: dict[str, Path] = {
        pop.name: pop_dir / f"{pop.name}.chr{chrom}.merged.hapstore.zarr" for pop in config.populations
    }
    if merged_manifest.exists() and population_hapstores and all(path.exists() for path in population_hapstores.values()):
        return QcInputs(
            mode="population_merged_hapstore",
            sample_manifest=merged_manifest,
            combined_vcf=None,
            combined_hapstore=None,
            population_hapstores=population_hapstores,
        )

    if n_shards != 1:
        raise FileNotFoundError(
            "Merged HapStore outputs not found for multi-shard QC; run merge first with "
            "outputs.write_hapstore=true."
        )

    single_vcf = config.run_root / "synthetic" / f"chr{chrom}" / "shard_0_of_1" / f"combined.chr{chrom}.vcf.gz"
    single_hapstore = (
        config.run_root / "synthetic" / f"chr{chrom}" / "shard_0_of_1" / f"combined.chr{chrom}.hapstore.zarr"
    )
    single_manifest = config.run_root / "manifests" / f"synthetic_samples_chr{chrom}_shard_0_of_1.tsv"
    if single_hapstore.exists() and single_manifest.exists():
        return QcInputs(
            mode="single_shard_combined_hapstore",
            sample_manifest=single_manifest,
            combined_vcf=single_vcf if single_vcf.exists() else None,
            combined_hapstore=single_hapstore,
            population_hapstores={},
        )

    raise FileNotFoundError(
        "Expected HapStore-backed synthetic outputs for QC were not found for "
        f"chr{chrom}: {single_hapstore} / {single_manifest}. "
        "Run simulation with outputs.write_hapstore=true."
    )


def _validate_population_variant_axes(stores: dict[str, HapStore], chrom: int) -> int:
    return matrix_backend.validate_population_variant_axes_impl(stores, chrom=int(chrom))


def _run_qc_matrix_hapstore(
    config: RunConfig,
    chrom: int,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    within_checks: list[str],
    numba_threads: int | None,
) -> tuple[dict[str, float], dict[str, float]]:
    return matrix_backend.run_qc_matrix_hapstore_impl(
        config=config,
        chrom=int(chrom),
        inputs=inputs,
        groups=groups,
        pairs=pairs,
        within_checks=within_checks,
        numba_threads=numba_threads,
        logger=logger,
    )


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
        resolve_pairwise_sets_fn=_resolve_pairwise_sets,
        resolve_donor_panel_source_for_population_fn=_resolve_donor_panel_source_for_population,
        load_sample_haplotypes_from_panel_fn=_load_sample_haplotypes_from_panel,
        resolve_python_qc_numba_threads_fn=_resolve_python_qc_numba_threads,
    )


def _pairwise_fst_from_counts_map(
    *,
    pairs: list[tuple[str, str]],
    counts_by_population: dict[str, tuple[np.ndarray, np.ndarray]],
    numba_threads: int | None,
) -> dict[str, float]:
    return fst_gate.pairwise_fst_from_counts_map_impl(
        pairs=pairs,
        counts_by_population=counts_by_population,
        numba_threads=numba_threads,
    )


def _resolve_fst_stochastic_gate_activation(
    *,
    config: RunConfig,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    donors_by_population: dict[str, list[str]],
) -> dict[str, Any]:
    return fst_gate.resolve_fst_stochastic_gate_activation_impl(
        config=config,
        groups=groups,
        pairs=pairs,
        donors_by_population=donors_by_population,
    )


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
    return fst_gate.compute_stochastic_donor_fst_gate_impl(
        config=config,
        chrom=int(chrom),
        groups=groups,
        pairs=pairs,
        pairwise_fst=pairwise_fst,
        donors_by_population=donors_by_population,
        sampled_diploids_by_population=sampled_diploids_by_population,
        numba_threads=numba_threads,
        resolve_donor_panel_source_for_population_fn=_resolve_donor_panel_source_for_population,
        load_sample_haplotypes_from_panel_fn=_load_sample_haplotypes_from_panel,
    )


def _round_robin_for_population_chrom(
    *,
    config: RunConfig,
    pop: str,
    chrom: int,
    sample_ids: list[str],
    n_folds: int,
    seed: int,
    numba_threads: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    source_kind, source_path = _resolve_donor_panel_source_for_population(
        config,
        chrom=chrom,
        pop=pop,
    )
    return rr_support.round_robin_for_population_chrom(
        pop=pop,
        chrom=chrom,
        sample_ids=sample_ids,
        n_folds=n_folds,
        seed=seed,
        numba_threads=numba_threads,
        even_folds_fn=_even_folds,
        load_sample_haplotypes_fn=lambda fold_ids: _load_sample_haplotypes_from_panel(
            config=config,
            source_path=source_path,
            chrom=chrom,
            sample_ids=list(fold_ids),
            source_kind=source_kind,
        ),
        matrix_hudson_fst_from_counts_fn=matrix_hudson_fst_from_counts,
    )


def run_donor_within_pop_round_robin(
    config: RunConfig,
    chrom: int,
    fold_fraction: float = 0.2,
    seed: int = 42,
    workers: int = 1,
    populations: list[str] | None = None,
) -> DonorWithinPopRoundRobinReport:
    return donor_round_robin_support.run_donor_within_pop_round_robin_impl(
        config=config,
        chrom=int(chrom),
        fold_fraction=float(fold_fraction),
        seed=int(seed),
        workers=int(workers),
        populations=populations,
        round_robin_folds_from_fraction_fn=_round_robin_folds_from_fraction,
        load_manifest_fn=load_manifest,
        resolve_requested_populations_fn=_resolve_requested_populations,
        ensure_dir_fn=ensure_dir,
        round_robin_for_population_fn=lambda **kwargs: _round_robin_for_population_chrom(
            config=config,
            **kwargs,
        ),
        write_json_fn=write_json,
    )


def run_qc(config: RunConfig, chrom: int, n_shards: int = 1) -> QcReport:
    return run_support.run_qc_impl(
        config=config,
        chrom=int(chrom),
        n_shards=int(n_shards),
        logger=logger,
        t0=perf_counter(),
        stage_timings={},
        resolve_synthetic_outputs_fn=_resolve_synthetic_outputs,
        sample_groups_from_manifest_fn=_sample_groups_from_manifest,
        resolve_pairwise_sets_fn=_resolve_pairwise_sets,
        resolve_within_split_checks_fn=_resolve_within_split_checks,
        resolve_python_qc_numba_threads_fn=_resolve_python_qc_numba_threads,
        is_admixed_qc_enabled_fn=_is_admixed_qc_enabled,
        is_admixed_involved_pair_fn=_is_admixed_involved_pair,
        run_qc_matrix_hapstore_fn=_run_qc_matrix_hapstore,
        evaluate_main_qc_gates_fn=gate_support.evaluate_main_qc_gates,
        prepare_extended_metric_context_fn=metric_support.prepare_extended_metric_context,
        run_registered_metrics_fn=metric_support.run_registered_metrics,
        compute_admixed_optional_diagnostics_fn=report_support.compute_admixed_optional_diagnostics,
        build_population_status_fn=report_support.build_population_status,
        build_admixed_qc_summary_fn=report_support.build_admixed_qc_summary,
        write_qc_diagnostics_and_finalize_fn=report_support.write_qc_diagnostics_and_finalize,
        load_manifest_fn=load_manifest,
        apply_admixed_non_gating_fn=_apply_admixed_non_gating_to_metric,
        resolve_donor_panel_source_for_population_fn=lambda c, pop: _resolve_donor_panel_source_for_population(
            config,
            chrom=int(c),
            pop=str(pop),
        ),
        load_sample_haplotypes_from_panel_fn=_load_sample_haplotypes_from_panel,
        load_synthetic_population_haplotypes_fn=_load_synthetic_population_haplotypes,
        donor_baseline_fst_fn=donor_baseline_fst,
        resolve_fst_stochastic_gate_activation_fn=_resolve_fst_stochastic_gate_activation,
        compute_stochastic_donor_fst_gate_fn=_compute_stochastic_donor_fst_gate,
        compute_msprime_tract_composition_gate_fn=_compute_msprime_tract_composition_gate,
        compute_admixed_tract_length_gate_fn=_compute_admixed_tract_length_gate,
        compute_admixed_ancestry_variance_gate_fn=_compute_admixed_ancestry_variance_gate,
        compute_admixed_pca_cline_diagnostic_fn=_compute_admixed_pca_cline_diagnostic,
        compute_admixed_king_diagnostic_fn=_compute_admixed_king_diagnostic,
        telemetry_span_fn=telemetry_span,
        ensure_dir_fn=ensure_dir,
        subsample_ids_fn=_subsample_ids,
        metric_registry=METRIC_REGISTRY,
    )
