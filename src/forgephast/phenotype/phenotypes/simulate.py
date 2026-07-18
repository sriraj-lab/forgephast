"""Continuous polygenic phenotype simulation orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids

from ..architectures.base import ArchitectureInput, create_plugin
from ..causal.hybrid_sampler import sample_membership_for_replicate
from ..causal.selection import build_bucketed_pool
from ..config import GlobalPCConfig, RunConfig
from ..covariates.model import build_covariate_effects
from ..data.genotypes import (
    GenotypeAdapter,
    build_genotype_adapter,
)
from ..effects.betas import (
    build_variant_scale_by_maf,
    resolve_effect_geometry,
    sample_population_specific_betas,
)
from .background import (
    _build_background_component,
    _exclude_causal_grm_variants,
    _read_background_variant_ids,
    _sample_background_from_grm_factor,
    _standardize_grm_genotypes,
)
from .binary import (
    _admix_alpha_weights_from_local_ancestry,
    _binary_phenotype_from_liability,
    _calibrate_logistic_intercept,
)
from .components import (
    _component_variance_diagnostics,
    _resolve_cohort_component_targets,
    _scale_component_to_variance,
    _scale_effects_detail_betas,
    _scale_factor_to_variance,
)
from .cohort_inputs import CohortInputCaches, prepare_cohort_inputs

logger = logging.getLogger(__name__)


def _stable_string_seed(value: str) -> int:
    """Deterministic seed component from string."""
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value))


def _sample_id_prefix(sample_id: str) -> str:
    token = str(sample_id).strip()
    prefix, sep, _ = token.rpartition("_syn")
    return prefix if sep else token


def _required_source_populations(
    *,
    sample_populations: List[str],
    la_enabled: bool,
    ancestry_order: List[str] | None,
) -> List[str]:
    """Return source populations whose fixed-effect scales must be known."""
    if la_enabled:
        if ancestry_order is None:
            raise ValueError("Local ancestry mode requires ancestry_order to resolve source populations")
        return [str(pop) for pop in ancestry_order]
    return sorted({str(pop) for pop in sample_populations if str(pop).upper() != "ADMIX"})


def _sample_frame(sample_ids: List[str]) -> pd.DataFrame:
    """Minimal IID/FID frame for downstream interfaces that require it."""
    return pd.DataFrame({"FID": sample_ids, "IID": sample_ids}, copy=False)


def _parse_chr_pos_variant_id(variant_id: str) -> tuple[int, int]:
    value = str(variant_id)
    if not value.startswith("chr") or ":" not in value:
        raise ValueError(f"Expected chrN:POS variant_id, got {variant_id!r}")
    chrom_str, pos_str = value[3:].split(":", 1)
    return int(chrom_str), int(pos_str)


def _build_global_pcs(
    mode_cfg: GlobalPCConfig,
    adapter: GenotypeAdapter,
    sample_frame: pd.DataFrame,
    sample_ids: List[str],
    sample_populations: List[str],
    out_dir: Path,
    chromosomes: List[int] | None = None,
) -> pd.DataFrame:
    """Construct global PC covariate table for one record."""
    if mode_cfg.mode == "none":
        return sample_frame.copy()

    mode = mode_cfg.canonical_mode()
    if mode in {"plink2_pca", "svd_pca"}:
        return adapter.compute_global_pcs(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            n_components=mode_cfg.n_components,
            out_dir=out_dir,
            mode_cfg=mode_cfg,
            chromosomes=chromosomes,
        )

    pre = pd.read_csv(mode_cfg.precomputed_file, sep="\t")
    aligned = align_by_ids(pre, mode_cfg.id_column, sample_ids, "global PCs alignment")
    aligned = aligned.rename(columns={mode_cfg.id_column: "IID"})
    if "FID" not in aligned.columns:
        aligned.insert(0, "FID", aligned["IID"])
    keep_cols = ["FID", "IID"] + [c for c in aligned.columns if c.startswith("PC")]
    return aligned[keep_cols].copy()


def run_simulation(
    run_cfg: RunConfig,
    run_dir: Path,
    adapter: GenotypeAdapter | None = None,
    runtime_cache: dict[str, Any] | None = None,
    timing_rows: List[Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """Run simulation stage and return in-memory records for writing."""
    if timing_rows is None:
        timing_rows = []

    def _timed(stage: str, started: float, **extras: object) -> None:
        elapsed = float(time.perf_counter() - started)
        timing_rows.append(
            {
                "stage": stage,
                "elapsed_seconds": elapsed,
                "run_name": str(run_cfg.name),
                **extras,
            }
        )
        logger.info(
            "SIMULATE_TIMING run=%s stage=%s elapsed_seconds=%.3f",
            str(run_cfg.name),
            str(stage),
            float(elapsed),
        )

    if adapter is None:
        t0 = time.perf_counter()
        adapter = build_genotype_adapter(
            hapstore_cfg=run_cfg.genotype_input.hapstore,
        )
        adapter.validate_files()
        _timed("simulate.adapter_init", t0)

    t_pool = time.perf_counter()
    bucketed_pool = build_bucketed_pool(run_cfg.causal_selection)
    _timed("simulate.build_bucketed_pool", t_pool)
    records: List[Dict[str, object]] = []
    configured_background_variant_ids: List[str] = []
    if run_cfg.background_genetic.mode == "compute_grm":
        configured_background_variant_ids = _read_background_variant_ids(
            str(run_cfg.background_genetic.variant_ids_file)
        )

    variant_col = run_cfg.causal_selection.variant_id_column
    default_population = run_cfg.effect_model.population_order[0]
    effect_scale_maf_col = run_cfg.causal_selection.effect_scale_maf_column
    if effect_scale_maf_col is None:
        anchor_population = run_cfg.causal_selection.anchor_population
        if anchor_population is None:
            anchor_population = next(iter(run_cfg.causal_selection.maf_columns_by_population.keys()))
        effect_scale_maf_col = run_cfg.causal_selection.maf_columns_by_population[anchor_population]
    if runtime_cache is None:
        kinship_cache: Dict[str, np.ndarray] = {}
        effect_geometry_cache: Dict[str, Dict[str, object]] = {}
    else:
        kinship_cache = runtime_cache.setdefault("simulate_kinship_cache", {})
        effect_geometry_cache = runtime_cache.setdefault("simulate_effect_geometry_cache", {})
    cohort_input_caches = CohortInputCaches.from_runtime_cache(runtime_cache)

    for replicate in range(1, run_cfg.phenotype.n_replicates + 1):
        rep_t0 = time.perf_counter()
        logger.info("[simulate] replicate %d/%d", replicate, run_cfg.phenotype.n_replicates)

        t_membership = time.perf_counter()
        membership = sample_membership_for_replicate(
            replicate_idx=replicate,
            full_pool=bucketed_pool.full_pool,
            fixed_set=bucketed_pool.fixed_set,
            config=run_cfg.causal_selection,
        )
        _timed("simulate.sample_membership", t_membership, replicate=int(replicate))
        variant_ids = membership[variant_col].astype(str).tolist()
        parsed = [_parse_chr_pos_variant_id(vid) for vid in variant_ids]
        variant_chrom = np.asarray([c for c, _ in parsed], dtype=np.int32)
        variant_pos_bp = np.asarray([p for _, p in parsed], dtype=np.int64)
        variant_indices_by_chrom = {
            int(chrom): np.where(variant_chrom == int(chrom))[0].astype(np.int64)
            for chrom in run_cfg.chromosomes
        }
        _timed(
            "simulate.parse_variants",
            t_membership,
            replicate=int(replicate),
            n_variants=int(len(variant_ids)),
        )
        variant_scale_by_id = build_variant_scale_by_maf(
            membership_df=membership,
            variant_id_column=variant_col,
            maf_column=effect_scale_maf_col,
            effect_model=run_cfg.effect_model,
        )
        _timed("simulate.variant_scale_by_maf", t_membership, replicate=int(replicate))

        for cohort in run_cfg.cohorts:
            cohort_t0 = time.perf_counter()
            prepared = prepare_cohort_inputs(
                run_cfg=run_cfg,
                cohort=cohort,
                replicate=replicate,
                adapter=adapter,
                run_dir=run_dir,
                default_population=default_population,
                variant_ids=variant_ids,
                variant_indices_by_chrom=variant_indices_by_chrom,
                variant_pos_bp=variant_pos_bp,
                configured_background_variant_ids=configured_background_variant_ids,
                caches=cohort_input_caches,
                timed=_timed,
            )
            sample_ids = prepared.sample_ids
            sample_populations = prepared.sample_populations
            sample_frame = prepared.sample_frame
            genotype_matrix = prepared.genotype_matrix
            truth_admix_indices = prepared.truth_admix_indices
            truth_admix_sample_ids = prepared.truth_admix_sample_ids
            admix_haplotypes = prepared.admix_haplotypes
            admix_local_ancestry_codes = prepared.admix_local_ancestry_codes
            donor_fallback_indices = prepared.donor_fallback_indices
            donor_fallback_sample_ids = prepared.donor_fallback_sample_ids
            donor_fallback_sample_pops = prepared.donor_fallback_sample_pops
            la_genotypes_by_population = prepared.la_genotypes_by_population
            dosage_effect_allele = prepared.dosage_effect_allele
            dosage_other_allele = prepared.dosage_other_allele
            background_grm_factor = prepared.background_grm_factor
            background_grm_variant_count = prepared.background_grm_variant_count
            background_grm_excluded_causal_count = (
                prepared.background_grm_excluded_causal_count
            )
            la_enabled = prepared.la_enabled
            cohort_tmp = prepared.cohort_tmp

            t_cov = time.perf_counter()
            cov_out = build_covariate_effects(
                cfg=run_cfg.covariates,
                sample_ids=sample_ids,
                replicate_seed=run_cfg.covariates.random_seed + replicate,
                runtime_cache=runtime_cache,
            )
            _timed(
                "simulate.covariates",
                t_cov,
                replicate=int(replicate),
                cohort=str(cohort.name),
                n_samples=int(len(sample_ids)),
            )

            # In plink2_pca mode, PCs are refreshed from exported PGEN in
            # the GWAS prep stage; avoid expensive in-simulation SVD here.
            if run_cfg.gwas.global_pcs.canonical_mode() == "plink2_pca":
                pcs_df = sample_frame.copy()
            else:
                pcs_df = _build_global_pcs(
                    mode_cfg=run_cfg.gwas.global_pcs,
                    adapter=adapter,
                    sample_frame=sample_frame,
                    sample_ids=sample_ids,
                    sample_populations=sample_populations,
                    out_dir=run_dir / "tmp" / "global_pcs" / f"rep{replicate}" / cohort.name,
                    chromosomes=run_cfg.chromosomes,
                )

            for architecture in run_cfg.architectures:
                plugin = create_plugin(architecture.plugin)
                if architecture.name in effect_geometry_cache:
                    effect_geometry = effect_geometry_cache[architecture.name]
                else:
                    effect_geometry = resolve_effect_geometry(run_cfg.effect_model)
                    effect_geometry_cache[architecture.name] = effect_geometry

                beta_seed = (
                    run_cfg.effect_model.random_seed
                    + 1000 * replicate
                    + _stable_string_seed(architecture.name)
                )
                t_beta = time.perf_counter()
                beta_df = sample_population_specific_betas(
                    variant_ids=variant_ids,
                    architecture=architecture,
                    effect_model=run_cfg.effect_model,
                    replicate_seed=beta_seed,
                    variant_scale_by_id=variant_scale_by_id,
                    resolved_geometry=effect_geometry,
                )
                _timed(
                    "simulate.sample_betas",
                    t_beta,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    architecture=str(architecture.name),
                    n_variants=int(len(variant_ids)),
                )
                targets = _resolve_cohort_component_targets(
                    run_cfg=run_cfg,
                    architecture_params=architecture.params,
                )
                required_pops = _required_source_populations(
                    sample_populations=sample_populations,
                    la_enabled=la_enabled,
                    ancestry_order=list(run_cfg.local_ancestry.ancestry_order) if la_enabled else None,
                )

                def _compute_fixed_component_with_betas(
                    beta_df_for_compute: pd.DataFrame,
                ) -> tuple[np.ndarray, pd.DataFrame]:
                    fallback_rules_by_prefix = {
                        str(rule.sample_id_prefix): {
                            str(pop): float(weight)
                            for pop, weight in rule.beta_weights.items()
                        }
                        for rule in run_cfg.local_ancestry.donor_mixture_fallbacks
                    }
                    if la_enabled:
                        # Local ancestry computation:
                        # - homogeneous EUR/AFR: unphased genotypes are sufficient (ancestry constant)
                        # - admixed samples: need phased haplotypes + per-haplotype ancestry codes
                        if not hasattr(adapter, "export_genotypes_u8"):
                            raise ValueError(
                                "Local ancestry mode requires HapStoreAdapter with export_genotypes_u8()"
                            )

                        beta_pivot = beta_df_for_compute.pivot(
                            index="variant_id",
                            columns="population",
                            values="beta",
                        )
                        genetic_component = np.zeros(len(sample_ids), dtype=np.float64)
                        pops_arr = np.asarray(sample_populations, dtype=object)

                        for pop_label in required_pops:
                            row_idx = np.where(
                                np.char.upper(pops_arr.astype(str)) == str(pop_label).upper()
                            )[0].astype(np.int64)
                            if int(row_idx.size) == 0:
                                continue
                            if pop_label in la_genotypes_by_population:
                                geno_u8 = la_genotypes_by_population[pop_label]
                            else:
                                sub_ids = [sample_ids[int(i)] for i in row_idx.tolist()]
                                sub_pops = [sample_populations[int(i)] for i in row_idx.tolist()]
                                t_geno = time.perf_counter()
                                geno_u8 = adapter.export_genotypes_u8(
                                    sample_ids=sub_ids,
                                    sample_populations=sub_pops,
                                    variant_ids=variant_ids,
                                    out_dir=cohort_tmp,
                                    chromosomes=run_cfg.chromosomes,
                                )
                                _timed(
                                    "simulate.export_genotypes_u8",
                                    t_geno,
                                    replicate=int(replicate),
                                    cohort=str(cohort.name),
                                    population=str(pop_label),
                                    n_samples=int(len(sub_ids)),
                                    n_variants=int(len(variant_ids)),
                                )
                                la_genotypes_by_population[pop_label] = geno_u8
                            if pop_label not in beta_pivot.columns:
                                raise ValueError(f"Missing beta population column for homogeneous pop={pop_label}")
                            betas = beta_pivot.loc[variant_ids, pop_label].to_numpy(dtype=np.float32)
                            genetic_component[row_idx] = geno_u8.astype(np.float32, copy=False) @ betas

                        if truth_admix_indices:
                            if (
                                admix_haplotypes is None
                                or admix_local_ancestry_codes is None
                                or truth_admix_sample_ids is None
                            ):
                                raise ValueError("Missing ADMIX haplotypes/codes for local ancestry simulation")
                            admix_idx_arr = np.asarray(truth_admix_indices, dtype=np.int64)
                            la_input = ArchitectureInput(
                                sample_ids=list(truth_admix_sample_ids),
                                sample_populations=["ADMIX"] * len(truth_admix_sample_ids),
                                dosage_df=_sample_frame(list(truth_admix_sample_ids)),
                                haplotypes=admix_haplotypes,
                                local_ancestry_codes=admix_local_ancestry_codes,
                                ancestry_order=list(run_cfg.local_ancestry.ancestry_order),
                                la_chunk_n_samples=int(run_cfg.local_ancestry.chunk_n_samples),
                                la_chunk_n_variants=int(run_cfg.local_ancestry.chunk_n_variants),
                                use_numba=bool(run_cfg.local_ancestry.use_numba),
                                membership_df=membership,
                                beta_df=beta_df_for_compute,
                                variant_id_column=variant_col,
                            )
                            genetic_component[admix_idx_arr] = plugin._compute_local_ancestry_genetic_component(la_input)  # type: ignore[attr-defined]

                        if donor_fallback_indices:
                            if donor_fallback_sample_ids is None or donor_fallback_sample_pops is None:
                                raise ValueError("Missing donor-mixture fallback sample metadata")
                            geno_key = "__admix_donor_mixture_fallback__"
                            if geno_key in la_genotypes_by_population:
                                donor_geno_u8 = la_genotypes_by_population[geno_key]
                            else:
                                t_geno = time.perf_counter()
                                donor_geno_u8 = adapter.export_genotypes_u8(
                                    sample_ids=list(donor_fallback_sample_ids),
                                    sample_populations=list(donor_fallback_sample_pops),
                                    variant_ids=variant_ids,
                                    out_dir=cohort_tmp,
                                    chromosomes=run_cfg.chromosomes,
                                )
                                _timed(
                                    "simulate.export_genotypes_u8",
                                    t_geno,
                                    replicate=int(replicate),
                                    cohort=str(cohort.name),
                                    population="ADMIX_DONOR_FALLBACK",
                                    n_samples=int(len(donor_fallback_sample_ids)),
                                    n_variants=int(len(variant_ids)),
                                )
                                la_genotypes_by_population[geno_key] = donor_geno_u8
                            donor_idx_arr = np.asarray(donor_fallback_indices, dtype=np.int64)
                            donor_prefixes = np.asarray(
                                [_sample_id_prefix(sample_id) for sample_id in donor_fallback_sample_ids],
                                dtype=object,
                            )
                            for sample_prefix, beta_weights in fallback_rules_by_prefix.items():
                                local_rows = np.where(donor_prefixes == sample_prefix)[0].astype(np.int64)
                                if int(local_rows.size) == 0:
                                    continue
                                betas = np.zeros(len(variant_ids), dtype=np.float32)
                                for population, weight in beta_weights.items():
                                    if population not in beta_pivot.columns:
                                        raise ValueError(
                                            f"Missing beta population column for donor-mixture fallback pop={population}"
                                        )
                                    betas += float(weight) * beta_pivot.loc[variant_ids, population].to_numpy(dtype=np.float32)
                                genetic_component[donor_idx_arr[local_rows]] = (
                                    donor_geno_u8[local_rows, :].astype(np.float32, copy=False) @ betas
                                )

                        effects_detail = membership[
                            [variant_col, "maf_bucket", "membership_type", "replicate"]
                        ].merge(
                            beta_df_for_compute,
                            left_on=variant_col,
                            right_on="variant_id",
                            how="left",
                        )
                        if variant_col != "variant_id":
                            effects_detail = effects_detail.drop(columns=["variant_id"])
                            effects_detail = effects_detail.rename(columns={variant_col: "variant_id"})
                        return genetic_component, effects_detail

                    return plugin.compute_genetic_component(
                        ArchitectureInput(
                            sample_ids=sample_ids,
                            sample_populations=sample_populations,
                            dosage_df=sample_frame,
                            genotype_matrix=genotype_matrix,
                            ancestry_order=None,
                            la_chunk_n_samples=None,
                            la_chunk_n_variants=None,
                            use_numba=bool(run_cfg.local_ancestry.use_numba),
                            membership_df=membership,
                            beta_df=beta_df_for_compute,
                            variant_id_column=variant_col,
                        )
                    )

                t_fixed = time.perf_counter()
                # Compute the fixed component from jittered per-population betas.
                # No per-population β rescaling is applied — per-population variance
                # ratios are a free output of the MAF structure and jitter model.
                genetic_fixed_raw, effects_detail = _compute_fixed_component_with_betas(beta_df)

                # Anchor the pooled genetic fixed-effect variance to the
                # architecture-level h² target.  A single scalar is applied to
                # the entire vector, preserving per-population variance ratios
                # while giving realistic absolute h².
                _target_h2_fixed = float(targets["genetic_fixed"])
                genetic_fixed_beta_scale_factor = _scale_factor_to_variance(
                    genetic_fixed_raw, _target_h2_fixed
                )
                genetic_fixed = genetic_fixed_raw * genetic_fixed_beta_scale_factor
                effects_detail = _scale_effects_detail_betas(
                    effects_detail, genetic_fixed_beta_scale_factor
                )

                _timed(
                    "simulate.compute_genetic_fixed",
                    t_fixed,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    architecture=str(architecture.name),
                    n_samples=int(len(sample_ids)),
                    n_variants=int(len(variant_ids)),
                )

                # Population-stratified variance is diagnostic only; the target
                # and scaling operation are defined over the full cohort.
                populations_for_realized = np.asarray(sample_populations, dtype=object)
                realized_fixed_variance_by_population: Dict[str, float] = {}
                for pop in sorted({str(pop) for pop in sample_populations}):
                    row_idx = np.where(populations_for_realized == pop)[0]
                    if row_idx.size == 0:
                        continue
                    realized_fixed_variance_by_population[pop] = float(
                        np.var(genetic_fixed[row_idx])
                    )
                bg_seed = (
                    run_cfg.background_genetic.seed
                    + 2000 * replicate
                    + _stable_string_seed(architecture.name)
                    + _stable_string_seed(cohort.name)
                )
                t_bg = time.perf_counter()
                genetic_bg_raw = _build_background_component(
                    run_cfg=run_cfg,
                    sample_ids=sample_ids,
                    grm_factor=background_grm_factor,
                    cache_key=f"{cohort.name}",
                    replicate_seed=bg_seed,
                    kinship_cache=kinship_cache,
                )
                _timed(
                    "simulate.compute_genetic_bg",
                    t_bg,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    architecture=str(architecture.name),
                    n_samples=int(len(sample_ids)),
                    n_variants=int(len(variant_ids)),
                )

                t_scale = time.perf_counter()
                rng = np.random.default_rng(run_cfg.phenotype.random_seed + replicate + _stable_string_seed(architecture.name))
                residual_raw = rng.normal(0.0, 1.0, size=len(sample_ids))
                genetic_bg = _scale_component_to_variance(
                    genetic_bg_raw, targets["genetic_bg"]
                )
                covariate = _scale_component_to_variance(
                    cov_out.contribution, targets["covariate"]
                )
                residual = _scale_component_to_variance(
                    residual_raw, targets["residual"]
                )

                genetic = genetic_fixed + genetic_bg
                liability = genetic + covariate + residual
                component_diagnostics = _component_variance_diagnostics(
                    genetic_fixed=genetic_fixed,
                    genetic_bg=genetic_bg,
                    covariate=covariate,
                    residual=residual,
                    liability=liability,
                )
                if component_diagnostics["covariance_warning"]:
                    logger.warning(
                        "PHENOTYPE_COMPONENT_COVARIANCE_WARNING "
                        "replicate=%d cohort=%s architecture=%s "
                        "covariance_contribution=%.6g relative_to_component_variance_sum=%.6g "
                        "realized fractions may differ materially from configured variance targets",
                        int(replicate),
                        str(cohort.name),
                        str(architecture.name),
                        float(component_diagnostics["covariance_contribution"]),
                        float(component_diagnostics["covariance_relative_contribution"]),
                    )
                phenotype: np.ndarray = liability
                binary_probability: np.ndarray | None = None
                binary_summary: Dict[str, Any] = {}
                trait_type = str(run_cfg.phenotype.trait_type)
                if trait_type == "binary":
                    admix_alpha_weights_by_row: Dict[int, Dict[str, float]] = {}
                    if (
                        truth_admix_indices
                        and (run_cfg.phenotype.binary.prevalence_by_population or run_cfg.phenotype.binary.alpha_by_population)
                        and admix_local_ancestry_codes is not None
                    ):
                        admix_alpha_weights_by_row = _admix_alpha_weights_from_local_ancestry(
                            admix_row_indices=[int(i) for i in truth_admix_indices],
                            admix_local_ancestry_codes=admix_local_ancestry_codes,
                            ancestry_order=list(run_cfg.local_ancestry.ancestry_order),
                            effects_detail=effects_detail,
                            variant_ids=variant_ids,
                            liability=liability,
                            liability_logit_scale=float(run_cfg.phenotype.binary.liability_logit_scale),
                        )
                    binary_rng = np.random.default_rng(
                        int(run_cfg.phenotype.binary.random_seed)
                        + 1000 * int(replicate)
                        + _stable_string_seed(architecture.name)
                        + _stable_string_seed(cohort.name)
                    )
                    phenotype, binary_probability, binary_alpha, binary_summary = _binary_phenotype_from_liability(
                        liability=liability,
                        sample_populations=sample_populations,
                        prevalence=float(run_cfg.phenotype.binary.prevalence),
                        liability_logit_scale=float(run_cfg.phenotype.binary.liability_logit_scale),
                        prevalence_by_population=dict(
                            run_cfg.phenotype.binary.prevalence_by_population
                        ),
                        alpha_by_population=dict(run_cfg.phenotype.binary.alpha_by_population),
                        admix_alpha_weights_by_row=admix_alpha_weights_by_row,
                        rng=binary_rng,
                    )
                _timed(
                    "simulate.scale_and_combine",
                    t_scale,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    architecture=str(architecture.name),
                    n_samples=int(len(sample_ids)),
                )

                phenotype_payload: Dict[str, Any] = {
                    "FID": sample_ids,
                    "IID": sample_ids,
                    run_cfg.phenotype.phenotype_column: phenotype,
                    "GENETIC_FIXED_COMPONENT": genetic_fixed,
                    "GENETIC_BG_COMPONENT": genetic_bg,
                    "GENETIC_COMPONENT": genetic,
                    "COVARIATE_COMPONENT": covariate,
                    "RESIDUAL_COMPONENT": residual,
                }
                if trait_type == "binary":
                    phenotype_payload["LIABILITY"] = liability
                    phenotype_payload["BINARY_PROBABILITY"] = binary_probability
                    phenotype_payload["BINARY_ALPHA"] = binary_alpha
                phenotype_df = pd.DataFrame(phenotype_payload)

                effects_out = effects_detail.copy()
                effects_out["dosage_effect_allele"] = (
                    effects_out["variant_id"].astype(str).map(dosage_effect_allele).fillna("")
                )
                effects_out["dosage_other_allele"] = (
                    effects_out["variant_id"].astype(str).map(dosage_other_allele).fillna("")
                )
                effects_out["architecture"] = architecture.name
                effects_out["cohort"] = cohort.name
                effects_out["replicate"] = replicate

                membership_out = membership.copy()
                membership_out["architecture"] = architecture.name
                membership_out["cohort"] = cohort.name

                records.append(
                    {
                        "replicate": replicate,
                        "cohort": cohort.name,
                        "architecture": architecture.name,
                        "plugin": architecture.plugin,
                        "pgen_prefix": cohort.genotype_prefix_override or "",
                        "sample_ids_file": cohort.sample_ids_file,
                        "trait_type": trait_type,
                        "phenotype_df": phenotype_df,
                        "covariate_df": cov_out.covariate_table,
                        "pcs_df": pcs_df,
                        "membership_df": membership_out,
                        "effects_df": effects_out,
                        "coefficients_df": cov_out.used_coefficients,
                        "component_targets": targets,
                        "component_realized": {
                            "genetic_fixed": float(np.var(genetic_fixed)),
                            "genetic_bg": float(np.var(genetic_bg)),
                            "genetic_total": float(np.var(genetic)),
                            "covariate": float(np.var(covariate)),
                            "residual": float(np.var(residual)),
                            "phenotype": float(np.var(phenotype)),
                            "liability": float(np.var(liability)),
                        },
                        "component_diagnostics": component_diagnostics,
                        "binary_summary": dict(binary_summary),
                        "background_genetic_mode": str(run_cfg.background_genetic.mode),
                        "background_grm_variant_count": int(background_grm_variant_count),
                        "background_grm_excluded_causal_count": int(
                            background_grm_excluded_causal_count
                        ),
                        "genetic_fixed_beta_scale_factor": float(genetic_fixed_beta_scale_factor),
                        "realized_fixed_variance_by_population": dict(
                            realized_fixed_variance_by_population
                        ),
                        "tightness_summary": dict(bucketed_pool.tightness_summary),
                        "effect_geometry": effect_geometry,
                    }
                )

            logger.info(
                "[simulate] cohort_done replicate=%d cohort=%s elapsed_seconds=%.3f",
                int(replicate),
                str(cohort.name),
                float(time.perf_counter() - cohort_t0),
            )

        logger.info(
            "[simulate] replicate_done replicate=%d elapsed_seconds=%.3f",
            int(replicate),
            float(time.perf_counter() - rep_t0),
        )

    try:
        truth_dir = run_dir / "truth"
        truth_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(timing_rows).to_csv(truth_dir / "simulate_timing.tsv", sep="\t", index=False)
        logger.info("SIMULATE_TIMING_WRITTEN run=%s path=%s", str(run_cfg.name), truth_dir / "simulate_timing.tsv")
    except Exception as exc:  # noqa: BLE001
        logger.warning("SIMULATE_TIMING_WRITE_FAILED run=%s error=%s", str(run_cfg.name), str(exc))

    return records


__all__ = [
    "run_simulation",
    "_scale_component_to_variance",
]
