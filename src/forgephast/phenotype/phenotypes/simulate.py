"""Continuous polygenic phenotype simulation orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids, read_sample_ids

from ..architectures.base import ArchitectureInput, create_plugin
from ..causal.hybrid_sampler import sample_membership_for_replicate
from ..causal.selection import build_bucketed_pool
from ..config import CohortConfig, GlobalPCConfig, RunConfig
from ..covariates.model import build_covariate_effects
from ..data.genotypes import (
    GenotypeAdapter,
    build_genotype_adapter,
    check_id_alignment,
)
from ..effects.betas import (
    build_variant_scale_by_maf,
    resolve_effect_geometry,
    sample_population_specific_betas,
)
from ..local_ancestry.tracts import load_tracts_tsv_gz, rasterize_tracts_to_variant_ancestry

logger = logging.getLogger(__name__)


def _stable_string_seed(value: str) -> int:
    """Deterministic seed component from string."""
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value))


def _scale_component_to_variance(component: np.ndarray, target_variance: float) -> np.ndarray:
    """Scale a component vector to desired variance; returns zeros when impossible."""
    if target_variance <= 0:
        return np.zeros_like(component)
    observed = float(np.var(component))
    if observed <= 0:
        return np.zeros_like(component)
    return component * np.sqrt(target_variance / observed)


def _scale_factor_to_variance(component: np.ndarray, target_variance: float) -> float:
    """Return multiplicative scale factor applied by _scale_component_to_variance."""
    if target_variance <= 0:
        return 0.0
    observed = float(np.var(component))
    if observed <= 0:
        return 0.0
    return float(np.sqrt(float(target_variance) / observed))


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


def _update_population_beta_scales(
    *,
    sample_populations: List[str],
    genetic_raw: np.ndarray,
    required_populations: List[str],
    population_targets: Dict[str, Any],
    scale_cache: Dict[str, float],
) -> List[str]:
    """Infer any missing source-population beta scales from observed raw components."""
    populations = np.asarray(sample_populations, dtype=object)
    for pop in required_populations:
        if pop in scale_cache:
            continue
        row_idx = np.where(populations == pop)[0]
        if row_idx.size == 0:
            continue
        scale_cache[pop] = _scale_factor_to_variance(
            genetic_raw[row_idx],
            float(population_targets["resolved_h2_fixed_by_population"][pop]),
        )
    return [pop for pop in required_populations if pop not in scale_cache]


def _apply_population_beta_scales(
    beta_df: pd.DataFrame,
    scale_cache: Dict[str, float],
) -> pd.DataFrame:
    """Apply resolved source-population scales to per-population beta rows."""
    scaled = beta_df.copy()
    beta_scales = scaled["population"].astype(str).map(scale_cache).fillna(1.0).astype(float)
    for col in ("beta", "beta_base", "beta_delta"):
        if col in scaled.columns:
            scaled[col] = pd.to_numeric(scaled[col], errors="coerce") * beta_scales
    scaled["beta_scale"] = beta_scales
    return scaled


def _resolve_population_h2_targets(
    run_cfg: RunConfig,
    architecture_name: str,
    architecture_params: Dict[str, float],
    replicate: int,
) -> Dict[str, Any]:
    """Resolve per-population fixed-heritability targets for one replicate x architecture."""
    base_h2_fixed = float(run_cfg.variance_budget.h2_fixed)
    if "h2" in architecture_params:
        base_h2_fixed = float(architecture_params["h2"])
    max_fixed = max(
        0.0,
        1.0
        - float(run_cfg.variance_budget.h2_bg)
        - float(run_cfg.covariates.target_variance if run_cfg.covariates.target_variance is not None else run_cfg.variance_budget.covariate),
    )

    resolved_by_population: Dict[str, float] = {}
    delta_by_population: Dict[str, float] = {}
    clipped_by_population: Dict[str, bool] = {}
    jitter_enabled = float(run_cfg.variance_budget.h2_jitter_sd) > 0.0 and float(base_h2_fixed) > 0.0
    for pop in run_cfg.effect_model.population_order:
        delta = 0.0
        clipped = False
        if jitter_enabled:
            jitter_seed = (
                int(run_cfg.variance_budget.h2_jitter_seed)
                + 1000 * int(replicate)
                + _stable_string_seed(architecture_name)
                + _stable_string_seed(pop)
            )
            rng = np.random.default_rng(jitter_seed)
            delta = float(rng.normal(0.0, float(run_cfg.variance_budget.h2_jitter_sd)))
        resolved = float(np.clip(base_h2_fixed + delta, 0.0, max_fixed))
        clipped = not np.isclose(resolved, base_h2_fixed + delta, atol=1e-12)
        resolved_by_population[pop] = resolved
        delta_by_population[pop] = float(delta)
        clipped_by_population[pop] = bool(clipped)

    return {
        "base_h2_fixed": float(base_h2_fixed),
        "resolved_h2_fixed_by_population": resolved_by_population,
        "h2_jitter_delta_by_population": delta_by_population,
        "h2_jitter_clipped_by_population": clipped_by_population,
        "genetic_bg": float(run_cfg.variance_budget.h2_bg),
        "covariate": float(
            run_cfg.covariates.target_variance
            if run_cfg.covariates.target_variance is not None
            else run_cfg.variance_budget.covariate
        ),
        "residual_baseline": float(run_cfg.variance_budget.residual),
        "strict_sum_to_one": bool(run_cfg.variance_budget.strict_sum_to_one),
        "h2_jitter_enabled": bool(jitter_enabled),
    }


def _resolve_cohort_component_targets(
    population_targets: Dict[str, Any],
    cohort_populations: List[str],
    realized_genetic_fixed_variance: float,
) -> Dict[str, Any]:
    """Resolve cohort-level reporting targets under population-inherited beta scaling."""
    non_admix = sorted({str(pop) for pop in cohort_populations if str(pop).upper() != "ADMIX"})
    has_single_source = len(non_admix) == 1
    target_fixed = float("nan")
    resolved_h2_fixed = float("nan")
    h2_jitter_delta = float("nan")
    h2_jitter_clipped = False
    target_mode = "inherited_source_betas"
    if has_single_source:
        pop = non_admix[0]
        target_fixed = float(population_targets["resolved_h2_fixed_by_population"][pop])
        resolved_h2_fixed = target_fixed
        h2_jitter_delta = float(population_targets["h2_jitter_delta_by_population"][pop])
        h2_jitter_clipped = bool(population_targets["h2_jitter_clipped_by_population"][pop])
        target_mode = "single_population"

    residual_target = (
        1.0
        - float(realized_genetic_fixed_variance)
        - float(population_targets["genetic_bg"])
        - float(population_targets["covariate"])
    )
    if residual_target < -1e-10 and bool(population_targets["strict_sum_to_one"]):
        raise ValueError(
            "Resolved residual variance is negative after inherited source-beta scaling "
            f"(genetic_fixed={realized_genetic_fixed_variance}, "
            f"genetic_bg={population_targets['genetic_bg']}, covariate={population_targets['covariate']})"
        )
    residual_target = max(0.0, float(residual_target))
    return {
        "genetic_fixed": target_fixed,
        "genetic_bg": float(population_targets["genetic_bg"]),
        "covariate": float(population_targets["covariate"]),
        "residual": residual_target,
        "genetic_total": (
            target_fixed + float(population_targets["genetic_bg"])
            if np.isfinite(target_fixed)
            else float("nan")
        ),
        "base_h2_fixed": float(population_targets["base_h2_fixed"]),
        "resolved_h2_fixed": resolved_h2_fixed,
        "h2_jitter_delta": h2_jitter_delta,
        "h2_jitter_clipped": bool(h2_jitter_clipped),
        "target_mode": target_mode,
        "resolved_h2_fixed_by_population_json": json.dumps(
            population_targets["resolved_h2_fixed_by_population"],
            sort_keys=True,
        ),
        "h2_jitter_delta_by_population_json": json.dumps(
            population_targets["h2_jitter_delta_by_population"],
            sort_keys=True,
        ),
        "h2_jitter_clipped_by_population_json": json.dumps(
            population_targets["h2_jitter_clipped_by_population"],
            sort_keys=True,
        ),
    }


def _load_precomputed_kinship(
    kinship_file: str,
    sample_ids: List[str],
) -> np.ndarray:
    """Load an IID-indexed dense kinship matrix and align to sample order."""
    kin = pd.read_csv(kinship_file, sep="\t")
    id_col = "IID" if "IID" in kin.columns else "#IID" if "#IID" in kin.columns else None
    if id_col is None:
        raise ValueError("precomputed kinship must include IID or #IID column")
    kin = kin.set_index(id_col)

    missing_rows = [iid for iid in sample_ids if iid not in kin.index]
    if missing_rows:
        raise ValueError(
            f"Kinship matrix missing sample rows; first few: {missing_rows[:5]} (n={len(missing_rows)})"
        )
    missing_cols = [iid for iid in sample_ids if iid not in kin.columns]
    if missing_cols:
        raise ValueError(
            f"Kinship matrix missing sample columns; first few: {missing_cols[:5]} (n={len(missing_cols)})"
        )
    aligned = kin.loc[sample_ids, sample_ids]
    arr = aligned.to_numpy(dtype=float)
    arr = 0.5 * (arr + arr.T)
    return arr


def _compute_grm_from_dosage(dosage_df: pd.DataFrame) -> np.ndarray:
    """Compute a simple GRM from standardized dosage columns."""
    variant_cols = [c for c in dosage_df.columns if c not in {"FID", "IID"}]
    x = dosage_df[variant_cols].to_numpy(dtype=float)
    return _compute_grm_from_matrix(x)


def _compute_grm_from_matrix(x: np.ndarray) -> np.ndarray:
    """Compute a simple GRM from an additive genotype matrix."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("Cannot compute GRM with zero dosage variants")
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    keep = (std > 1e-12).reshape(-1)
    if not np.any(keep):
        raise ValueError("Cannot compute GRM: all dosage variants are monomorphic")
    x = x[:, keep]
    x = x / std[:, keep]
    m = max(1, x.shape[1])
    grm = (x @ x.T) / float(m)
    grm = 0.5 * (grm + grm.T)
    return grm


def _sample_frame(sample_ids: List[str]) -> pd.DataFrame:
    """Minimal IID/FID frame for downstream interfaces that require it."""
    return pd.DataFrame({"FID": sample_ids, "IID": sample_ids}, copy=False)


def _sample_mvnormal_from_cov(
    cov: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample one vector from N(0, cov) with eigenvalue clipping for PSD safety."""
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    rng = np.random.default_rng(int(seed))
    z = rng.normal(0.0, 1.0, size=cov.shape[0])
    return evecs @ (np.sqrt(evals) * z)


def _build_background_component(
    run_cfg: RunConfig,
    sample_ids: List[str],
    dosage_df: pd.DataFrame | None,
    genotype_matrix: np.ndarray | None,
    cache_key: str,
    replicate_seed: int,
    kinship_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Build infinitesimal background genetic effect component."""
    mode = run_cfg.background_genetic.mode
    if mode == "none":
        return np.zeros(len(sample_ids), dtype=float)

    if mode == "precomputed_kinship":
        if cache_key not in kinship_cache:
            kinship_cache[cache_key] = _load_precomputed_kinship(
                kinship_file=str(run_cfg.background_genetic.kinship_file),
                sample_ids=sample_ids,
            )
        cov = kinship_cache[cache_key]
    elif mode == "compute_grm":
        if genotype_matrix is not None:
            cov = _compute_grm_from_matrix(genotype_matrix)
        elif dosage_df is not None:
            cov = _compute_grm_from_dosage(dosage_df)
        else:
            raise ValueError("compute_grm mode requires dosage_df or genotype_matrix")
    else:
        raise ValueError(f"Unsupported background_genetic.mode={mode}")

    return _sample_mvnormal_from_cov(cov=cov, seed=replicate_seed)


def _load_population_labels(
    cohort: CohortConfig,
    sample_ids: List[str],
    default_population: str,
) -> List[str]:
    """Load per-sample population labels aligned to sample IDs."""
    if not cohort.population_map_file:
        return [default_population] * len(sample_ids)

    pop_df = pd.read_csv(cohort.population_map_file, sep="\t")
    id_col = "IID" if "IID" in pop_df.columns else "#IID" if "#IID" in pop_df.columns else None
    if id_col is None:
        raise ValueError(
            f"cohort {cohort.name}: population_map_file must contain IID or #IID column"
        )

    pop_col = "population" if "population" in pop_df.columns else None
    if pop_col is None:
        candidates = [c for c in pop_df.columns if c != id_col]
        if not candidates:
            raise ValueError(
                f"cohort {cohort.name}: could not infer population column from map file"
            )
        pop_col = candidates[0]

    aligned = align_by_ids(
        pop_df[[id_col, pop_col]].copy(),
        id_column=id_col,
        ids=sample_ids,
        context=f"cohort {cohort.name} population map alignment",
    )
    return aligned[pop_col].astype(str).tolist()


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

    variant_col = run_cfg.causal_selection.variant_id_column
    default_population = run_cfg.effect_model.population_order[0]
    anchor_population = run_cfg.causal_selection.anchor_population
    if anchor_population is None:
        anchor_population = next(iter(run_cfg.causal_selection.maf_columns_by_population.keys()))
    anchor_maf_col = run_cfg.causal_selection.maf_columns_by_population[anchor_population]
    if runtime_cache is None:
        kinship_cache: Dict[str, np.ndarray] = {}
        tracts_cache: Dict[tuple[str, int, str], dict] = {}
        sample_ids_cache: Dict[str, List[str]] = {}
        sample_pops_cache: Dict[str, List[str]] = {}
        genotype_cache: dict[str, dict[str, Any]] = {}
        effect_geometry_cache: Dict[str, Dict[str, object]] = {}
        population_beta_scale_cache: Dict[str, Dict[str, float]] = {}
        max_cache_bytes: int | None = None
    else:
        kinship_cache = runtime_cache.setdefault("simulate_kinship_cache", {})
        tracts_cache = runtime_cache.setdefault("simulate_tracts_cache", {})
        sample_ids_cache = runtime_cache.setdefault("simulate_sample_ids_cache", {})
        sample_pops_cache = runtime_cache.setdefault("simulate_sample_pops_cache", {})
        genotype_cache = runtime_cache.setdefault("simulate_genotype_cache", {})
        effect_geometry_cache = runtime_cache.setdefault("simulate_effect_geometry_cache", {})
        population_beta_scale_cache = runtime_cache.setdefault("simulate_population_beta_scale_cache", {})
        max_cache_bytes = runtime_cache.get("max_cache_bytes")
        if max_cache_bytes is not None:
            max_cache_bytes = int(max_cache_bytes)

    def _bump(name: str, hit: bool) -> None:
        if runtime_cache is None:
            return
        bucket_name = "cache_hits" if hit else "cache_misses"
        counters = runtime_cache.setdefault(bucket_name, {})
        counters[name] = int(counters.get(name, 0)) + 1

    def _cache_bytes(entry: dict[str, Any]) -> int:
        total = 0
        sample_frame = entry.get("sample_frame")
        if isinstance(sample_frame, pd.DataFrame):
            total += int(sample_frame.memory_usage(deep=True).sum())
        genotype_matrix = entry.get("genotype_matrix")
        if isinstance(genotype_matrix, np.ndarray):
            total += int(genotype_matrix.nbytes)
        admix_haplotypes = entry.get("admix_haplotypes")
        if isinstance(admix_haplotypes, np.ndarray):
            total += int(admix_haplotypes.nbytes)
        admix_local_ancestry_codes = entry.get("admix_local_ancestry_codes")
        if isinstance(admix_local_ancestry_codes, np.ndarray):
            total += int(admix_local_ancestry_codes.nbytes)
        la_genotypes_by_population = entry.get("la_genotypes_by_population")
        if isinstance(la_genotypes_by_population, dict):
            for value in la_genotypes_by_population.values():
                if isinstance(value, np.ndarray):
                    total += int(value.nbytes)
        return total

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
            maf_column=anchor_maf_col,
            effect_model=run_cfg.effect_model,
        )
        _timed("simulate.variant_scale_by_maf", t_membership, replicate=int(replicate))

        for cohort in run_cfg.cohorts:
            cohort_t0 = time.perf_counter()
            if cohort.name in sample_ids_cache:
                sample_ids = sample_ids_cache[cohort.name]
            else:
                sample_ids = read_sample_ids(cohort.sample_ids_file)
                sample_ids_cache[cohort.name] = sample_ids

            if cohort.name in sample_pops_cache:
                sample_populations = sample_pops_cache[cohort.name]
            else:
                sample_populations = _load_population_labels(
                    cohort=cohort,
                    sample_ids=sample_ids,
                    default_population=default_population,
                )
                sample_pops_cache[cohort.name] = sample_populations

            la_enabled = bool(run_cfg.local_ancestry.enabled and cohort.local_ancestry_tracts_pattern)

            # Grid groups may set a per-group HapStore I/O thread budget to balance
            # parallel groups within one SLURM allocation.
            if runtime_cache is not None and hasattr(adapter, "set_io_threads"):
                try:
                    adapter.set_io_threads(int(runtime_cache.get("hapstore_io_threads", 1)))
                except Exception:
                    pass

            cohort_tmp = run_dir / "tmp" / "dosages" / f"rep{replicate}" / cohort.name
            genotype_cache_key = f"{replicate}|{cohort.name}|{la_enabled}|{'|'.join(variant_ids)}"
            if genotype_cache_key in genotype_cache:
                _bump("simulate_genotype", True)
                cached = genotype_cache[genotype_cache_key]
                sample_frame = cached["sample_frame"]
                genotype_matrix = cached.get("genotype_matrix")
                admix_indices = cached.get("admix_indices")
                admix_sample_ids = cached.get("admix_sample_ids")
                admix_haplotypes = cached.get("admix_haplotypes")
                admix_local_ancestry_codes = cached.get("admix_local_ancestry_codes")
                la_genotypes_by_population = cached.get("la_genotypes_by_population", {})
                dosage_effect_allele = dict(cached["dosage_effect_allele"])
            else:
                _bump("simulate_genotype", False)
                sample_frame = _sample_frame(sample_ids)
                genotype_matrix = None
                admix_indices = None
                admix_sample_ids = None
                admix_haplotypes = None
                admix_local_ancestry_codes = None
                la_genotypes_by_population: dict[str, np.ndarray] = {}

                if la_enabled:
                    # Local ancestry mode needs phased haplotypes for *admixed* samples only.
                    # Avoid exporting a full dense dosage table (very expensive at 245k samples).
                    ancestry_order = list(run_cfg.local_ancestry.ancestry_order)
                    ancestry_to_index = {a: i for i, a in enumerate(ancestry_order)}

                    admix_indices = [i for i, p in enumerate(sample_populations) if str(p).upper() == "ADMIX"]
                    admix_sample_ids = [sample_ids[i] for i in admix_indices]
                    admix_sample_pops = [sample_populations[i] for i in admix_indices]

                    if admix_sample_ids:
                        t_hap = time.perf_counter()
                        admix_haplotypes = adapter.export_haplotypes(
                            sample_ids=admix_sample_ids,
                            sample_populations=admix_sample_pops,
                            variant_ids=variant_ids,
                            out_dir=cohort_tmp,
                            chromosomes=run_cfg.chromosomes,
                        )
                        _timed(
                            "simulate.export_haplotypes_admix",
                            t_hap,
                            replicate=int(replicate),
                            cohort=str(cohort.name),
                            n_samples=int(len(admix_sample_ids)),
                            n_variants=int(len(variant_ids)),
                        )
                        if admix_haplotypes.shape != (len(admix_sample_ids), len(variant_ids), 2):
                            raise ValueError(
                                f"Unexpected ADMIX haplotype array shape: {admix_haplotypes.shape} "
                                f"(expected {(len(admix_sample_ids), len(variant_ids), 2)})"
                            )
                        admix_local_ancestry_codes = np.full(
                            (len(admix_sample_ids), len(variant_ids), 2),
                            -1,
                            dtype=np.int8,
                        )

                        pattern = str(cohort.local_ancestry_tracts_pattern)
                        for chrom in run_cfg.chromosomes:
                            idx_for_chrom = variant_indices_by_chrom[int(chrom)]
                            if int(idx_for_chrom.shape[0]) == 0:
                                continue
                            positions_bp = variant_pos_bp[idx_for_chrom]

                            tracts_path = (
                                pattern.format(chrom=int(chrom)) if "{chrom}" in pattern else pattern
                            )
                            cache_key = (cohort.name, int(chrom), tracts_path)
                            if cache_key in tracts_cache:
                                tracts_by_sample_hap = tracts_cache[cache_key]
                            else:
                                t_load = time.perf_counter()
                                tracts_by_sample_hap = load_tracts_tsv_gz(
                                    tracts_path,
                                    sample_ids=admix_sample_ids,
                                    label_map=dict(run_cfg.local_ancestry.label_map),
                                )
                                tracts_cache[cache_key] = tracts_by_sample_hap
                                _timed(
                                    "simulate.load_tracts",
                                    t_load,
                                    replicate=int(replicate),
                                    cohort=str(cohort.name),
                                    chrom=int(chrom),
                                    n_admix_samples=int(len(admix_sample_ids)),
                                )

                            t_rast = time.perf_counter()
                            codes_subset = rasterize_tracts_to_variant_ancestry(
                                sample_ids=admix_sample_ids,
                                tracts_by_sample_hap=tracts_by_sample_hap,
                                variant_positions_bp=positions_bp,
                                ancestry_to_index=ancestry_to_index,
                                chunk_n_samples=int(run_cfg.local_ancestry.chunk_n_samples),
                                use_numba=bool(run_cfg.local_ancestry.use_numba),
                            )
                            _timed(
                                "simulate.rasterize_tracts",
                                t_rast,
                                replicate=int(replicate),
                                cohort=str(cohort.name),
                                chrom=int(chrom),
                                n_admix_samples=int(len(admix_sample_ids)),
                                n_variants=int(len(positions_bp)),
                            )
                            admix_local_ancestry_codes[:, idx_for_chrom, :] = codes_subset.astype(
                                np.int8, copy=False
                            )

                        if np.any(admix_local_ancestry_codes < 0):
                            bad = np.where(admix_local_ancestry_codes < 0)
                            first = (int(bad[0][0]), int(bad[1][0]), int(bad[2][0]))
                            raise ValueError(
                                "ADMIX local ancestry codes are incomplete after tracts rasterization. "
                                f"First missing admix_sample_idx={first[0]} variant_idx={first[1]} hap_idx={first[2]}"
                            )
                else:
                    t_export = time.perf_counter()
                    dosage_df = adapter.export_dosages(
                        sample_ids=sample_ids,
                        sample_populations=sample_populations,
                        variant_ids=variant_ids,
                        out_dir=cohort_tmp,
                        chromosomes=run_cfg.chromosomes,
                    )
                    _timed(
                        "simulate.export_dosages",
                        t_export,
                        replicate=int(replicate),
                        cohort=str(cohort.name),
                        n_samples=int(len(sample_ids)),
                        n_variants=int(len(variant_ids)),
                    )
                    sample_frame = dosage_df[["FID", "IID"]].copy()
                    genotype_matrix = dosage_df[variant_ids].to_numpy(dtype=np.float32, copy=False)
                dosage_effect_allele = dict(adapter.last_export_allele_by_variant)
                cache_entry = {
                    "sample_frame": sample_frame,
                    "genotype_matrix": genotype_matrix,
                    "admix_indices": list(admix_indices) if admix_indices is not None else None,
                    "admix_sample_ids": list(admix_sample_ids) if admix_sample_ids is not None else None,
                    "admix_haplotypes": admix_haplotypes,
                    "admix_local_ancestry_codes": admix_local_ancestry_codes,
                    "la_genotypes_by_population": la_genotypes_by_population,
                    "dosage_effect_allele": dict(dosage_effect_allele),
                }
                entry_bytes = _cache_bytes(cache_entry)
                if max_cache_bytes is None:
                    genotype_cache[genotype_cache_key] = cache_entry
                else:
                    used_bytes = int(runtime_cache.get("simulate_genotype_cache_bytes", 0)) if runtime_cache is not None else 0
                    if used_bytes + entry_bytes <= max_cache_bytes:
                        genotype_cache[genotype_cache_key] = cache_entry
                        if runtime_cache is not None:
                            runtime_cache["simulate_genotype_cache_bytes"] = used_bytes + entry_bytes

            check_id_alignment(sample_ids, sample_frame)

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
                population_targets = _resolve_population_h2_targets(
                    run_cfg=run_cfg,
                    architecture_name=architecture.name,
                    architecture_params=architecture.params,
                    replicate=replicate,
                )
                scale_cache_key = f"{replicate}|{architecture.name}"
                resolved_beta_scales = population_beta_scale_cache.setdefault(scale_cache_key, {})
                required_pops = _required_source_populations(
                    sample_populations=sample_populations,
                    la_enabled=la_enabled,
                    ancestry_order=list(run_cfg.local_ancestry.ancestry_order) if la_enabled else None,
                )

                def _compute_fixed_component_with_betas(
                    beta_df_for_compute: pd.DataFrame,
                ) -> tuple[np.ndarray, pd.DataFrame]:
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

                        if admix_indices:
                            if admix_haplotypes is None or admix_local_ancestry_codes is None or admix_sample_ids is None:
                                raise ValueError("Missing ADMIX haplotypes/codes for local ancestry simulation")
                            admix_idx_arr = np.asarray(admix_indices, dtype=np.int64)
                            la_input = ArchitectureInput(
                                sample_ids=list(admix_sample_ids),
                                sample_populations=["ADMIX"] * len(admix_sample_ids),
                                dosage_df=_sample_frame(list(admix_sample_ids)),
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
                if any(pop not in resolved_beta_scales for pop in required_pops):
                    provisional_raw, _ = _compute_fixed_component_with_betas(beta_df)
                    missing_after_update = _update_population_beta_scales(
                        sample_populations=sample_populations,
                        genetic_raw=provisional_raw,
                        required_populations=required_pops,
                        population_targets=population_targets,
                        scale_cache=resolved_beta_scales,
                    )
                    if missing_after_update:
                        raise ValueError(
                            "Could not resolve source-population h2 scales before inherited-beta simulation; "
                            f"missing populations={missing_after_update} cohort={cohort.name} "
                            f"architecture={architecture.name} replicate={replicate}"
                        )

                scaled_beta_df = _apply_population_beta_scales(beta_df, resolved_beta_scales)
                genetic_fixed, effects_detail = _compute_fixed_component_with_betas(scaled_beta_df)
                _timed(
                    "simulate.compute_genetic_fixed",
                    t_fixed,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    architecture=str(architecture.name),
                    n_samples=int(len(sample_ids)),
                    n_variants=int(len(variant_ids)),
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
                    dosage_df=None,
                    genotype_matrix=genotype_matrix,
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
                targets = _resolve_cohort_component_targets(
                    population_targets=population_targets,
                    cohort_populations=sample_populations,
                    realized_genetic_fixed_variance=float(np.var(genetic_fixed)),
                )

                genetic_bg = _scale_component_to_variance(genetic_bg_raw, targets["genetic_bg"])
                covariate = _scale_component_to_variance(cov_out.contribution, targets["covariate"])
                residual = _scale_component_to_variance(residual_raw, targets["residual"])

                genetic = genetic_fixed + genetic_bg
                phenotype = genetic + covariate + residual
                _timed(
                    "simulate.scale_and_combine",
                    t_scale,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    architecture=str(architecture.name),
                    n_samples=int(len(sample_ids)),
                )

                phenotype_df = pd.DataFrame(
                    {
                        "FID": sample_ids,
                        "IID": sample_ids,
                        run_cfg.phenotype.phenotype_column: phenotype,
                        "GENETIC_FIXED_COMPONENT": genetic_fixed,
                        "GENETIC_BG_COMPONENT": genetic_bg,
                        "GENETIC_COMPONENT": genetic,
                        "COVARIATE_COMPONENT": covariate,
                        "RESIDUAL_COMPONENT": residual,
                    }
                )

                effects_out = effects_detail.copy()
                effects_out["dosage_effect_allele"] = (
                    effects_out["variant_id"].astype(str).map(dosage_effect_allele).fillna("")
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
                        },
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
