"""Configuration schema models for polygenic phenotype simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .gwas.registry import validate_engine_names


# ── Validation helpers ───────────────────────────────────────────────


def _check_positive(value: int | float, name: str) -> None:
    """Raise if value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _check_non_negative(value: int | float, name: str) -> None:
    """Raise if value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _check_in_range(value: float, lo: float, hi: float, name: str) -> None:
    """Raise if value is not in [lo, hi]."""
    if value < lo or value > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")


def _check_one_of(value: str, choices: set[str], name: str) -> None:
    """Raise if value is not one of the allowed choices."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _check_non_empty(value: str, name: str) -> None:
    """Raise if string value is empty."""
    if not value:
        raise ValueError(f"{name} cannot be empty")


# ── Correlation matrix validator ─────────────────────────────────────


def _validate_correlation_matrix(
    matrix: List[List[float]],
    size: int,
    name: str,
) -> None:
    """Validate that a matrix is square, symmetric, and PSD with unit diagonal."""
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (size, size):
        raise ValueError(f"{name} must be {size}x{size}, got {arr.shape}")
    if not np.allclose(arr, arr.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(arr), np.ones(size), atol=1e-8):
        raise ValueError(f"{name} must have unit diagonal")
    evals = np.linalg.eigvalsh(arr)
    if np.min(evals) < -1e-8:
        raise ValueError(
            f"{name} must be positive semidefinite; minimum eigenvalue={np.min(evals):.3e}"
        )


@dataclass
class HapStoreInputConfig:
    """HapStore input configuration."""

    root_dir: str = ""
    layout: str = "per_pop_per_chr"
    file_pattern: str = "{population}/chr{chrom}.hapstore.zarr"
    store_path: Optional[str] = None
    missing_sentinel: int = 255
    missing_fill_value: float = 0.0
    io_threads: int = 1
    coalesce_max_gap_variants: int = 0
    max_slice_variants: int = 0

    def validate(self) -> None:
        _check_one_of(self.layout, {"per_pop_per_chr", "single_store"}, "genotype_input.hapstore.layout")
        if self.layout == "per_pop_per_chr":
            _check_non_empty(self.root_dir, "genotype_input.hapstore.root_dir")
            if "{population}" not in self.file_pattern or "{chrom}" not in self.file_pattern:
                raise ValueError(
                    "genotype_input.hapstore.file_pattern must include {population} and {chrom}"
                )
        elif self.layout == "single_store":
            if not self.store_path:
                raise ValueError("genotype_input.hapstore.store_path is required for layout=single_store")
        _check_in_range(int(self.missing_sentinel), 0, 255, "genotype_input.hapstore.missing_sentinel")
        if not np.isfinite(float(self.missing_fill_value)):
            raise ValueError("genotype_input.hapstore.missing_fill_value must be finite")
        _check_positive(int(self.io_threads), "genotype_input.hapstore.io_threads")
        _check_non_negative(int(self.coalesce_max_gap_variants), "genotype_input.hapstore.coalesce_max_gap_variants")
        _check_non_negative(int(self.max_slice_variants), "genotype_input.hapstore.max_slice_variants")


@dataclass
class GenotypeInputConfig:
    """Input genotype source configuration."""

    format: str = "hapstore"
    plink2_path: str = "plink2"
    hapstore: HapStoreInputConfig = field(default_factory=HapStoreInputConfig)

    def validate(self) -> None:
        if self.format != "hapstore":
            raise ValueError(
                f"genotype_input.format must be 'hapstore'; "
                f"non-hapstore formats (pgen/vcf/bfile) have been removed. Got {self.format!r}"
            )
        self.hapstore.validate()


@dataclass
class CohortConfig:
    """Cohort definition for simulation and GWAS."""

    name: str
    sample_ids_file: str
    genotype_prefix_override: Optional[str] = None
    population_map_file: Optional[str] = None
    local_ancestry_tracts_pattern: Optional[str] = None

    def validate(self) -> None:
        _check_non_empty(self.name, "cohort.name")
        _check_non_empty(self.sample_ids_file, f"cohort {self.name}: sample_ids_file")
        if self.local_ancestry_tracts_pattern is not None and not str(self.local_ancestry_tracts_pattern):
            raise ValueError(
                f"cohort {self.name}: local_ancestry_tracts_pattern cannot be empty when provided"
            )


@dataclass
class ArchitectureConfig:
    """Architecture plugin and parameters."""

    name: str
    plugin: str = "polygenic_no_pleiotropy"
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _check_non_empty(self.name, "architecture.name")
        _check_non_empty(self.plugin, f"architecture {self.name}: plugin")
        sigma_beta = self.params.get("sigma_beta", 1.0)
        h2 = self.params.get("h2", 0.4)
        _check_positive(sigma_beta, f"architecture {self.name}: sigma_beta")
        if h2 <= 0 or h2 >= 1:
            raise ValueError(f"architecture {self.name}: h2 must be in (0, 1)")


@dataclass
class MafBucketConfig:
    """MAF bucket settings for hybrid fixed+subset causal selection."""

    name: str
    maf_min: float
    maf_max: float
    fixed_count: int
    subset_count: int

    def validate(self) -> None:
        _check_non_empty(self.name, "maf bucket name")
        if not 0 <= self.maf_min < self.maf_max <= 0.5:
            raise ValueError(
                f"bucket {self.name}: invalid MAF bounds [{self.maf_min}, {self.maf_max}]"
            )
        _check_non_negative(self.fixed_count, f"bucket {self.name}: fixed_count")
        _check_non_negative(self.subset_count, f"bucket {self.name}: subset_count")
        if self.fixed_count + self.subset_count == 0:
            raise ValueError(f"bucket {self.name}: fixed_count + subset_count cannot be 0")


@dataclass
class CausalSelectionConfig:
    """Causal pool and MAF bucket selection settings."""

    source_pairs_file: str
    variant_id_column: str = "causal_id"
    maf_columns_by_population: Dict[str, str] = field(default_factory=dict)
    anchor_population: Optional[str] = None
    maf_buckets: List[MafBucketConfig] = field(default_factory=list)
    random_seed: int = 42

    def validate(self) -> None:
        _check_non_empty(self.source_pairs_file, "causal_selection.source_pairs_file")
        _check_non_empty(self.variant_id_column, "causal_selection.variant_id_column")
        if not self.maf_columns_by_population:
            raise ValueError("causal_selection.maf_columns_by_population cannot be empty")
        if self.anchor_population is None:
            self.anchor_population = next(iter(self.maf_columns_by_population.keys()))
        if self.anchor_population not in self.maf_columns_by_population:
            raise ValueError(
                "causal_selection.anchor_population must exist in maf_columns_by_population"
            )
        if not self.maf_buckets:
            raise ValueError("causal_selection.maf_buckets cannot be empty")
        for bucket in self.maf_buckets:
            bucket.validate()

        buckets = sorted(self.maf_buckets, key=lambda b: b.maf_min)
        for left, right in zip(buckets[:-1], buckets[1:]):
            if left.maf_max > right.maf_min:
                raise ValueError(
                    f"MAF buckets overlap: {left.name} ({left.maf_min}-{left.maf_max}) "
                    f"and {right.name} ({right.maf_min}-{right.maf_max})"
                )


@dataclass
class EffectModelConfig:
    """Effect model and population-specific jitter settings."""

    base_beta_distribution: str = "normal"
    base_beta_sd: float = 1.0
    student_t_df: float = 4.0
    mixture_slab_prob: float = 0.1
    mixture_slab_sd_ratio: float = 10.0
    maf_scaling_mode: str = "inverse_sqrt_2pq"
    maf_power_alpha: float = -0.5
    maf_min_clip: float = 1e-4
    maf_max_clip: float = 0.4999
    population_jitter_mode: str = "correlated_deltas"
    population_order: List[str] = field(default_factory=list)
    population_corr_matrix: List[List[float]] = field(default_factory=list)
    jitter_sd_by_population: Dict[str, float] = field(default_factory=dict)
    random_seed: int = 42

    def validate(self) -> None:
        _check_one_of(
            self.base_beta_distribution,
            {"normal", "laplace", "student_t", "mixture_normal"},
            "effect_model.base_beta_distribution",
        )
        _check_positive(self.base_beta_sd, "effect_model.base_beta_sd")
        if self.base_beta_distribution == "student_t":
            if not np.isfinite(self.student_t_df) or float(self.student_t_df) <= 2.0:
                raise ValueError("effect_model.student_t_df must be finite and > 2.0")
        if self.base_beta_distribution == "mixture_normal":
            if not (0.0 < float(self.mixture_slab_prob) < 1.0):
                raise ValueError("effect_model.mixture_slab_prob must be in (0, 1)")
            if not np.isfinite(self.mixture_slab_sd_ratio) or float(self.mixture_slab_sd_ratio) <= 1.0:
                raise ValueError("effect_model.mixture_slab_sd_ratio must be finite and > 1.0")

        _check_one_of(
            self.maf_scaling_mode,
            {"none", "inverse_sqrt_2pq", "power_2pq"},
            "effect_model.maf_scaling_mode",
        )
        if not np.isfinite(self.maf_power_alpha):
            raise ValueError("effect_model.maf_power_alpha must be finite")
        if not 0 < self.maf_min_clip < self.maf_max_clip <= 0.5:
            raise ValueError("effect_model maf clips must satisfy 0 < maf_min_clip < maf_max_clip <= 0.5")

        _check_one_of(
            self.population_jitter_mode,
            {"correlated_deltas"},
            "effect_model.population_jitter_mode",
        )
        if not self.population_order:
            raise ValueError("effect_model.population_order cannot be empty")
        for pop in self.population_order:
            _check_non_negative(
                self.jitter_sd_by_population.get(pop, 0.0),
                f"effect_model jitter SD for {pop}",
            )
        if not self.population_corr_matrix:
            size = len(self.population_order)
            self.population_corr_matrix = np.eye(size).tolist()
        _validate_correlation_matrix(
            self.population_corr_matrix,
            len(self.population_order),
            "effect_model.population_corr_matrix",
        )


@dataclass
class LocalAncestryConfig:
    """Local ancestry truth integration for admixed cohorts."""

    enabled: bool = False
    use_numba: bool = False
    mode: str = "tracts_tsv"
    ancestry_order: List[str] = field(default_factory=list)
    label_map: Dict[str, str] = field(default_factory=dict)
    chunk_n_samples: int = 5000
    chunk_n_variants: int = 2000

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("local_ancestry.enabled must be a boolean")
        if not isinstance(self.use_numba, bool):
            raise ValueError("local_ancestry.use_numba must be a boolean")
        if not self.enabled:
            return
        _check_one_of(self.mode, {"tracts_tsv"}, "local_ancestry.mode")
        if not self.ancestry_order:
            raise ValueError("local_ancestry.ancestry_order cannot be empty when enabled")
        if len(set(self.ancestry_order)) != len(self.ancestry_order):
            raise ValueError("local_ancestry.ancestry_order must contain unique labels")
        for key, value in self.label_map.items():
            if not str(key) or not str(value):
                raise ValueError("local_ancestry.label_map keys/values cannot be empty")
        _check_positive(int(self.chunk_n_samples), "local_ancestry.chunk_n_samples")
        _check_positive(int(self.chunk_n_variants), "local_ancestry.chunk_n_variants")


@dataclass
class CovariateConfig:
    """Non-genetic covariate effect settings."""

    effects_mode: str = "hybrid"
    covariate_file: str = ""
    id_column: str = "IID"
    fixed_effects_file: Optional[str] = None
    random_effect_sd: float = 0.1
    random_seed: int = 42
    target_variance: Optional[float] = None

    def validate(self) -> None:
        _check_one_of(self.effects_mode, {"fixed", "random", "hybrid"}, "covariates.effects_mode")
        _check_non_empty(self.covariate_file, "covariates.covariate_file")
        if self.effects_mode in {"fixed", "hybrid"} and not self.fixed_effects_file:
            raise ValueError(
                "covariates.fixed_effects_file is required when effects_mode is fixed or hybrid"
            )
        _check_non_negative(self.random_effect_sd, "covariates.random_effect_sd")
        if self.target_variance is not None:
            _check_non_negative(self.target_variance, "covariates.target_variance")


@dataclass
class PhenotypeConfig:
    """Phenotype simulation settings."""

    continuous_only: bool = True
    n_replicates: int = 10
    phenotype_column: str = "PHENO"
    component_variance_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "genetic": 0.4,
            "covariate": 0.2,
            "residual": 0.4,
        }
    )
    random_seed: int = 42

    def validate(self) -> None:
        if not self.continuous_only:
            raise ValueError("phase 1 supports only continuous phenotypes")
        _check_positive(self.n_replicates, "phenotype.n_replicates")
        required = {"genetic", "covariate", "residual"}
        if set(self.component_variance_targets.keys()) != required:
            raise ValueError(
                "phenotype.component_variance_targets must contain exactly genetic/covariate/residual"
            )
        vals = self.component_variance_targets
        if any(v < 0 for v in vals.values()):
            raise ValueError("phenotype component variance targets must be non-negative")
        total = vals["genetic"] + vals["covariate"] + vals["residual"]
        _check_positive(total, "phenotype component variance targets sum")


@dataclass
class VarianceBudgetConfig:
    """Variance decomposition targets for simulation components."""

    h2_fixed: float = 0.4
    h2_bg: float = 0.0
    covariate: float = 0.2
    residual: float = 0.4
    strict_sum_to_one: bool = True

    def validate(self) -> None:
        for name, val in [
            ("h2_fixed", self.h2_fixed),
            ("h2_bg", self.h2_bg),
            ("covariate", self.covariate),
            ("residual", self.residual),
        ]:
            _check_non_negative(val, f"variance_budget.{name}")
        total = float(self.h2_fixed + self.h2_bg + self.covariate + self.residual)
        if self.strict_sum_to_one:
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"variance_budget must sum to 1.0 when strict_sum_to_one=true; got {total:.6f}"
                )
        elif total <= 0:
            raise ValueError("variance_budget must sum to > 0")


@dataclass
class BackgroundGeneticConfig:
    """Background infinitesimal genetic component settings."""

    mode: str = "none"
    kinship_file: Optional[str] = None
    grm_variant_source: str = "selected_dosages"
    seed: int = 42

    def validate(self) -> None:
        _check_one_of(self.mode, {"none", "precomputed_kinship", "compute_grm"}, "background_genetic.mode")
        if self.mode == "precomputed_kinship" and not self.kinship_file:
            raise ValueError(
                "background_genetic.kinship_file is required for mode=precomputed_kinship"
            )
        _check_one_of(self.grm_variant_source, {"selected_dosages"}, "background_genetic.grm_variant_source")


@dataclass
class PgenExportConfig:
    """HapStore->PGEN export settings for downstream GWAS."""

    enabled: bool = True
    out_dir: Optional[str] = None
    reuse_exports: bool = True
    task_workers: int = 1
    concat_workers: int = 1
    pgen_shard_workers: int = 1
    variant_scope: str = "all"
    variant_list_file: Optional[str] = None
    target_n_variants: Optional[int] = None
    random_seed: int = 42

    def validate(self) -> None:
        _check_one_of(
            self.variant_scope,
            {"all", "causal_only", "causal_plus_tags", "causal_plus_tags_random", "variant_list"},
            "pgen_export.variant_scope",
        )
        if self.variant_scope == "variant_list" and not self.variant_list_file:
            raise ValueError("pgen_export.variant_list_file is required for variant_scope=variant_list")
        if self.target_n_variants is not None:
            _check_positive(int(self.target_n_variants), "pgen_export.target_n_variants")
        _check_positive(self.task_workers, "pgen_export.task_workers")
        _check_positive(self.concat_workers, "pgen_export.concat_workers")
        _check_positive(self.pgen_shard_workers, "pgen_export.pgen_shard_workers")
        if not isinstance(self.reuse_exports, bool):
            raise ValueError("pgen_export.reuse_exports must be a boolean")


@dataclass
class GlobalPCConfig:
    """Global PC covariate settings."""

    mode: str = "plink2_pca"
    reuse_cache: bool = True
    n_components: int = 10
    plink2_approx: bool = True
    plink2_extra_args: List[str] = field(default_factory=list)
    precomputed_file: Optional[str] = None
    id_column: str = "IID"
    svd_max_variants: int = 50000
    svd_min_maf: float = 0.01
    svd_random_seed: int = 42

    def validate(self) -> None:
        _check_one_of(self.mode, {"precomputed", "none", "plink2_pca", "svd_pca"}, "gwas.global_pcs.mode")
        if not isinstance(self.reuse_cache, bool):
            raise ValueError("gwas.global_pcs.reuse_cache must be a boolean")
        if self.mode == "precomputed" and not self.precomputed_file:
            raise ValueError("gwas.global_pcs.precomputed_file is required for mode=precomputed")
        if self.mode != "none":
            _check_positive(self.n_components, "gwas.global_pcs.n_components")
        _check_positive(self.svd_max_variants, "gwas.global_pcs.svd_max_variants")
        _check_in_range(self.svd_min_maf, 0.0, 0.5, "gwas.global_pcs.svd_min_maf")

    def canonical_mode(self) -> str:
        return self.mode


@dataclass
class Plink2EngineConfig:
    """PLINK2 engine options."""

    path: str = "plink2"
    threads: int = 1
    extra_args: List[str] = field(default_factory=list)

    def validate(self) -> None:
        _check_positive(self.threads, "gwas.plink2.threads")


@dataclass
class RegenieEngineConfig:
    """REGENIE engine options."""

    path: str = "regenie"
    threads: int = 1
    bsize_step1: int = 1000
    bsize_step2: int = 400
    extra_args_step1: List[str] = field(default_factory=list)
    extra_args_step2: List[str] = field(default_factory=list)

    def validate(self) -> None:
        _check_positive(self.threads, "gwas.regenie.threads")
        _check_positive(self.bsize_step1, "gwas.regenie.bsize_step1")
        _check_positive(self.bsize_step2, "gwas.regenie.bsize_step2")


@dataclass
class GwasConfig:
    """GWAS orchestration settings."""

    engines: List[str] = field(default_factory=lambda: ["plink2", "regenie"])
    task_workers: int = 1
    snp_scope: str = "all"
    adjustment_profile: str = "non_genetic_plus_pcs"
    global_pcs: GlobalPCConfig = field(default_factory=GlobalPCConfig)
    plink2: Plink2EngineConfig = field(default_factory=Plink2EngineConfig)
    regenie: RegenieEngineConfig = field(default_factory=RegenieEngineConfig)

    def validate(self) -> None:
        if not self.engines:
            raise ValueError("gwas.engines cannot be empty")
        validate_engine_names(list(self.engines))
        _check_positive(self.task_workers, "gwas.task_workers")
        _check_one_of(self.snp_scope, {"all"}, "gwas.snp_scope")
        _check_one_of(self.adjustment_profile, {"non_genetic_plus_pcs"}, "gwas.adjustment_profile")
        self.global_pcs.validate()
        self.plink2.validate()
        self.regenie.validate()


@dataclass
class RoundTripValidationConfig:
    """Post-GWAS truth-vs-estimate validation settings."""

    enabled: bool = True
    top_k: int = 100
    report_path: Optional[str] = None
    require_positive_slope: bool = True

    def validate(self) -> None:
        _check_positive(int(self.top_k), "roundtrip_validation.top_k")


@dataclass
class PathConfig:
    """Output and temporary path settings."""

    output_root: str
    log_dir: Optional[str] = None
    tmp_dir: Optional[str] = None

    def __post_init__(self) -> None:
        output_root = Path(self.output_root)
        if self.log_dir is None:
            self.log_dir = str(output_root / "logs")
        if self.tmp_dir is None:
            self.tmp_dir = str(output_root / "tmp")

    def create_directories(self) -> None:
        for path_str in [self.output_root, self.log_dir, self.tmp_dir]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


__all__ = [
    "ArchitectureConfig",
    "BackgroundGeneticConfig",
    "CausalSelectionConfig",
    "CohortConfig",
    "CovariateConfig",
    "EffectModelConfig",
    "GenotypeInputConfig",
    "GlobalPCConfig",
    "GwasConfig",
    "HapStoreInputConfig",
    "LocalAncestryConfig",
    "MafBucketConfig",
    "PathConfig",
    "PgenExportConfig",
    "PhenotypeConfig",
    "Plink2EngineConfig",
    "RegenieEngineConfig",
    "RoundTripValidationConfig",
    "VarianceBudgetConfig",
]
