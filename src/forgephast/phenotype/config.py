"""Configuration schema and validation for polygenic phenotype simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import yaml


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
        valid_layouts = {"per_pop_per_chr", "single_store"}
        if self.layout not in valid_layouts:
            raise ValueError(
                f"genotype_input.hapstore.layout must be one of {valid_layouts}, got {self.layout}"
            )
        if self.layout == "per_pop_per_chr":
            if not self.root_dir:
                raise ValueError("genotype_input.hapstore.root_dir cannot be empty")
            if "{population}" not in self.file_pattern or "{chrom}" not in self.file_pattern:
                raise ValueError(
                    "genotype_input.hapstore.file_pattern must include {population} and {chrom}"
                )
        elif self.layout == "single_store":
            if not self.store_path:
                raise ValueError("genotype_input.hapstore.store_path is required for layout=single_store")
        else:
            raise ValueError(f"Unsupported genotype_input.hapstore.layout={self.layout}")
        if not 0 <= int(self.missing_sentinel) <= 255:
            raise ValueError("genotype_input.hapstore.missing_sentinel must be in [0, 255]")
        if not np.isfinite(float(self.missing_fill_value)):
            raise ValueError("genotype_input.hapstore.missing_fill_value must be finite")
        if int(self.io_threads) <= 0:
            raise ValueError("genotype_input.hapstore.io_threads must be > 0")
        if int(self.coalesce_max_gap_variants) < 0:
            raise ValueError("genotype_input.hapstore.coalesce_max_gap_variants must be >= 0")
        if int(self.max_slice_variants) < 0:
            raise ValueError("genotype_input.hapstore.max_slice_variants must be >= 0")


@dataclass
class GenotypeInputConfig:
    """Input genotype source configuration."""

    format: str
    path_prefix: str = ""
    conversion_enabled: bool = False
    plink2_path: str = "plink2"
    pgen_dosage_backend: str = "plink2_export"
    hapstore: Optional[HapStoreInputConfig] = None

    def validate(self) -> None:
        valid = {"pgen", "vcf", "bfile", "hapstore"}
        if self.format not in valid:
            raise ValueError(f"genotype_input.format must be one of {valid}, got {self.format}")
        valid_dosage_backends = {"plink2_export", "pgenlib_direct"}
        if self.pgen_dosage_backend not in valid_dosage_backends:
            raise ValueError(
                "genotype_input.pgen_dosage_backend must be one of "
                f"{valid_dosage_backends}, got {self.pgen_dosage_backend}"
            )
        if self.format != "hapstore" and not self.path_prefix:
            raise ValueError("genotype_input.path_prefix cannot be empty")
        if self.format in {"vcf", "bfile"} and not self.conversion_enabled:
            raise ValueError("conversion_enabled must be true when genotype format is not pgen")
        if self.format == "hapstore":
            if self.hapstore is None:
                raise ValueError("genotype_input.hapstore block is required for format=hapstore")
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
        if not self.name:
            raise ValueError("cohort.name cannot be empty")
        if not self.sample_ids_file:
            raise ValueError(f"cohort {self.name}: sample_ids_file cannot be empty")
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
        if not self.name:
            raise ValueError("architecture.name cannot be empty")
        if not self.plugin:
            raise ValueError(f"architecture {self.name}: plugin cannot be empty")
        sigma_beta = self.params.get("sigma_beta", 1.0)
        h2 = self.params.get("h2", 0.4)
        if sigma_beta <= 0:
            raise ValueError(f"architecture {self.name}: sigma_beta must be > 0")
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
        if not self.name:
            raise ValueError("maf bucket name cannot be empty")
        if not 0 <= self.maf_min < self.maf_max <= 0.5:
            raise ValueError(
                f"bucket {self.name}: invalid MAF bounds [{self.maf_min}, {self.maf_max}]"
            )
        if self.fixed_count < 0 or self.subset_count < 0:
            raise ValueError(f"bucket {self.name}: counts must be non-negative")
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
        if not self.source_pairs_file:
            raise ValueError("causal_selection.source_pairs_file cannot be empty")
        if not self.variant_id_column:
            raise ValueError("causal_selection.variant_id_column cannot be empty")
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
        valid_distributions = {"normal", "laplace", "student_t", "mixture_normal"}
        if self.base_beta_distribution not in valid_distributions:
            raise ValueError(
                f"effect_model.base_beta_distribution must be one of {valid_distributions}, got {self.base_beta_distribution!r}"
            )
        if self.base_beta_sd <= 0:
            raise ValueError("effect_model.base_beta_sd must be > 0")
        if self.base_beta_distribution == "student_t":
            if not np.isfinite(self.student_t_df) or float(self.student_t_df) <= 2.0:
                raise ValueError("effect_model.student_t_df must be finite and > 2.0")
        if self.base_beta_distribution == "mixture_normal":
            if not (0.0 < float(self.mixture_slab_prob) < 1.0):
                raise ValueError("effect_model.mixture_slab_prob must be in (0, 1)")
            if not np.isfinite(self.mixture_slab_sd_ratio) or float(self.mixture_slab_sd_ratio) <= 1.0:
                raise ValueError("effect_model.mixture_slab_sd_ratio must be finite and > 1.0")

        valid_maf_scaling = {"none", "inverse_sqrt_2pq", "power_2pq"}
        if self.maf_scaling_mode not in valid_maf_scaling:
            raise ValueError(
                f"effect_model.maf_scaling_mode must be one of {valid_maf_scaling}, got {self.maf_scaling_mode}"
            )
        if not np.isfinite(self.maf_power_alpha):
            raise ValueError("effect_model.maf_power_alpha must be finite")
        if not 0 < self.maf_min_clip < self.maf_max_clip <= 0.5:
            raise ValueError("effect_model maf clips must satisfy 0 < maf_min_clip < maf_max_clip <= 0.5")

        if self.population_jitter_mode != "correlated_deltas":
            raise ValueError(
                "effect_model.population_jitter_mode currently supports only 'correlated_deltas'"
            )
        if not self.population_order:
            raise ValueError("effect_model.population_order cannot be empty")
        for pop in self.population_order:
            if self.jitter_sd_by_population.get(pop, 0.0) < 0:
                raise ValueError(f"effect_model jitter SD for {pop} must be non-negative")
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
        if self.mode != "tracts_tsv":
            raise ValueError("local_ancestry.mode currently supports only 'tracts_tsv'")
        if not self.ancestry_order:
            raise ValueError("local_ancestry.ancestry_order cannot be empty when enabled")
        if len(set(self.ancestry_order)) != len(self.ancestry_order):
            raise ValueError("local_ancestry.ancestry_order must contain unique labels")
        for key, value in self.label_map.items():
            if not str(key) or not str(value):
                raise ValueError("local_ancestry.label_map keys/values cannot be empty")
        if int(self.chunk_n_samples) <= 0:
            raise ValueError("local_ancestry.chunk_n_samples must be > 0")
        if int(self.chunk_n_variants) <= 0:
            raise ValueError("local_ancestry.chunk_n_variants must be > 0")


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
        valid_modes = {"fixed", "random", "hybrid"}
        if self.effects_mode not in valid_modes:
            raise ValueError(
                f"covariates.effects_mode must be one of {valid_modes}, got {self.effects_mode}"
            )
        if not self.covariate_file:
            raise ValueError("covariates.covariate_file cannot be empty")
        if self.effects_mode in {"fixed", "hybrid"} and not self.fixed_effects_file:
            raise ValueError(
                "covariates.fixed_effects_file is required when effects_mode is fixed or hybrid"
            )
        if self.random_effect_sd < 0:
            raise ValueError("covariates.random_effect_sd must be >= 0")
        if self.target_variance is not None and self.target_variance < 0:
            raise ValueError("covariates.target_variance must be >= 0")


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
        if self.n_replicates <= 0:
            raise ValueError("phenotype.n_replicates must be > 0")
        required = {"genetic", "covariate", "residual"}
        if set(self.component_variance_targets.keys()) != required:
            raise ValueError(
                "phenotype.component_variance_targets must contain exactly genetic/covariate/residual"
            )
        vals = self.component_variance_targets
        if any(v < 0 for v in vals.values()):
            raise ValueError("phenotype component variance targets must be non-negative")
        total = vals["genetic"] + vals["covariate"] + vals["residual"]
        if total <= 0:
            raise ValueError("phenotype component variance targets must sum to > 0")


@dataclass
class VarianceBudgetConfig:
    """Variance decomposition targets for simulation components."""

    h2_fixed: float = 0.4
    h2_bg: float = 0.0
    covariate: float = 0.2
    residual: float = 0.4
    strict_sum_to_one: bool = True

    def validate(self) -> None:
        vals = [self.h2_fixed, self.h2_bg, self.covariate, self.residual]
        if any(v < 0 for v in vals):
            raise ValueError("variance_budget fields must be non-negative")
        total = float(sum(vals))
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
        valid_modes = {"none", "precomputed_kinship", "compute_grm"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"background_genetic.mode must be one of {valid_modes}, got {self.mode}"
            )
        if self.mode == "precomputed_kinship" and not self.kinship_file:
            raise ValueError(
                "background_genetic.kinship_file is required for mode=precomputed_kinship"
            )
        if self.grm_variant_source not in {"selected_dosages"}:
            raise ValueError(
                "background_genetic.grm_variant_source currently supports only 'selected_dosages'"
            )


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
        valid_scope = {
            "all",
            "causal_only",
            "causal_plus_tags",
            "causal_plus_tags_random",
            "variant_list",
        }
        if self.variant_scope not in valid_scope:
            raise ValueError(
                f"pgen_export.variant_scope must be one of {valid_scope}, got {self.variant_scope}"
            )
        if self.variant_scope == "variant_list" and not self.variant_list_file:
            raise ValueError("pgen_export.variant_list_file is required for variant_scope=variant_list")
        if self.target_n_variants is not None and int(self.target_n_variants) <= 0:
            raise ValueError("pgen_export.target_n_variants must be > 0 when provided")
        if self.task_workers <= 0:
            raise ValueError("pgen_export.task_workers must be > 0")
        if self.concat_workers <= 0:
            raise ValueError("pgen_export.concat_workers must be > 0")
        if self.pgen_shard_workers <= 0:
            raise ValueError("pgen_export.pgen_shard_workers must be > 0")
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
        valid_modes = {"compute", "precomputed", "none", "plink2_pca", "svd_pca"}
        if self.mode not in valid_modes:
            raise ValueError(f"gwas.global_pcs.mode must be one of {valid_modes}")
        if not isinstance(self.reuse_cache, bool):
            raise ValueError("gwas.global_pcs.reuse_cache must be a boolean")
        if self.mode == "compute":
            warnings.warn(
                "gwas.global_pcs.mode=compute is deprecated; use mode=plink2_pca",
                stacklevel=2,
            )
        if self.mode == "precomputed" and not self.precomputed_file:
            raise ValueError("gwas.global_pcs.precomputed_file is required for mode=precomputed")
        if self.mode != "none" and self.n_components <= 0:
            raise ValueError("gwas.global_pcs.n_components must be > 0")
        if self.svd_max_variants <= 0:
            raise ValueError("gwas.global_pcs.svd_max_variants must be > 0")
        if not 0.0 <= self.svd_min_maf <= 0.5:
            raise ValueError("gwas.global_pcs.svd_min_maf must be in [0, 0.5]")

    def canonical_mode(self) -> str:
        """Return canonical mode key while preserving legacy alias compatibility."""
        if self.mode == "compute":
            return "plink2_pca"
        return self.mode


@dataclass
class Plink2EngineConfig:
    """PLINK2 engine options."""

    path: str = "plink2"
    threads: int = 1
    extra_args: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.threads <= 0:
            raise ValueError("gwas.plink2.threads must be > 0")


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
        if self.threads <= 0:
            raise ValueError("gwas.regenie.threads must be > 0")
        if self.bsize_step1 <= 0 or self.bsize_step2 <= 0:
            raise ValueError("gwas.regenie bsize values must be > 0")


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
        valid_engines = {"plink2", "regenie"}
        if not self.engines:
            raise ValueError("gwas.engines cannot be empty")
        invalid = [e for e in self.engines if e not in valid_engines]
        if invalid:
            raise ValueError(f"unsupported GWAS engine(s): {invalid}")
        if self.task_workers <= 0:
            raise ValueError("gwas.task_workers must be > 0")
        if self.snp_scope != "all":
            raise ValueError("phase 1 currently supports only gwas.snp_scope=all")
        if self.adjustment_profile != "non_genetic_plus_pcs":
            raise ValueError(
                "phase 1 currently supports only gwas.adjustment_profile=non_genetic_plus_pcs"
            )
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
        if int(self.top_k) <= 0:
            raise ValueError("roundtrip_validation.top_k must be > 0")


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


@dataclass
class RunConfig:
    """Top-level run configuration."""

    name: str
    description: str = ""
    genotype_input: GenotypeInputConfig = field(default_factory=GenotypeInputConfig)
    cohorts: List[CohortConfig] = field(default_factory=list)
    architectures: List[ArchitectureConfig] = field(default_factory=list)
    causal_selection: CausalSelectionConfig = field(default_factory=CausalSelectionConfig)
    effect_model: EffectModelConfig = field(default_factory=EffectModelConfig)
    local_ancestry: LocalAncestryConfig = field(default_factory=LocalAncestryConfig)
    covariates: CovariateConfig = field(default_factory=CovariateConfig)
    phenotype: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    variance_budget: VarianceBudgetConfig = field(default_factory=VarianceBudgetConfig)
    background_genetic: BackgroundGeneticConfig = field(default_factory=BackgroundGeneticConfig)
    pgen_export: PgenExportConfig = field(default_factory=PgenExportConfig)
    gwas: GwasConfig = field(default_factory=GwasConfig)
    roundtrip_validation: RoundTripValidationConfig = field(default_factory=RoundTripValidationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    chromosomes: List[int] = field(default_factory=lambda: [1, 3, 4, 5, 6])

    def validate(self) -> None:
        if not self.name:
            raise ValueError("run name cannot be empty")
        self.genotype_input.validate()

        if not self.cohorts:
            raise ValueError("at least one cohort must be specified")
        for cohort in self.cohorts:
            cohort.validate()

        if not self.architectures:
            raise ValueError("at least one architecture must be specified")
        for architecture in self.architectures:
            architecture.validate()

        self.causal_selection.validate()

        if not self.effect_model.population_order:
            self.effect_model.population_order = list(
                self.causal_selection.maf_columns_by_population.keys()
            )
        self.effect_model.validate()
        self.local_ancestry.validate()
        self.covariates.validate()
        self.phenotype.validate()
        self.variance_budget.validate()
        self.background_genetic.validate()
        self.pgen_export.validate()
        self.gwas.validate()
        self.roundtrip_validation.validate()

        if self.genotype_input.format == "hapstore":
            if self.genotype_input.hapstore is None:
                raise ValueError("genotype_input.hapstore must be provided for format=hapstore")
            if self.genotype_input.hapstore.layout == "per_pop_per_chr":
                for cohort in self.cohorts:
                    if not cohort.population_map_file:
                        raise ValueError(
                            f"cohort {cohort.name}: population_map_file is required for hapstore input layout=per_pop_per_chr"
                        )
            if self.gwas.global_pcs.canonical_mode() == "plink2_pca" and not self.pgen_export.enabled:
                raise ValueError(
                    "pgen_export.enabled must be true for hapstore runs with PLINK2 global PCA"
                )
        if self.local_ancestry.enabled:
            la_cohorts = [c for c in self.cohorts if c.local_ancestry_tracts_pattern]
            if la_cohorts and self.genotype_input.format != "hapstore":
                raise ValueError(
                    "local_ancestry.enabled requires genotype_input.format=hapstore"
                )
            missing = [a for a in self.local_ancestry.ancestry_order if a not in self.effect_model.population_order]
            if missing:
                raise ValueError(
                    f"local_ancestry.ancestry_order labels must exist in effect_model.population_order; missing={missing}"
                )
            for cohort in la_cohorts:
                pattern = str(cohort.local_ancestry_tracts_pattern)
                if len(self.chromosomes) > 1 and "{chrom}" not in pattern:
                    raise ValueError(
                        f"cohort {cohort.name}: local_ancestry_tracts_pattern must include {{chrom}} when multiple chromosomes are configured"
                    )

        if not self.chromosomes:
            raise ValueError("chromosomes cannot be empty")
        if any(chrom <= 0 for chrom in self.chromosomes):
            raise ValueError("chromosomes must be positive integers")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunConfig":
        """Build a validated config object from a dictionary."""
        geno_payload = dict(payload["genotype_input"])
        if "temp_dir" in geno_payload:
            warnings.warn(
                "genotype_input.temp_dir is deprecated and ignored",
                stacklevel=2,
            )
            geno_payload.pop("temp_dir", None)
        hapstore_payload = geno_payload.get("hapstore")
        if hapstore_payload is not None:
            geno_payload["hapstore"] = HapStoreInputConfig(**hapstore_payload)
        genotype_input = GenotypeInputConfig(**geno_payload)

        cohorts = [CohortConfig(**item) for item in payload.get("cohorts", [])]
        architectures = [ArchitectureConfig(**item) for item in payload.get("architectures", [])]

        causal_payload = dict(payload["causal_selection"])
        causal_payload["maf_buckets"] = [
            MafBucketConfig(**item) for item in causal_payload.get("maf_buckets", [])
        ]
        causal_selection = CausalSelectionConfig(**causal_payload)

        effect_model = EffectModelConfig(**payload.get("effect_model", {}))
        local_ancestry = LocalAncestryConfig(**payload.get("local_ancestry", {}))
        covariates = CovariateConfig(**payload["covariates"])
        phenotype = PhenotypeConfig(**payload.get("phenotype", {}))

        variance_payload = payload.get("variance_budget")
        if variance_payload is None:
            legacy_targets = dict(phenotype.component_variance_targets)
            variance_payload = {
                "h2_fixed": float(legacy_targets.get("genetic", 0.4)),
                "h2_bg": 0.0,
                "covariate": float(legacy_targets.get("covariate", 0.2)),
                "residual": float(legacy_targets.get("residual", 0.4)),
                "strict_sum_to_one": False,
            }
        variance_budget = VarianceBudgetConfig(**variance_payload)

        background_genetic = BackgroundGeneticConfig(**payload.get("background_genetic", {}))
        pgen_payload = dict(payload.get("pgen_export", {}))
        for legacy_key in ("ld_tag_mode", "ld_window_kb", "ld_r2"):
            if legacy_key in pgen_payload:
                warnings.warn(
                    f"pgen_export.{legacy_key} is deprecated and ignored",
                    stacklevel=2,
                )
                pgen_payload.pop(legacy_key, None)
        pgen_export = PgenExportConfig(**pgen_payload)

        gwas_payload = dict(payload.get("gwas", {}))
        global_pcs_payload = dict(gwas_payload.get("global_pcs", {}))
        if global_pcs_payload.get("mode") == "compute":
            warnings.warn(
                "gwas.global_pcs.mode=compute is deprecated; rewriting to plink2_pca",
                stacklevel=2,
            )
            global_pcs_payload["mode"] = "plink2_pca"
        gwas = GwasConfig(
            engines=gwas_payload.get("engines", ["plink2", "regenie"]),
            task_workers=int(gwas_payload.get("task_workers", 1)),
            snp_scope=gwas_payload.get("snp_scope", "all"),
            adjustment_profile=gwas_payload.get(
                "adjustment_profile", "non_genetic_plus_pcs"
            ),
            global_pcs=GlobalPCConfig(**global_pcs_payload),
            plink2=Plink2EngineConfig(**gwas_payload.get("plink2", {})),
            regenie=RegenieEngineConfig(**gwas_payload.get("regenie", {})),
        )

        roundtrip_validation = RoundTripValidationConfig(
            **payload.get("roundtrip_validation", {})
        )

        paths = PathConfig(**payload["paths"])

        config = cls(
            name=payload["name"],
            description=payload.get("description", ""),
            genotype_input=genotype_input,
            cohorts=cohorts,
            architectures=architectures,
            causal_selection=causal_selection,
            effect_model=effect_model,
            local_ancestry=local_ancestry,
            covariates=covariates,
            phenotype=phenotype,
            variance_budget=variance_budget,
            background_genetic=background_genetic,
            pgen_export=pgen_export,
            gwas=gwas,
            roundtrip_validation=roundtrip_validation,
            paths=paths,
            chromosomes=payload.get("chromosomes", [1, 3, 4, 5, 6]),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Load configuration from YAML path."""
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if not isinstance(payload, dict):
            raise ValueError("Top-level YAML payload must be a mapping")
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary."""
        return asdict(self)
