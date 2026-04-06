"""Configuration schema loading and cross-section validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .config_schema import (
    AnnotationSamplingConfig,
    ArchitectureConfig,
    BackgroundGeneticConfig,
    CausalSelectionConfig,
    CohortConfig,
    CovariateConfig,
    CustomAnnotationConfig,
    DefaultGeneWindowConfig,
    EffectModelConfig,
    GenotypeInputConfig,
    GlobalPCConfig,
    GwasConfig,
    HapStoreInputConfig,
    LocalAncestryConfig,
    MafBucketConfig,
    MetapopulationJitterConfig,
    PathConfig,
    PgenExportConfig,
    PhenotypeConfig,
    Plink2EngineConfig,
    RegenieEngineConfig,
    RoundTripPlotConfig,
    RoundTripValidationConfig,
    VarianceBudgetConfig,
)


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
    roundtrip_plots: RoundTripPlotConfig = field(default_factory=RoundTripPlotConfig)
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
        self.roundtrip_plots.validate()

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
            missing = [a for a in self.local_ancestry.ancestry_order if a not in self.effect_model.population_order]
            if missing:
                raise ValueError(
                    f"local_ancestry.ancestry_order labels must exist in effect_model.population_order; missing={missing}"
                )
            la_cohorts = [c for c in self.cohorts if c.local_ancestry_tracts_pattern]
            if not la_cohorts:
                raise ValueError(
                    "local_ancestry.enabled=true requires at least one cohort.local_ancestry_tracts_pattern"
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
        for removed_key in ("temp_dir", "pgen_dosage_backend"):
            if removed_key in geno_payload:
                raise ValueError(f"genotype_input.{removed_key} was removed from the public config contract")
        # Legacy: silently drop keys removed from GenotypeInputConfig.
        for legacy_key in ("path_prefix", "conversion_enabled"):
            geno_payload.pop(legacy_key, None)
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
        annotation_sampling_payload = dict(causal_payload.get("annotation_sampling", {}))
        annotation_sampling_payload["default_gene_window"] = DefaultGeneWindowConfig(
            **annotation_sampling_payload.get("default_gene_window", {})
        )
        annotation_sampling_payload["custom_annotations"] = [
            CustomAnnotationConfig(**item)
            for item in annotation_sampling_payload.get("custom_annotations", [])
        ]
        causal_payload["annotation_sampling"] = AnnotationSamplingConfig(**annotation_sampling_payload)
        causal_selection = CausalSelectionConfig(**causal_payload)

        effect_payload = dict(payload.get("effect_model", {}))
        effect_payload["metapopulation_jitter"] = [
            MetapopulationJitterConfig(**item)
            for item in effect_payload.get("metapopulation_jitter", [])
        ]
        effect_model = EffectModelConfig(**effect_payload)
        local_ancestry = LocalAncestryConfig(**payload.get("local_ancestry", {}))
        covariates = CovariateConfig(**payload["covariates"])
        phenotype = PhenotypeConfig(**payload.get("phenotype", {}))

        # Legacy fallback: derive variance_budget from phenotype.component_variance_targets
        # when no explicit variance_budget section is present in the config YAML.
        # Kept for backward compatibility with pre-variance-budget configs.
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
                raise ValueError(f"pgen_export.{legacy_key} was removed from the public config contract")
        pgen_export = PgenExportConfig(**pgen_payload)

        gwas_payload = dict(payload.get("gwas", {}))
        global_pcs_payload = dict(gwas_payload.get("global_pcs", {}))
        if global_pcs_payload.get("mode") == "compute":
            raise ValueError("gwas.global_pcs.mode=compute was removed; use mode=plink2_pca")
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
        roundtrip_plots = RoundTripPlotConfig(
            **payload.get("roundtrip_plots", {})
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
            roundtrip_plots=roundtrip_plots,
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
