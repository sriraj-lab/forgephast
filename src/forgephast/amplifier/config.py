"""Configuration models and loaders for population-amplifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _check_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be > 0")

def _check_non_negative(value, name):
    if value < 0:
        raise ValueError(f"{name} must be >= 0")

def _check_in_range(value, lo, hi, name):
    if value < lo or value > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")

def _check_one_of(value, choices, name):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")

def _check_non_empty(value, name):
    if not value:
        raise ValueError(f"{name} cannot be empty")


@dataclass
class InputConfig:
    metadata_path: str
    population_column: str
    sample_id_column: str
    genetic_map_pattern: str
    panel_vcf_pattern: str | None = None
    panel_hapstore_pattern: str | None = None
    metadata_sep: str | None = None
    allowed_samples_path: str | None = None
    require_chrom_sharded_panel: bool = False
    donor_cache_dir: str | None = None

    def validate(self) -> None:
        for field_name in (
            "metadata_path",
            "population_column",
            "sample_id_column",
            "genetic_map_pattern",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"input.{field_name} cannot be empty")
        if self.panel_vcf_pattern is not None and not str(self.panel_vcf_pattern).strip():
            raise ValueError("input.panel_vcf_pattern cannot be empty when provided")
        if self.panel_hapstore_pattern is not None and not str(self.panel_hapstore_pattern).strip():
            raise ValueError("input.panel_hapstore_pattern cannot be empty when provided")
        if self.panel_vcf_pattern is None and self.panel_hapstore_pattern is None:
            raise ValueError(
                "input requires at least one donor panel pattern: "
                "input.panel_vcf_pattern or input.panel_hapstore_pattern"
            )
        if self.require_chrom_sharded_panel:
            if self.panel_vcf_pattern is not None and "{chrom}" not in self.panel_vcf_pattern:
                raise ValueError(
                    "input.require_chrom_sharded_panel=true requires input.panel_vcf_pattern "
                    "to contain '{chrom}'"
                )
            if self.panel_hapstore_pattern is not None and "{chrom}" not in self.panel_hapstore_pattern:
                raise ValueError(
                    "input.require_chrom_sharded_panel=true requires input.panel_hapstore_pattern "
                    "to contain '{chrom}'"
                )
            if "{chrom}" not in self.genetic_map_pattern:
                raise ValueError(
                    "input.require_chrom_sharded_panel=true requires input.genetic_map_pattern "
                    "to contain '{chrom}'"
                )
        if self.donor_cache_dir is not None and not str(self.donor_cache_dir).strip():
            raise ValueError("input.donor_cache_dir cannot be empty when provided")


@dataclass
class MsprimeSourceSpec:
    population: str
    weight: float

    def validate(self) -> None:
        if not str(self.population).strip():
            raise ValueError("msprime.sources[].population cannot be empty")
        if self.weight <= 0.0 or self.weight > 1.0:
            raise ValueError("msprime.sources[].weight must be in (0.0, 1.0]")


@dataclass
class MsprimePopulationConfig:
    sources: list[MsprimeSourceSpec]
    admixture_time_generations: float | None = None
    end_time_generations: float = 10.0
    write_full_tract_tsv: bool = False
    write_ancestry_source_zarr: bool = False

    def validate(self, *, population_name: str) -> None:
        if not self.sources:
            raise ValueError(f"population {population_name}: msprime.sources cannot be empty")
        if not isinstance(self.write_full_tract_tsv, bool):
            raise ValueError(f"population {population_name}: msprime.write_full_tract_tsv must be a boolean")
        if not isinstance(self.write_ancestry_source_zarr, bool):
            raise ValueError(f"population {population_name}: msprime.write_ancestry_source_zarr must be a boolean")
        for source in self.sources:
            source.validate()
        total_weight = float(sum(float(source.weight) for source in self.sources))
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"population {population_name}: msprime source weights must sum to 1.0 (got {total_weight})"
            )
        if float(self.end_time_generations) <= 0.0:
            raise ValueError(
                f"population {population_name}: msprime.end_time_generations must be > 0"
            )
        if len(self.sources) > 1:
            if self.admixture_time_generations is None:
                raise ValueError(
                    f"population {population_name}: msprime.admixture_time_generations is required when "
                    "len(msprime.sources) > 1"
                )
            if float(self.admixture_time_generations) <= 0.0:
                raise ValueError(
                    f"population {population_name}: msprime.admixture_time_generations must be > 0"
                )
            if float(self.end_time_generations) < float(self.admixture_time_generations):
                raise ValueError(
                    f"population {population_name}: msprime.end_time_generations must be >= "
                    "msprime.admixture_time_generations"
                )
        else:
            if self.admixture_time_generations is not None:
                raise ValueError(
                    f"population {population_name}: msprime.admixture_time_generations must be null when "
                    "len(msprime.sources) == 1"
                )


@dataclass
class PopulationSpec:
    name: str
    n_diploids: int
    selector: Any | None = None
    selector_weights: dict[str, float] | None = None
    engine: str = "mosaic"
    msprime: MsprimePopulationConfig | None = None
    donor_only: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError("population name cannot be empty")
        if not isinstance(self.donor_only, bool):
            raise ValueError(f"population {self.name}: donor_only must be a boolean")
        if self.donor_only and self.engine != "mosaic":
            raise ValueError(f"population {self.name}: donor_only requires engine='mosaic'")
        if self.n_diploids < 0 or (self.n_diploids == 0 and not self.donor_only):
            raise ValueError(
                f"population {self.name}: n_diploids must be > 0 unless donor_only=true"
            )
        if self.engine not in {"mosaic", "msprime_tracts"}:
            raise ValueError(
                f"population {self.name}: engine must be one of {{'mosaic', 'msprime_tracts'}}"
            )
        if self.engine == "mosaic":
            if self.selector is None:
                raise ValueError(f"population {self.name}: selector is required for engine='mosaic'")
            if self.selector_weights is not None:
                if not self.selector_weights:
                    raise ValueError(f"population {self.name}: selector_weights cannot be empty when provided")
                total_weight = 0.0
                for label, weight in self.selector_weights.items():
                    if not str(label).strip():
                        raise ValueError(
                            f"population {self.name}: selector_weights keys cannot be empty"
                        )
                    if float(weight) <= 0.0:
                        raise ValueError(
                            f"population {self.name}: selector_weights values must be > 0"
                        )
                    total_weight += float(weight)
                if abs(total_weight - 1.0) > 1e-6:
                    raise ValueError(
                        f"population {self.name}: selector_weights must sum to 1.0 (got {total_weight})"
                    )
            if self.msprime is not None:
                raise ValueError(f"population {self.name}: msprime config is only allowed for engine='msprime_tracts'")
        else:
            if self.selector_weights is not None:
                raise ValueError(
                    f"population {self.name}: selector_weights is only allowed for engine='mosaic'"
                )
            if self.msprime is None:
                raise ValueError(f"population {self.name}: msprime config is required for engine='msprime_tracts'")
            self.msprime.validate(population_name=self.name)


@dataclass
class SimulationConfig:
    chromosomes: list[int]
    random_seed: int = 1729
    mean_tract_cm: float = 1.0
    threads_per_task: int = 1
    donor_io_workers: int = 1
    haplotype_batch_size: int = 128

    # Unified simulate settings for shard-pool execution.
    n_shards: int | None = None
    task_workers: int | None = None
    task_start_method: str = "fork"

    def validate(self) -> None:
        if not self.chromosomes:
            raise ValueError("simulation.chromosomes cannot be empty")
        if any(c <= 0 for c in self.chromosomes):
            raise ValueError("simulation.chromosomes must be positive integers")
        if self.mean_tract_cm <= 0:
            raise ValueError("simulation.mean_tract_cm must be > 0")
        if self.threads_per_task <= 0:
            raise ValueError("simulation.threads_per_task must be > 0")
        if self.donor_io_workers <= 0:
            raise ValueError("simulation.donor_io_workers must be > 0")
        if self.haplotype_batch_size <= 0:
            raise ValueError("simulation.haplotype_batch_size must be > 0")
        if self.n_shards is not None and self.n_shards <= 0:
            raise ValueError("simulation.n_shards must be > 0 when provided")
        if self.task_workers is not None and self.task_workers <= 0:
            raise ValueError("simulation.task_workers must be > 0 when provided")
        valid_start = {"fork", "spawn", "forkserver"}
        if self.task_start_method not in valid_start:
            raise ValueError(f"simulation.task_start_method must be one of {valid_start}")


@dataclass
class OutputConfig:
    root: str
    scratch_first: bool = True
    scratch_root: str | None = None
    write_phased_vcf: bool = False
    write_pgen: bool = True
    write_hapstore: bool = True
    hapstore_chunk_variant_size: int = 500
    hapstore_chunk_haplotype_size: int | None = None
    hapstore_chunk_sample_size: int | None = None
    hapstore_compression_level: int = 9
    shard_hapstore_compression_level: int | None = None
    hapstore_store_genotypes: bool = False
    write_combined_shard_output: bool = True
    merge_combined_populations: bool = True
    cleanup_shard_hapstores_after_merge: bool = False
    plink2_path: str = "plink2"
    hapstore_vcf_plink2_location: str | None = "system"
    hapstore_vcf_keep_intermediate_pgen: bool = False
    hapstore_vcf_n_shards: int = 1
    hapstore_merge_batch_size: int = 20
    merge_population_workers: int = 1

    def validate(self) -> None:
        _check_non_empty(self.root, "outputs.root")
        if not isinstance(self.scratch_first, bool):
            raise ValueError("outputs.scratch_first must be a boolean")
        if self.scratch_root is not None and not str(self.scratch_root).strip():
            raise ValueError("outputs.scratch_root cannot be empty when provided")
        if not self.write_hapstore:
            raise ValueError("outputs.write_hapstore must be true; amplifier outputs are HapStore-native")
        _check_positive(self.hapstore_chunk_variant_size, "outputs.hapstore_chunk_variant_size")
        if self.hapstore_chunk_haplotype_size is not None:
            _check_positive(self.hapstore_chunk_haplotype_size, "outputs.hapstore_chunk_haplotype_size")
        if self.hapstore_chunk_sample_size is not None:
            _check_positive(self.hapstore_chunk_sample_size, "outputs.hapstore_chunk_sample_size")
        _check_non_negative(self.hapstore_compression_level, "outputs.hapstore_compression_level")
        if self.shard_hapstore_compression_level is not None:
            _check_non_negative(
                self.shard_hapstore_compression_level,
                "outputs.shard_hapstore_compression_level",
            )
        if self.hapstore_vcf_plink2_location is not None and not str(self.hapstore_vcf_plink2_location).strip():
            raise ValueError("outputs.hapstore_vcf_plink2_location cannot be empty when provided")
        _check_positive(self.hapstore_vcf_n_shards, "outputs.hapstore_vcf_n_shards")
        _check_positive(self.hapstore_merge_batch_size, "outputs.hapstore_merge_batch_size")
        _check_positive(self.merge_population_workers, "outputs.merge_population_workers")
        _check_non_empty(self.plink2_path, "outputs.plink2_path")


@dataclass
class RuntimeConfig:
    synthetic_premerge_rr_enabled: bool = False

    def validate(self) -> None:
        if not isinstance(self.synthetic_premerge_rr_enabled, bool):
            raise ValueError("runtime.synthetic_premerge_rr_enabled must be a boolean")


@dataclass
class QcConfig:
    pairwise_fst_sets: list[list[str]] | None = None
    within_pop_split_checks: list[str] | None = None
    fst_tolerance_relative: float = 0.2
    within_split_abs_max: float = 0.002
    backend: str = "python"
    main_pairwise_mode: str = "sampled"
    synthetic_shard_round_robin_enabled: bool = False
    synthetic_shard_round_robin_pairwise_mode: str = "exact"
    seed: int = 42
    sampled_max_diploids_per_population: int = 4000
    sampled_max_variants: int = 50000
    sampled_gate_tolerance_relative: float = 0.30
    fst_stochastic_gate_mode: str = "auto"
    fst_stochastic_small_pop_fraction_threshold: float = 0.75
    fst_stochastic_reps: int = 100
    fst_stochastic_replace: bool = True
    fst_stochastic_min_p_ge: float = 0.05
    matrix_workers: int = 8
    numba_threads: int | None = None
    matrix_variant_block_size: int = 2000
    synthetic_shard_round_robin_max_pairs_per_population: int | None = None
    synthetic_shard_round_robin_min_cpus_per_match: int = 1
    ld_decay_enabled: bool = False
    ld_decay_backend: str = "python"
    ld_decay_min_maf: float = 0.05
    ld_decay_max_distance_cm: float = 2.0
    ld_decay_bin_width_cm: float = 0.05
    ld_decay_min_pairs_per_bin: int = 200
    ld_decay_rmse_max: float = 0.02
    ld_decay_half_decay_rel_diff_max: float = 0.20
    ld_decay_half_decay_r2_threshold: float = 0.20

    ld_heatmap_n_position_bins: int = 200

    king_enabled: bool = False
    afs_enabled: bool = False
    pca_enabled: bool = False
    roh_enabled: bool = False
    hwe_enabled: bool = False
    inbreeding_f_enabled: bool = False

    ext_max_diploids_per_population: int = 2000
    ext_max_variants: int = 20000

    hwe_min_maf: float = 0.01
    hwe_min_n_obs: int = 20
    hwe_logp_max: float = 20.0
    hwe_logp_n_bins: int = 60
    hwe_hist_rmse_max: float | None = None
    hwe_tail_excess_abs_max: float | None = None

    inbreeding_f_min_maf: float = 0.01
    inbreeding_f_min_n_obs: int = 20
    inbreeding_f_hist_min: float = -0.5
    inbreeding_f_hist_max: float = 0.5
    inbreeding_f_n_bins: int = 80
    inbreeding_f_hist_rmse_max: float | None = None
    inbreeding_f_mean_shift_abs_max: float | None = None

    plots_enabled: bool = True
    plots_dirname: str = "plots"

    king_max_within_pairs_per_population: int = 5000
    king_max_cross_pairs_per_pop_pair: int = 5000
    king_hist_n_bins: int = 80
    king_hist_min: float = -0.10
    king_hist_max: float = 0.50

    afs_n_bins: int = 50

    pca_n_components: int = 10
    pca_auc_n_components: int = 10
    pca_min_maf: float = 0.01

    roh_max_diploids_per_population: int = 500
    roh_max_variants: int = 100000
    roh_min_segment_cm: float = 1.5
    roh_max_gap_cm: float = 0.25
    roh_hist_n_bins: int = 60

    engine_aware_gating: bool = True
    msprime_source_weight_abs_tolerance: float = 0.05
    admixed_qc_mode: str = "auto"
    admixed_skip_donor_baseline_pairs: bool = True
    admixed_tract_length_gate_enabled: bool = True
    admixed_tract_length_ks_max: float = 0.08
    admixed_tract_length_min_tracts: int = 5000
    admixed_ancestry_variance_gate_enabled: bool = True
    admixed_ancestry_variance_rel_tolerance: float = 0.30
    admixed_min_individuals_for_variance_gate: int = 200
    admixed_hwe_inbreeding_mode: str = "non_gating"
    admixed_diag_pca_cline_enabled: bool = True
    admixed_diag_king_enabled: bool = True
    admixed_diag_stratified_ld_enabled: bool = False

    def validate(self) -> None:
        _check_non_negative(self.fst_tolerance_relative, "qc.fst_tolerance_relative")
        _check_non_negative(self.within_split_abs_max, "qc.within_split_abs_max")
        _check_one_of(self.backend, {"python"}, "qc.backend")
        _check_one_of(self.main_pairwise_mode, {"sampled", "exact"}, "qc.main_pairwise_mode")
        _check_one_of(
            self.synthetic_shard_round_robin_pairwise_mode,
            {"sampled", "exact"},
            "qc.synthetic_shard_round_robin_pairwise_mode",
        )
        _check_non_negative(self.seed, "qc.seed")
        _check_positive(self.sampled_max_diploids_per_population, "qc.sampled_max_diploids_per_population")
        _check_positive(self.sampled_max_variants, "qc.sampled_max_variants")
        _check_non_negative(self.sampled_gate_tolerance_relative, "qc.sampled_gate_tolerance_relative")
        _check_one_of(self.fst_stochastic_gate_mode, {"auto", "on", "off"}, "qc.fst_stochastic_gate_mode")
        if (
            self.fst_stochastic_small_pop_fraction_threshold <= 0.0
            or self.fst_stochastic_small_pop_fraction_threshold > 1.0
        ):
            raise ValueError("qc.fst_stochastic_small_pop_fraction_threshold must be in (0.0, 1.0]")
        _check_positive(self.fst_stochastic_reps, "qc.fst_stochastic_reps")
        _check_in_range(self.fst_stochastic_min_p_ge, 0.0, 1.0, "qc.fst_stochastic_min_p_ge")
        _check_positive(self.matrix_workers, "qc.matrix_workers")
        if self.numba_threads is not None:
            _check_positive(self.numba_threads, "qc.numba_threads")
        _check_positive(self.matrix_variant_block_size, "qc.matrix_variant_block_size")
        if self.synthetic_shard_round_robin_max_pairs_per_population is not None:
            _check_positive(
                self.synthetic_shard_round_robin_max_pairs_per_population,
                "qc.synthetic_shard_round_robin_max_pairs_per_population",
            )
        _check_positive(
            self.synthetic_shard_round_robin_min_cpus_per_match,
            "qc.synthetic_shard_round_robin_min_cpus_per_match",
        )
        _check_one_of(self.ld_decay_backend, {"python"}, "qc.ld_decay_backend")
        if self.ld_decay_min_maf <= 0 or self.ld_decay_min_maf >= 0.5:
            raise ValueError("qc.ld_decay_min_maf must be in (0.0, 0.5)")
        _check_positive(self.ld_decay_max_distance_cm, "qc.ld_decay_max_distance_cm")
        _check_positive(self.ld_decay_bin_width_cm, "qc.ld_decay_bin_width_cm")
        if self.ld_decay_bin_width_cm >= self.ld_decay_max_distance_cm:
            raise ValueError("qc.ld_decay_bin_width_cm must be < qc.ld_decay_max_distance_cm")
        _check_positive(self.ld_decay_min_pairs_per_bin, "qc.ld_decay_min_pairs_per_bin")
        _check_non_negative(self.ld_decay_rmse_max, "qc.ld_decay_rmse_max")
        _check_non_negative(self.ld_decay_half_decay_rel_diff_max, "qc.ld_decay_half_decay_rel_diff_max")
        if self.ld_decay_half_decay_r2_threshold <= 0 or self.ld_decay_half_decay_r2_threshold >= 1:
            raise ValueError("qc.ld_decay_half_decay_r2_threshold must be in (0.0, 1.0)")
        if self.ld_heatmap_n_position_bins < 5:
            raise ValueError("qc.ld_heatmap_n_position_bins must be >= 5")
        _check_positive(self.ext_max_diploids_per_population, "qc.ext_max_diploids_per_population")
        if self.ext_max_variants <= 1:
            raise ValueError("qc.ext_max_variants must be > 1")
        _check_non_empty(self.plots_dirname, "qc.plots_dirname")
        _check_positive(self.king_max_within_pairs_per_population, "qc.king_max_within_pairs_per_population")
        _check_positive(self.king_max_cross_pairs_per_pop_pair, "qc.king_max_cross_pairs_per_pop_pair")
        if self.king_hist_n_bins < 5:
            raise ValueError("qc.king_hist_n_bins must be >= 5")
        if self.king_hist_max <= self.king_hist_min:
            raise ValueError("qc.king_hist_max must be > qc.king_hist_min")
        if self.afs_n_bins < 5:
            raise ValueError("qc.afs_n_bins must be >= 5")

        _check_positive(self.pca_n_components, "qc.pca_n_components")
        _check_positive(self.pca_auc_n_components, "qc.pca_auc_n_components")
        if self.pca_auc_n_components > self.pca_n_components:
            raise ValueError("qc.pca_auc_n_components must be <= qc.pca_n_components")
        if self.pca_min_maf <= 0 or self.pca_min_maf >= 0.5:
            raise ValueError("qc.pca_min_maf must be in (0.0, 0.5)")

        _check_positive(self.roh_max_diploids_per_population, "qc.roh_max_diploids_per_population")
        _check_positive(self.roh_max_variants, "qc.roh_max_variants")
        _check_positive(self.roh_min_segment_cm, "qc.roh_min_segment_cm")
        _check_positive(self.roh_max_gap_cm, "qc.roh_max_gap_cm")
        if self.roh_hist_n_bins < 5:
            raise ValueError("qc.roh_hist_n_bins must be >= 5")

        if self.hwe_min_maf <= 0.0 or self.hwe_min_maf >= 0.5:
            raise ValueError("qc.hwe_min_maf must be in (0.0, 0.5)")
        _check_positive(self.hwe_min_n_obs, "qc.hwe_min_n_obs")
        _check_positive(self.hwe_logp_max, "qc.hwe_logp_max")
        _check_positive(self.hwe_logp_n_bins, "qc.hwe_logp_n_bins")
        if self.hwe_hist_rmse_max is not None:
            _check_non_negative(self.hwe_hist_rmse_max, "qc.hwe_hist_rmse_max")
        if self.hwe_tail_excess_abs_max is not None:
            _check_non_negative(self.hwe_tail_excess_abs_max, "qc.hwe_tail_excess_abs_max")

        if self.inbreeding_f_min_maf <= 0.0 or self.inbreeding_f_min_maf >= 0.5:
            raise ValueError("qc.inbreeding_f_min_maf must be in (0.0, 0.5)")
        _check_positive(self.inbreeding_f_min_n_obs, "qc.inbreeding_f_min_n_obs")
        if self.inbreeding_f_hist_max <= self.inbreeding_f_hist_min:
            raise ValueError("qc.inbreeding_f_hist_max must be > qc.inbreeding_f_hist_min")
        _check_positive(self.inbreeding_f_n_bins, "qc.inbreeding_f_n_bins")
        if self.inbreeding_f_hist_rmse_max is not None:
            _check_non_negative(self.inbreeding_f_hist_rmse_max, "qc.inbreeding_f_hist_rmse_max")
        if self.inbreeding_f_mean_shift_abs_max is not None:
            _check_non_negative(self.inbreeding_f_mean_shift_abs_max, "qc.inbreeding_f_mean_shift_abs_max")

        _check_in_range(self.msprime_source_weight_abs_tolerance, 0.0, 1.0, "qc.msprime_source_weight_abs_tolerance")
        _check_one_of(self.admixed_qc_mode, {"auto", "off", "strict"}, "qc.admixed_qc_mode")
        _check_in_range(self.admixed_tract_length_ks_max, 0.0, 1.0, "qc.admixed_tract_length_ks_max")
        _check_positive(self.admixed_tract_length_min_tracts, "qc.admixed_tract_length_min_tracts")
        _check_in_range(self.admixed_ancestry_variance_rel_tolerance, 0.0, 10.0, "qc.admixed_ancestry_variance_rel_tolerance")
        _check_positive(self.admixed_min_individuals_for_variance_gate, "qc.admixed_min_individuals_for_variance_gate")
        _check_one_of(self.admixed_hwe_inbreeding_mode, {"non_gating", "legacy"}, "qc.admixed_hwe_inbreeding_mode")


@dataclass
class RunConfig:
    name: str
    input: InputConfig
    populations: list[PopulationSpec]
    simulation: SimulationConfig
    outputs: OutputConfig
    qc: QcConfig = field(default_factory=QcConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")
        self.input.validate()
        if not self.populations:
            raise ValueError("populations cannot be empty")
        seen: set[str] = set()
        pop_by_name: dict[str, PopulationSpec] = {}
        for pop in self.populations:
            pop.validate()
            if pop.name in seen:
                raise ValueError(f"duplicate population name: {pop.name}")
            seen.add(pop.name)
            pop_by_name[pop.name] = pop
        for pop in self.populations:
            if pop.engine != "msprime_tracts":
                continue
            assert pop.msprime is not None
            for source in pop.msprime.sources:
                source_name = str(source.population)
                source_pop = pop_by_name.get(source_name)
                if source_pop is None:
                    raise ValueError(
                        f"population {pop.name}: msprime source population {source_name!r} is not defined"
                    )
                if source_pop.engine != "mosaic":
                    raise ValueError(
                        f"population {pop.name}: msprime source population {source_name!r} must use engine='mosaic'"
                    )
        self.simulation.validate()
        self.outputs.validate()
        self.qc.validate()
        self.runtime.validate()

    @property
    def run_root(self) -> Path:
        return Path(self.outputs.root) / self.name

    @property
    def donor_cache_root(self) -> Path:
        if self.input.donor_cache_dir:
            return Path(self.input.donor_cache_dir)
        return self.run_root / "cache" / "donor_panels"

    @staticmethod
    def _resolve_pattern_candidates(pattern: str, chrom: int, population: str) -> list[Path]:
        candidates: list[Path] = []
        if "{population}" in pattern:
            candidates.append(Path(pattern.format(chrom=chrom, population=population)))
            candidates.append(Path(pattern.format(chrom=chrom, population="")))
        else:
            candidates.append(Path(pattern.format(chrom=chrom)))
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def resolve_donor_panel_source(
        self,
        *,
        chrom: int,
        population: str,
        prefer_existing: bool = True,
    ) -> tuple[str, Path]:
        candidates_by_kind: list[tuple[str, list[Path]]] = []
        if self.input.panel_hapstore_pattern is not None:
            candidates_by_kind.append(
                (
                    "hapstore",
                    self._resolve_pattern_candidates(
                        self.input.panel_hapstore_pattern,
                        chrom=int(chrom),
                        population=str(population),
                    ),
                )
            )
        if self.input.panel_vcf_pattern is not None:
            candidates_by_kind.append(
                (
                    "vcf",
                    self._resolve_pattern_candidates(
                        self.input.panel_vcf_pattern,
                        chrom=int(chrom),
                        population=str(population),
                    ),
                )
            )
        if prefer_existing:
            for kind, candidates in candidates_by_kind:
                for candidate in candidates:
                    if candidate.exists():
                        return kind, candidate
        for kind, candidates in candidates_by_kind:
            if candidates:
                return kind, candidates[0]
        raise ValueError(
            "No donor panel source is configured. Set input.panel_vcf_pattern and/or "
            "input.panel_hapstore_pattern."
        )



def _optional_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    raw = data.get(key, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a mapping")
    return raw


def _construct_dataclass(section: str, cls: type[Any], kwargs: dict[str, Any]) -> Any:
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid {section} config keys: {exc}") from exc



def from_dict(data: dict[str, Any]) -> RunConfig:
    allowed_top_level = {
        "name",
        "input",
        "populations",
        "simulation",
        "outputs",
        "qc",
        "runtime",
        "calibration",
        "benchmark",
    }
    unknown_top_level = sorted(set(data).difference(allowed_top_level))
    if unknown_top_level:
        raise ValueError(f"Unknown top-level config key(s): {unknown_top_level}")

    removed_sections = [x for x in ("calibration", "benchmark") if x in data]
    if removed_sections:
        joined = ", ".join(removed_sections)
        raise ValueError(
            f"Unsupported top-level config section(s): {joined}. "
            "Benchmark/calibration were removed from core; use scripts/legacy."
        )

    input_cfg = _construct_dataclass("input", InputConfig, data["input"])
    populations_raw = data["populations"]
    if not isinstance(populations_raw, list):
        raise ValueError("populations must be a list")
    populations: list[PopulationSpec] = []
    for idx, entry in enumerate(populations_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"populations[{idx}] must be a mapping")
        payload = dict(entry)
        msprime_raw = payload.get("msprime")
        if msprime_raw is not None:
            if not isinstance(msprime_raw, dict):
                raise ValueError(f"populations[{idx}].msprime must be a mapping")
            msprime_payload = dict(msprime_raw)
            sources_raw = msprime_payload.get("sources", [])
            if not isinstance(sources_raw, list):
                raise ValueError(f"populations[{idx}].msprime.sources must be a list")
            msprime_payload["sources"] = [
                _construct_dataclass(
                    f"populations[{idx}].msprime.sources[{src_idx}]",
                    MsprimeSourceSpec,
                    source,
                )
                for src_idx, source in enumerate(sources_raw)
            ]
            payload["msprime"] = _construct_dataclass(
                f"populations[{idx}].msprime",
                MsprimePopulationConfig,
                msprime_payload,
            )
        populations.append(_construct_dataclass(f"populations[{idx}]", PopulationSpec, payload))

    sim_cfg = _construct_dataclass("simulation", SimulationConfig, data["simulation"])

    outputs_raw = dict(data["outputs"])
    removed_output_keys = {
        "pgen_source_preference": "outputs.pgen_source_preference was removed; PGEN export is now HapStore-direct only",
        "merge_backend": "outputs.merge_backend was removed; VCF merge backends are no longer supported",
        "bcftools_path": "outputs.bcftools_path was removed with VCF merge support",
        "bcftools_threads": "outputs.bcftools_threads was removed with VCF merge support",
        "bcftools_no_index": "outputs.bcftools_no_index was removed with VCF merge support",
        "plink1_path": "outputs.plink1_path was removed; PLINK1-based VCF/BED bridge paths are no longer supported",
    }
    for key, message in removed_output_keys.items():
        if key in outputs_raw:
            raise ValueError(message)
    outputs_cfg = _construct_dataclass("outputs", OutputConfig, outputs_raw)
    qc_raw = dict(_optional_section(data, "qc"))
    removed_qc_keys = {
        "rr_pairwise_mode": "qc.rr_pairwise_mode was removed; use qc.synthetic_shard_round_robin_pairwise_mode",
        "synthetic_round_robin_max_pairs_per_population": (
            "qc.synthetic_round_robin_max_pairs_per_population was removed; "
            "use qc.synthetic_shard_round_robin_max_pairs_per_population"
        ),
        "synthetic_rr_min_cpus_per_match": (
            "qc.synthetic_rr_min_cpus_per_match was removed; "
            "use qc.synthetic_shard_round_robin_min_cpus_per_match"
        ),
    }
    for key, message in removed_qc_keys.items():
        if key in qc_raw:
            raise ValueError(message)
    qc_cfg = _construct_dataclass("qc", QcConfig, qc_raw)
    runtime_cfg = _construct_dataclass("runtime", RuntimeConfig, _optional_section(data, "runtime"))

    cfg = RunConfig(
        name=data.get("name", "population_amplifier_run"),
        input=input_cfg,
        populations=populations,
        simulation=sim_cfg,
        outputs=outputs_cfg,
        qc=qc_cfg,
        runtime=runtime_cfg,
    )
    cfg.validate()
    return cfg



def from_yaml(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config YAML must decode to a mapping")
    return from_dict(data)
