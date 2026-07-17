"""Config validation tests."""

from __future__ import annotations

import copy

import pytest

from forgephast.phenotype.config import RunConfig


def _base_payload() -> dict:
    return {
        "name": "unit_test",
        "genotype_input": {
            "format": "hapstore",
            "hapstore": {
                "root_dir": "/tmp/fake",
                "layout": "per_pop_per_chr",
                "file_pattern": "{population}/chr{chrom}.hapstore.zarr",
            },
        },
        "cohorts": [
            {
                "name": "c1",
                "sample_ids_file": "/tmp/samples.txt",
                "population_map_file": "/tmp/pop_map.tsv",
            },
        ],
        "architectures": [
            {
                "name": "a1",
                "plugin": "polygenic_no_pleiotropy",
                "params": {"sigma_beta": 1.0, "h2": 0.1},
            }
        ],
        "causal_selection": {
            "source_pairs_file": "/tmp/pairs.tsv",
            "variant_id_column": "causal_id",
            "maf_columns_by_population": {"EUR": "maf_eur", "AFR": "maf_afr"},
            "anchor_population": "EUR",
            "maf_buckets": [
                {
                    "name": "b1",
                    "maf_min": 0.01,
                    "maf_max": 0.02,
                    "fixed_count": 5,
                    "subset_count": 3,
                }
            ],
        },
        "effect_model": {
            "population_order": ["EUR", "AFR"],
            "population_corr_matrix": [[1.0, 0.4], [0.4, 1.0]],
            "jitter_sd_by_population": {"EUR": 0.1, "AFR": 0.2},
        },
        "covariates": {
            "effects_mode": "hybrid",
            "covariate_file": "/tmp/cov.tsv",
            "id_column": "IID",
            "fixed_effects_file": "/tmp/cov_effect.tsv",
            "random_effect_sd": 0.1,
            "target_variance": 0.2,
        },
        "phenotype": {
            "continuous_only": True,
            "n_replicates": 1,
            "component_variance_targets": {
                "genetic": 0.2,
                "covariate": 0.2,
                "residual": 0.6,
            },
        },
        "variance_budget": {
            "h2_fixed": 0.2,
            "h2_bg": 0.0,
            "covariate": 0.2,
            "residual": 0.6,
            "strict_sum_to_one": True,
        },
        "gwas": {
            "engines": ["plink2", "regenie"],
            "snp_scope": "all",
            "adjustment_profile": "non_genetic_plus_pcs",
            "global_pcs": {"mode": "plink2_pca", "n_components": 5},
        },
        "paths": {
            "output_root": "/tmp/polygenic-tests",
        },
    }


def test_valid_config_builds() -> None:
    payload = _base_payload()
    cfg = RunConfig.from_dict(payload)
    assert cfg.name == "unit_test"
    assert cfg.pgen_export.prep_mode == "per_chrom_plink_merge"
    assert cfg.pgen_export.filter_monomorphic is True
    assert cfg.pgen_export.min_maf == 1e-12
    assert cfg.pgen_export.shard_tmp_dir is None
    assert cfg.gwas.global_pcs.prune_before_pca is True
    assert cfg.gwas.regenie.step1_variant_mode == "ld_pruned_common"
    assert cfg.gwas.regenie.step1_min_maf == 0.01
    assert cfg.gwas.regenie.step1_min_mac == 100
    assert cfg.gwas.regenie.step1_ld_window_variants == 1000
    assert cfg.gwas.regenie.step1_ld_step_variants == 100
    assert cfg.gwas.regenie.step1_ld_r2 == 0.2
    assert cfg.gwas.regenie.step1_variant_warning_threshold == 500_000
    assert cfg.causal_selection.maf_bucket_column == "maf_eur"
    assert cfg.causal_selection.effect_scale_maf_column == "maf_eur"


def test_binary_phenotype_config_builds_with_narrow_contract() -> None:
    payload = _base_payload()
    payload["phenotype"] = {
        "trait_type": "binary",
        "n_replicates": 1,
        "binary": {
            "prevalence": 0.2,
            "liability_logit_scale": 1.5,
            "prevalence_by_population": {"AFR": 0.25},
        },
    }

    cfg = RunConfig.from_dict(payload)

    assert cfg.phenotype.trait_type == "binary"
    assert cfg.phenotype.continuous_only is False
    assert cfg.phenotype.binary.prevalence == 0.2
    assert cfg.phenotype.binary.liability_logit_scale == 1.5
    assert cfg.phenotype.binary.prevalence_by_population == {"AFR": 0.25}
    assert cfg.phenotype.binary.alpha_by_population == {}


def test_binary_phenotype_config_supports_explicit_source_alpha() -> None:
    payload = _base_payload()
    payload["phenotype"] = {
        "trait_type": "binary",
        "n_replicates": 1,
        "binary": {
            "prevalence": 0.2,
            "liability_logit_scale": 1.5,
            "alpha_by_population": {"EUR": -2.0, "AFR": -0.5},
        },
    }

    cfg = RunConfig.from_dict(payload)

    assert cfg.phenotype.binary.alpha_by_population == {"EUR": -2.0, "AFR": -0.5}


def test_binary_phenotype_rejects_mixed_prevalence_and_alpha_maps() -> None:
    payload = _base_payload()
    payload["phenotype"] = {
        "trait_type": "binary",
        "n_replicates": 1,
        "binary": {
            "prevalence_by_population": {"EUR": 0.2},
            "alpha_by_population": {"EUR": -1.0},
        },
    }

    with pytest.raises(ValueError, match="alternative source-alpha inputs"):
        RunConfig.from_dict(payload)


def test_binary_phenotype_rejects_contradictory_continuous_only() -> None:
    payload = _base_payload()
    payload["phenotype"] = {
        "trait_type": "binary",
        "continuous_only": True,
        "n_replicates": 1,
    }
    with pytest.raises(ValueError, match="continuous_only=true"):
        RunConfig.from_dict(payload)


def test_binary_phenotype_rejects_invalid_prevalence() -> None:
    payload = _base_payload()
    payload["phenotype"] = {
        "trait_type": "binary",
        "n_replicates": 1,
        "binary": {"prevalence": 1.0},
    }
    with pytest.raises(ValueError, match="phenotype.binary.prevalence"):
        RunConfig.from_dict(payload)


def test_config_accepts_explicit_bucket_and_effect_scale_maf_columns() -> None:
    payload = _base_payload()
    payload["causal_selection"]["maf_bucket_column"] = "maf_pooled"
    payload["causal_selection"]["effect_scale_maf_column"] = "maf_pooled"
    cfg = RunConfig.from_dict(payload)
    assert cfg.causal_selection.maf_bucket_column == "maf_pooled"
    assert cfg.causal_selection.effect_scale_maf_column == "maf_pooled"


def test_config_accepts_tightness_maf_columns_by_group() -> None:
    payload = _base_payload()
    payload["causal_selection"]["tightness_maf_columns_by_group"] = {
        "EUR": "maf_eur",
        "AFR_GULF": "maf_afr_gulf",
        "AFR_EAST": "maf_afr_east",
        "AFR_WEST": "maf_afr_west",
    }
    cfg = RunConfig.from_dict(payload)
    assert cfg.causal_selection.maf_columns_by_population == {"EUR": "maf_eur", "AFR": "maf_afr"}
    assert cfg.causal_selection.tightness_maf_columns() == {
        "EUR": "maf_eur",
        "AFR_GULF": "maf_afr_gulf",
        "AFR_EAST": "maf_afr_east",
        "AFR_WEST": "maf_afr_west",
    }


def test_config_tightness_maf_columns_fallback_to_population_map() -> None:
    cfg = RunConfig.from_dict(_base_payload())
    assert cfg.causal_selection.tightness_maf_columns() == {"EUR": "maf_eur", "AFR": "maf_afr"}


def test_effect_model_accepts_metapopulation_jitter_placeholder() -> None:
    payload = _base_payload()
    payload["effect_model"]["metapopulation_jitter"] = [
        {
            "coarse_label": "AFR",
            "subpopulation_column": "source_population",
            "subpopulations": ["AFR_WEST", "AFR_GULF", "AFR_EAST"],
            "within_label_subpopulation_deviation_variance": 0.0,
            "subpopulation_corr_matrix": [
                [1.0, 0.95, 0.80],
                [0.95, 1.0, 0.90],
                [0.80, 0.90, 1.0],
            ],
        }
    ]
    cfg = RunConfig.from_dict(payload)
    assert cfg.effect_model.metapopulation_jitter[0].coarse_label == "AFR"


def test_requires_cohort() -> None:
    payload = _base_payload()
    payload["cohorts"] = []
    with pytest.raises(ValueError):
        RunConfig.from_dict(payload)


def test_rejects_overlapping_buckets() -> None:
    payload = _base_payload()
    payload["causal_selection"]["maf_buckets"] = [
        {
            "name": "b1",
            "maf_min": 0.01,
            "maf_max": 0.03,
            "fixed_count": 5,
            "subset_count": 3,
        },
        {
            "name": "b2",
            "maf_min": 0.02,
            "maf_max": 0.04,
            "fixed_count": 5,
            "subset_count": 3,
        },
    ]
    with pytest.raises(ValueError):
        RunConfig.from_dict(payload)


def test_rejects_invalid_population_corr() -> None:
    payload = _base_payload()
    payload["effect_model"]["population_corr_matrix"] = [[1.0, 2.0], [0.1, 1.0]]
    with pytest.raises(ValueError):
        RunConfig.from_dict(payload)


def test_auto_fst_mode_requires_fst_file() -> None:
    payload = _base_payload()
    payload["effect_model"]["population_corr_mode"] = "auto_from_fst"
    payload["effect_model"]["population_corr_matrix"] = []
    with pytest.raises(ValueError, match="population_fst_file"):
        RunConfig.from_dict(payload)


def test_auto_fst_mode_rejects_manual_corr_matrix() -> None:
    payload = _base_payload()
    payload["effect_model"]["population_corr_mode"] = "auto_from_fst"
    payload["effect_model"]["population_fst_file"] = "/tmp/fst.tsv"
    with pytest.raises(ValueError, match="must be omitted"):
        RunConfig.from_dict(payload)


def test_annotation_sampling_rejects_missing_bed_path() -> None:
    payload = _base_payload()
    payload["causal_selection"]["annotation_sampling"] = {
        "enabled": True,
        "custom_annotations": [
            {
                "name": "enhancers",
                "bed_file": "/tmp/does_not_exist.bed",
                "weight_multiplier": 2.0,
            }
        ],
    }
    with pytest.raises(FileNotFoundError, match="bed_file not found"):
        RunConfig.from_dict(payload)


def test_annotation_sampling_rejects_invalid_weight_multiplier() -> None:
    payload = _base_payload()
    payload["causal_selection"]["annotation_sampling"] = {
        "enabled": True,
        "default_gene_window": {
            "enabled": True,
            "genome_build": "GRCh38",
            "weight_multiplier": 0.5,
        },
    }
    with pytest.raises(ValueError, match="weight_multiplier"):
        RunConfig.from_dict(payload)


def test_annotation_sampling_rejects_unsupported_genome_build() -> None:
    payload = _base_payload()
    payload["causal_selection"]["annotation_sampling"] = {
        "enabled": True,
        "default_gene_window": {
            "enabled": True,
            "genome_build": "GRCh37",
            "weight_multiplier": 2.0,
        },
    }
    with pytest.raises(ValueError, match="genome_build"):
        RunConfig.from_dict(payload)


def test_roundtrip_plots_accepts_sample_group_file(tmp_path) -> None:
    payload = _base_payload()
    group_file = tmp_path / "groups.tsv"
    group_file.write_text("IID\tpopulation\ns1\tEUR\ns2\tAFR\n", encoding="utf-8")
    payload["roundtrip_plots"] = {
        "enabled": True,
        "formats": ["png", "pdf"],
        "dpi": 150,
        "sample_group_file": str(group_file),
        "id_column": "IID",
        "group_column": "population",
    }
    cfg = RunConfig.from_dict(payload)
    assert cfg.roundtrip_plots.enabled is True
    assert cfg.roundtrip_plots.formats == ["png", "pdf"]


def test_roundtrip_plots_rejects_missing_group_file() -> None:
    payload = _base_payload()
    payload["roundtrip_plots"] = {
        "enabled": True,
        "sample_group_file": "/tmp/does_not_exist_groups.tsv",
    }
    with pytest.raises(FileNotFoundError, match="sample_group_file"):
        RunConfig.from_dict(payload)


def test_roundtrip_plots_rejects_invalid_format() -> None:
    payload = _base_payload()
    payload["roundtrip_plots"] = {"enabled": True, "formats": ["png", "jpg"]}
    with pytest.raises(ValueError, match="roundtrip_plots.formats"):
        RunConfig.from_dict(payload)


def test_rejects_invalid_maf_scaling_mode() -> None:
    payload = _base_payload()
    payload["effect_model"]["maf_scaling_mode"] = "invalid"
    with pytest.raises(ValueError):
        RunConfig.from_dict(payload)


def test_covariate_target_variance_override_is_optional() -> None:
    payload = _base_payload()
    payload["covariates"].pop("target_variance", None)
    cfg = RunConfig.from_dict(payload)
    assert cfg.covariates.target_variance is None


def test_hapstore_config_accepts_required_fields() -> None:
    payload = _base_payload()
    cfg = RunConfig.from_dict(payload)
    assert cfg.genotype_input.format == "hapstore"
    assert cfg.genotype_input.hapstore.root_dir == "/tmp/fake"


def test_hapstore_rejects_missing_population_map() -> None:
    payload = _base_payload()
    payload["cohorts"][0].pop("population_map_file", None)
    with pytest.raises(ValueError, match="population_map_file"):
        RunConfig.from_dict(payload)


def test_hapstore_rejects_invalid_file_pattern() -> None:
    payload = _base_payload()
    payload["genotype_input"]["hapstore"]["file_pattern"] = "chr{chrom}.hapstore.zarr"
    with pytest.raises(ValueError, match="file_pattern"):
        RunConfig.from_dict(payload)


def test_hapstore_rejects_plink_pca_when_export_disabled() -> None:
    payload = _base_payload()
    payload["gwas"]["global_pcs"]["mode"] = "plink2_pca"
    payload["pgen_export"] = {"enabled": False}
    with pytest.raises(ValueError, match="pgen_export.enabled"):
        RunConfig.from_dict(payload)


def test_rejects_pgen_format() -> None:
    payload = _base_payload()
    payload["genotype_input"] = {"format": "pgen"}
    with pytest.raises(ValueError, match="must be 'hapstore'"):
        RunConfig.from_dict(payload)


def test_rejects_invalid_variance_budget_sum_when_strict() -> None:
    payload = _base_payload()
    payload["variance_budget"]["residual"] = 0.61
    with pytest.raises(ValueError, match="sum to 1.0"):
        RunConfig.from_dict(payload)


def test_rejects_removed_population_h2_settings() -> None:
    payload = _base_payload()
    payload["variance_budget"]["h2_jitter_sd"] = 0.05
    with pytest.raises(ValueError, match="Removed population-level h2 settings"):
        RunConfig.from_dict(payload)


def test_removed_global_pcs_compute_mode_raises() -> None:
    payload = _base_payload()
    payload["gwas"]["global_pcs"]["mode"] = "compute"
    with pytest.raises(ValueError, match="mode=compute was removed"):
        RunConfig.from_dict(payload)


def test_removed_legacy_knobs_raise() -> None:
    payload = _base_payload()
    payload["genotype_input"]["temp_dir"] = "/tmp/old"
    payload["pgen_export"] = {
        "enabled": True,
        "ld_tag_mode": "include_ld_tags",
        "ld_window_kb": 250,
        "ld_r2": 0.8,
    }
    with pytest.raises(ValueError, match="genotype_input.temp_dir"):
        RunConfig.from_dict(payload)


def test_rejects_removed_pgen_dosage_backend() -> None:
    payload = _base_payload()
    payload["genotype_input"]["pgen_dosage_backend"] = "plink2_export"
    with pytest.raises(ValueError, match="genotype_input.pgen_dosage_backend"):
        RunConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("gwas.task_workers", 0, "gwas.task_workers"),
        ("pgen_export.task_workers", 0, "pgen_export.task_workers"),
        ("pgen_export.concat_workers", 0, "pgen_export.concat_workers"),
        ("pgen_export.pgen_shard_workers", 0, "pgen_export.pgen_shard_workers"),
    ],
)
def test_rejects_invalid_worker_knobs(path: str, value: int, match: str) -> None:
    payload = _base_payload()
    cur = payload
    keys = path.split(".")
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict(payload)


def test_rejects_non_boolean_global_pcs_reuse_cache() -> None:
    payload = _base_payload()
    payload["gwas"]["global_pcs"]["reuse_cache"] = "yes"
    with pytest.raises(ValueError, match="reuse_cache"):
        RunConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("step1_variant_mode", "causal", "step1_variant_mode"),
        ("step1_min_maf", 0.0, "step1_min_maf"),
        ("step1_min_mac", 0, "step1_min_mac"),
        ("step1_max_missing", 1.1, "step1_max_missing"),
        ("step1_hwe_p", 0.0, "step1_hwe_p"),
        ("step1_ld_window_variants", 0, "step1_ld_window_variants"),
        ("step1_ld_step_variants", 0, "step1_ld_step_variants"),
        ("step1_ld_r2", 1.0, "step1_ld_r2"),
        ("step1_variant_warning_threshold", 0, "step1_variant_warning_threshold"),
        ("step1_reuse_variant_list", "yes", "step1_reuse_variant_list"),
    ],
)
def test_rejects_invalid_regenie_step1_variant_config(
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _base_payload()
    payload["gwas"].setdefault("regenie", {})[field] = value
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict(payload)


@pytest.mark.parametrize("extract_arg", ["--extract", "--extract=/tmp/variants.txt"])
def test_rejects_regenie_step1_extract_collision(extract_arg: str) -> None:
    payload = _base_payload()
    payload["gwas"]["regenie"] = {
        "step1_variant_mode": "ld_pruned_common",
        "extra_args_step1": [extract_arg, "/tmp/variants.txt"],
    }
    with pytest.raises(ValueError, match="cannot contain --extract"):
        RunConfig.from_dict(payload)


def test_allows_disabled_regenie_step1_variant_warning() -> None:
    payload = _base_payload()
    payload["gwas"]["regenie"] = {"step1_variant_warning_threshold": None}
    cfg = RunConfig.from_dict(payload)
    assert cfg.gwas.regenie.step1_variant_warning_threshold is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("filter_monomorphic", "yes", "filter_monomorphic"),
        ("min_maf", 0.0, "min_maf"),
        ("min_maf", 0.6, "min_maf"),
        ("shard_tmp_dir", "", "shard_tmp_dir"),
    ],
)
def test_rejects_invalid_pgen_filter_config(field: str, value: object, match: str) -> None:
    payload = _base_payload()
    payload.setdefault("pgen_export", {})[field] = value
    with pytest.raises(ValueError, match=match):
        RunConfig.from_dict(payload)


def test_rejects_non_boolean_local_ancestry_use_numba() -> None:
    payload = _base_payload()
    payload["local_ancestry"] = {"enabled": False, "use_numba": "yes"}
    with pytest.raises(ValueError, match="use_numba"):
        RunConfig.from_dict(payload)


def test_local_ancestry_source_zarr_mode_builds() -> None:
    payload = _base_payload()
    payload["chromosomes"] = [22]
    payload["cohorts"][0]["local_ancestry_source_pattern"] = "/tmp/admix.chr{chrom}.local_ancestry.zarr"
    payload["local_ancestry"] = {
        "enabled": True,
        "mode": "source_zarr",
        "ancestry_order": ["EUR", "AFR"],
    }

    cfg = RunConfig.from_dict(payload)

    assert cfg.local_ancestry.mode == "source_zarr"
    assert cfg.cohorts[0].local_ancestry_source_pattern == "/tmp/admix.chr{chrom}.local_ancestry.zarr"


def test_local_ancestry_source_zarr_requires_source_pattern() -> None:
    payload = _base_payload()
    payload["chromosomes"] = [22]
    payload["cohorts"][0]["local_ancestry_tracts_pattern"] = "/tmp/admix.chr{chrom}.tracts.tsv.gz"
    payload["local_ancestry"] = {
        "enabled": True,
        "mode": "source_zarr",
        "ancestry_order": ["EUR", "AFR"],
    }

    with pytest.raises(ValueError, match="local_ancestry_source_pattern"):
        RunConfig.from_dict(payload)


def test_rejects_unknown_top_level_and_gwas_keys() -> None:
    payload = _base_payload()
    payload["chromsome"] = [1]
    with pytest.raises(ValueError, match="Unknown top-level phenotype config key"):
        RunConfig.from_dict(payload)

    payload = _base_payload()
    payload["gwas"]["taks_workers"] = 2
    with pytest.raises(ValueError, match="Unknown gwas config key"):
        RunConfig.from_dict(payload)


@pytest.mark.parametrize("removed_key", ["path_prefix", "conversion_enabled"])
def test_rejects_silently_ignored_legacy_genotype_keys(removed_key: str) -> None:
    payload = _base_payload()
    payload["genotype_input"][removed_key] = "legacy"
    with pytest.raises(ValueError, match=removed_key):
        RunConfig.from_dict(payload)
