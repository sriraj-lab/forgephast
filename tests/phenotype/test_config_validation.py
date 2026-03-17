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


def test_rejects_non_boolean_local_ancestry_use_numba() -> None:
    payload = _base_payload()
    payload["local_ancestry"] = {"enabled": False, "use_numba": "yes"}
    with pytest.raises(ValueError, match="use_numba"):
        RunConfig.from_dict(payload)
