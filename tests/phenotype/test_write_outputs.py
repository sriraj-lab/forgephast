"""Simulation output writer tests."""

from __future__ import annotations

import json

import pandas as pd

from forgephast.phenotype.io.write_outputs import write_simulation_outputs


def test_write_simulation_outputs_persists_h2_and_effect_geometry(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    record = {
        "replicate": 1,
        "architecture": "arch1",
        "cohort": "cohort1",
        "plugin": "polygenic_no_pleiotropy",
        "pgen_prefix": "",
        "sample_ids_file": "/tmp/samples.txt",
        "trait_type": "binary",
        "phenotype_df": pd.DataFrame(
            {"FID": ["s1"], "IID": ["s1"], "PHENO": [0.1]}
        ),
        "covariate_df": pd.DataFrame({"FID": ["s1"], "IID": ["s1"]}),
        "pcs_df": pd.DataFrame({"FID": ["s1"], "IID": ["s1"]}),
        "coefficients_df": pd.DataFrame({"covariate": ["age"], "beta": [0.2]}),
        "membership_df": pd.DataFrame(
            {
                "variant_id": ["v1", "v2"],
                "replicate": [1, 1],
                "architecture": ["arch1", "arch1"],
                "cohort": ["cohort1", "cohort1"],
                "maf_bucket": ["rare", "rare"],
                "annotation_hit": [True, False],
                "annotation_labels": ["default_gene_window", ""],
                "sampling_weight": [5.0, 1.0],
            }
        ),
        "effects_df": pd.DataFrame({"variant_id": ["v1"], "beta": [0.3]}),
        "component_targets": {
            "target_h2_fixed": 0.30,
            "genetic_fixed": 0.30,
            "genetic_bg": 0.10,
            "genetic_total": 0.40,
            "covariate": 0.15,
            "residual": 0.45,
        },
        "component_realized": {
            "genetic_fixed": 0.24,
            "genetic_bg": 0.09,
            "genetic_total": 0.34,
            "covariate": 0.14,
            "residual": 0.49,
            "phenotype": 0.99,
            "liability": 1.00,
        },
        "component_diagnostics": {
            "pairwise_covariances": {
                "genetic_fixed|genetic_bg": 0.01,
                "genetic_fixed|covariate": 0.04,
            },
            "component_variance_sum": 0.96,
            "covariance_contribution": 0.04,
            "covariance_relative_contribution": 0.0416666667,
            "covariance_warning_threshold": 0.05,
            "covariance_warning": False,
            "realized_fractions": {
                "genetic_fixed": 0.24,
                "genetic_bg": 0.09,
                "genetic_total": 0.34,
                "covariate": 0.14,
                "residual": 0.49,
            },
        },
        "binary_summary": {
            "binary_prevalence_target": 0.20,
            "binary_liability_logit_scale": 1.5,
            "binary_alpha": -1.25,
            "binary_alpha_by_population": '{"__cohort__": -1.25}',
            "binary_prevalence_target_by_population": '{"__cohort__": 0.2}',
            "binary_case_count": 1,
            "binary_control_count": 4,
            "binary_realized_prevalence": 0.2,
            "binary_mean_predicted_probability": 0.2,
        },
        "background_genetic_mode": "compute_grm",
        "background_grm_variant_count": 1200,
        "background_grm_excluded_causal_count": 2,
        "genetic_fixed_beta_scale_factor": 0.42,
        "effect_geometry": {
            "population_corr_mode": "auto_from_fst",
            "base_jitter_sd": 0.2,
            "resolved_jitter_sd_by_population": {"EUR": 0.2, "AFR": 0.3},
            "population_fst_file": "/tmp/fst.tsv",
            "fst_corr_transform": "exp_decay",
            "fst_corr_lambda": 10.0,
            "population_corr_matrix_json": "[[1.0,0.5],[0.5,1.0]]",
        },
    }

    manifest_df = write_simulation_outputs("run1", output_root, [record])
    completion = json.loads(
        (output_root / "run1" / "SIMULATE_COMPLETE.json").read_text(encoding="utf-8")
    )
    component_df = pd.read_csv(output_root / "run1" / "truth" / "component_summary.tsv", sep="\t")
    geometry_df = pd.read_csv(output_root / "run1" / "truth" / "effect_geometry.tsv", sep="\t")
    membership_summary_df = pd.read_csv(
        output_root / "run1" / "truth" / "causal_membership_summary.tsv",
        sep="\t",
    )

    assert float(component_df.loc[0, "target_h2_fixed"]) == 0.30
    assert float(component_df.loc[0, "genetic_fixed_beta_scale_factor"]) == 0.42
    assert float(component_df.loc[0, "target_var_genetic_fixed"]) == 0.30
    assert component_df.loc[0, "trait_type"] == "binary"
    assert manifest_df.loc[0, "trait_type"] == "binary"
    assert float(component_df.loc[0, "realized_var_liability"]) == 1.00
    assert float(component_df.loc[0, "realized_fraction_genetic_fixed"]) == 0.24
    assert bool(component_df.loc[0, "component_covariance_warning"]) is False
    assert float(component_df.loc[0, "realized_component_variance_sum"]) == 0.96
    assert float(component_df.loc[0, "component_covariance_contribution"]) == 0.04
    assert float(component_df.loc[0, "component_covariance_warning_threshold"]) == 0.05
    assert int(component_df.loc[0, "background_grm_variant_count"]) == 1200
    assert int(manifest_df.loc[0, "background_grm_excluded_causal_count"]) == 2
    assert json.loads(component_df.loc[0, "component_pairwise_covariances"])[
        "genetic_fixed|covariate"
    ] == 0.04
    assert float(component_df.loc[0, "binary_prevalence_target"]) == 0.20
    assert float(manifest_df.loc[0, "binary_liability_logit_scale"]) == 1.5
    assert component_df.loc[0, "binary_alpha_by_population"] == '{"__cohort__": -1.25}'
    assert int(manifest_df.loc[0, "binary_case_count"]) == 1
    assert int(manifest_df.loc[0, "binary_control_count"]) == 4
    assert completion["schema_version"] == 2
    assert completion["n_manifest_rows"] == 1
    assert float(manifest_df.loc[0, "binary_realized_prevalence"]) == 0.2
    assert float(manifest_df.loc[0, "target_h2_fixed"]) == 0.30
    assert float(manifest_df.loc[0, "genetic_fixed_beta_scale_factor"]) == 0.42
    assert geometry_df.loc[0, "population_corr_mode"] == "auto_from_fst"
    assert geometry_df.loc[0, "resolved_jitter_sd_by_population"] == '{"AFR": 0.3, "EUR": 0.2}'
    assert int(membership_summary_df.loc[0, "selected_causal_count"]) == 2
    assert int(membership_summary_df.loc[0, "annotation_hit_count"]) == 1
    assert float(membership_summary_df.loc[0, "annotation_hit_fraction"]) == 0.5


def test_write_simulation_outputs_population_variance_diagnostic_and_tightness(
    tmp_path,
) -> None:
    output_root = tmp_path / "outputs"
    record = {
        "replicate": 2,
        "architecture": "arch2",
        "cohort": "cohort_multi",
        "plugin": "polygenic_no_pleiotropy",
        "pgen_prefix": "",
        "sample_ids_file": "/tmp/samples.txt",
        "phenotype_df": pd.DataFrame(
            {"FID": ["s1", "s2"], "IID": ["s1", "s2"], "PHENO": [0.1, -0.2]}
        ),
        "covariate_df": pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}),
        "pcs_df": pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}),
        "coefficients_df": pd.DataFrame({"covariate": ["age"], "beta": [0.2]}),
        "membership_df": pd.DataFrame(
            {
                "variant_id": ["v1"],
                "replicate": [2],
                "architecture": ["arch2"],
                "cohort": ["cohort_multi"],
                "maf_bucket": ["rare"],
                "annotation_hit": [False],
                "annotation_labels": [""],
                "sampling_weight": [1.0],
            }
        ),
        "effects_df": pd.DataFrame({"variant_id": ["v1"], "beta": [0.3]}),
        "component_targets": {
            "target_h2_fixed": 0.30,
            "genetic_fixed": 0.30,
            "genetic_bg": 0.10,
            "genetic_total": 0.40,
            "covariate": 0.15,
            "residual": 0.45,
        },
        "component_realized": {
            "genetic_fixed": 0.21,
            "genetic_bg": 0.09,
            "genetic_total": 0.30,
            "covariate": 0.14,
            "residual": 0.40,
            "phenotype": 0.84,
        },
        "genetic_fixed_beta_scale_factor": 0.75,
        "realized_fixed_variance_by_population": {"EUR": 0.18, "AFR": 0.24},
        "tightness_summary": {
            "tier": "moderate",
            "fold_change_cap": 5.0,
            "maf_floor": 1e-4,
            "pre_filter_count": 10,
            "post_filter_count": 7,
            "dropped": 3,
            "fold_change_max_kept": 4.75,
            "per_pair_quantiles": {
                "EUR|AFR": {
                    "q50": 1.5,
                    "q75": 3.0,
                    "q90": 4.0,
                    "q99": 4.7,
                    "max": 4.75,
                }
            },
        },
        "effect_geometry": {
            "population_corr_mode": "manual",
            "base_jitter_sd": 0.1,
            "resolved_jitter_sd_by_population": {"EUR": 0.1, "AFR": 0.1},
            "population_fst_file": "",
            "fst_corr_transform": "",
            "fst_corr_lambda": 0.0,
            "population_corr_matrix_json": "[[1.0,0.8],[0.8,1.0]]",
        },
    }

    manifest_df = write_simulation_outputs("run_multi", output_root, [record])
    component_df = pd.read_csv(
        output_root / "run_multi" / "truth" / "component_summary.tsv", sep="\t"
    )

    assert float(component_df.loc[0, "target_h2_fixed"]) == 0.30
    assert float(manifest_df.loc[0, "target_h2_fixed"]) == 0.30
    assert float(component_df.loc[0, "genetic_fixed_beta_scale_factor"]) == 0.75
    assert float(manifest_df.loc[0, "genetic_fixed_beta_scale_factor"]) == 0.75
    # realized per-pop h2 is serialized with sorted keys
    assert (
        component_df.loc[0, "realized_var_genetic_fixed_by_population"]
        == '{"AFR": 0.24, "EUR": 0.18}'
    )
    assert (
        manifest_df.loc[0, "realized_var_genetic_fixed_by_population"]
        == '{"AFR": 0.24, "EUR": 0.18}'
    )
    assert component_df.loc[0, "shift_tightness_tier"] == "moderate"
    assert float(component_df.loc[0, "shift_tightness_fold_change_cap"]) == 5.0
    assert float(component_df.loc[0, "shift_tightness_fold_change_max_kept"]) == 4.75
    assert int(component_df.loc[0, "shift_tightness_pre_filter_count"]) == 10
    assert int(component_df.loc[0, "shift_tightness_post_filter_count"]) == 7
    # Quantiles JSON should round-trip
    import json as _json

    quantiles = _json.loads(component_df.loc[0, "shift_tightness_fold_change_quantiles_json"])
    assert "EUR|AFR" in quantiles
    assert quantiles["EUR|AFR"]["max"] == 4.75
