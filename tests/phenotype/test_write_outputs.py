"""Simulation output writer tests."""

from __future__ import annotations

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
            "base_h2_fixed": 0.30,
            "resolved_h2_fixed": 0.25,
            "h2_jitter_delta": -0.05,
            "h2_jitter_clipped": False,
            "target_mode": "single_population",
            "resolved_h2_fixed_by_population_json": '{"AFR": 0.28, "EUR": 0.25}',
            "h2_jitter_delta_by_population_json": '{"AFR": -0.02, "EUR": -0.05}',
            "h2_jitter_clipped_by_population_json": '{"AFR": false, "EUR": false}',
            "genetic_fixed": 0.25,
            "genetic_bg": 0.10,
            "genetic_total": 0.35,
            "covariate": 0.15,
            "residual": 0.50,
        },
        "component_realized": {
            "genetic_fixed": 0.24,
            "genetic_bg": 0.09,
            "genetic_total": 0.34,
            "covariate": 0.14,
            "residual": 0.49,
            "phenotype": 0.99,
        },
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
    component_df = pd.read_csv(output_root / "run1" / "truth" / "component_summary.tsv", sep="\t")
    geometry_df = pd.read_csv(output_root / "run1" / "truth" / "effect_geometry.tsv", sep="\t")
    membership_summary_df = pd.read_csv(
        output_root / "run1" / "truth" / "causal_membership_summary.tsv",
        sep="\t",
    )

    assert float(component_df.loc[0, "base_h2_fixed"]) == 0.30
    assert float(component_df.loc[0, "resolved_h2_fixed"]) == 0.25
    assert float(component_df.loc[0, "h2_jitter_delta"]) == -0.05
    assert bool(component_df.loc[0, "h2_jitter_clipped"]) is False
    assert component_df.loc[0, "target_mode"] == "single_population"
    assert component_df.loc[0, "resolved_h2_fixed_by_population"] == '{"AFR": 0.28, "EUR": 0.25}'
    assert float(manifest_df.loc[0, "resolved_h2_fixed"]) == 0.25
    assert manifest_df.loc[0, "target_mode"] == "single_population"
    assert geometry_df.loc[0, "population_corr_mode"] == "auto_from_fst"
    assert geometry_df.loc[0, "resolved_jitter_sd_by_population"] == '{"AFR": 0.3, "EUR": 0.2}'
    assert int(membership_summary_df.loc[0, "selected_causal_count"]) == 2
    assert int(membership_summary_df.loc[0, "annotation_hit_count"]) == 1
    assert float(membership_summary_df.loc[0, "annotation_hit_fraction"]) == 0.5
