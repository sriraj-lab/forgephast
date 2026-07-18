"""Phenotype component scaling tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.phenotypes.simulate import (
    _admix_alpha_weights_from_local_ancestry,
    _binary_phenotype_from_liability,
    _calibrate_logistic_intercept,
    _component_variance_diagnostics,
    _exclude_causal_grm_variants,
    _resolve_cohort_component_targets,
    _sample_background_from_grm_factor,
    _scale_effects_detail_betas,
    _scale_component_to_variance,
    _standardize_grm_genotypes,
    run_simulation,
)


def test_scale_component_to_variance_hits_target() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = _scale_component_to_variance(x, 0.25)
    assert abs(np.var(y) - 0.25) < 0.02


def test_scale_component_zero_target() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = _scale_component_to_variance(x, 0.0)
    assert np.allclose(y, 0.0)


def test_scale_effects_detail_betas_applies_one_scalar_to_persisted_columns() -> None:
    frame = np.array(
        [
            ("v1", "EUR", 0.2, 0.05, 0.25, 1.4),
            ("v1", "AFR", 0.2, -0.03, 0.17, 1.4),
        ],
        dtype=[
            ("variant_id", "U8"),
            ("population", "U8"),
            ("beta_base", "f8"),
            ("beta_delta", "f8"),
            ("beta", "f8"),
            ("effect_scale", "f8"),
        ],
    )
    import pandas as pd

    scaled = _scale_effects_detail_betas(pd.DataFrame(frame), 0.5)

    assert np.allclose(scaled["beta_base"].to_numpy(dtype=float), [0.1, 0.1])
    assert np.allclose(scaled["beta_delta"].to_numpy(dtype=float), [0.025, -0.015])
    assert np.allclose(scaled["beta"].to_numpy(dtype=float), [0.125, 0.085])
    assert np.allclose(scaled["effect_scale"].to_numpy(dtype=float), [1.4, 1.4])


def test_component_variance_diagnostics_flags_material_covariance() -> None:
    fixed = np.asarray([-1.0, 1.0, -1.0, 1.0])
    background = np.zeros(4)
    covariate = fixed.copy()
    residual = np.asarray([-1.0, -1.0, 1.0, 1.0])
    liability = fixed + background + covariate + residual

    diagnostics = _component_variance_diagnostics(
        genetic_fixed=fixed,
        genetic_bg=background,
        covariate=covariate,
        residual=residual,
        liability=liability,
    )

    assert diagnostics["pairwise_covariances"]["genetic_fixed|covariate"] == pytest.approx(
        1.0
    )
    assert diagnostics["covariance_contribution"] == pytest.approx(2.0)
    assert diagnostics["covariance_warning"] is True
    assert diagnostics["realized_fractions"]["genetic_fixed"] == pytest.approx(0.2)


def test_background_grm_factor_excludes_causals_and_samples_implicit_grm() -> None:
    retained, excluded = _exclude_causal_grm_variants(
        configured_variant_ids=["chr1:10", "chr1:20", "chr1:30"],
        causal_variant_ids=["chr1:20"],
    )
    assert retained == ["chr1:10", "chr1:30"]
    assert excluded == 1

    genotypes = np.asarray(
        [
            [0, 0, 1],
            [1, 0, 1],
            [2, 0, 0],
        ],
        dtype=np.uint8,
    )
    factor = _standardize_grm_genotypes(genotypes)
    assert factor.shape == (3, 2)
    assert np.allclose(factor.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(factor.std(axis=0), 1.0, atol=1e-6)

    sampled = _sample_background_from_grm_factor(factor, seed=17)
    rng = np.random.default_rng(17)
    expected_effects = rng.normal(
        0.0,
        1.0 / np.sqrt(float(factor.shape[1])),
        size=factor.shape[1],
    ).astype(np.float32)
    assert np.allclose(sampled, factor @ expected_effects)


def test_run_simulation_compute_grm_uses_separate_noncausal_markers(tmp_path: Path) -> None:
    sample_ids = ["s1", "s2", "s3", "s4"]
    sample_file = tmp_path / "samples.txt"
    sample_file.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    population_file = tmp_path / "populations.tsv"
    population_file.write_text(
        "IID\tpopulation\n" + "".join(f"{sample}\tEUR\n" for sample in sample_ids),
        encoding="utf-8",
    )
    covariate_file = tmp_path / "covariates.tsv"
    covariate_file.write_text(
        "IID\tage\ns1\t20\ns2\t30\ns3\t40\ns4\t50\n",
        encoding="utf-8",
    )
    coefficient_file = tmp_path / "coefficients.tsv"
    coefficient_file.write_text("covariate\tbeta\nage\t0\n", encoding="utf-8")
    source_pairs_file = tmp_path / "source_pairs.tsv"
    source_pairs_file.write_text(
        "causal_id\tmaf_eur\nchr1:10\t0.2\n",
        encoding="utf-8",
    )
    grm_variant_file = tmp_path / "grm_variants.txt"
    grm_variant_file.write_text(
        "chr1:10\nchr1:20\nchr1:30\n",
        encoding="utf-8",
    )

    payload = {
        "name": "grm_integration",
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
                "sample_ids_file": str(sample_file),
                "population_map_file": str(population_file),
            }
        ],
        "architectures": [
            {
                "name": "a1",
                "plugin": "polygenic_no_pleiotropy",
                "params": {"sigma_beta": 1.0, "h2": 0.3},
            }
        ],
        "causal_selection": {
            "source_pairs_file": str(source_pairs_file),
            "variant_id_column": "causal_id",
            "maf_columns_by_population": {"EUR": "maf_eur"},
            "anchor_population": "EUR",
            "maf_buckets": [
                {
                    "name": "common",
                    "maf_min": 0.1,
                    "maf_max": 0.3,
                    "fixed_count": 1,
                    "subset_count": 0,
                }
            ],
        },
        "effect_model": {
            "maf_scaling_mode": "none",
            "population_order": ["EUR"],
            "population_corr_matrix": [[1.0]],
            "jitter_sd_by_population": {"EUR": 0.0},
        },
        "covariates": {
            "effects_mode": "fixed",
            "covariate_file": str(covariate_file),
            "fixed_effects_file": str(coefficient_file),
        },
        "phenotype": {"continuous_only": True, "n_replicates": 1},
        "variance_budget": {
            "h2_fixed": 0.3,
            "h2_bg": 0.2,
            "covariate": 0.0,
            "residual": 0.5,
            "strict_sum_to_one": True,
        },
        "background_genetic": {
            "mode": "compute_grm",
            "variant_ids_file": str(grm_variant_file),
            "seed": 11,
        },
        "gwas": {"global_pcs": {"mode": "none"}},
        "paths": {"output_root": str(tmp_path / "outputs")},
        "chromosomes": [1],
    }

    genotype_by_variant = {
        "chr1:10": np.asarray([0, 1, 1, 2], dtype=np.uint8),
        "chr1:20": np.asarray([0, 0, 1, 2], dtype=np.uint8),
        "chr1:30": np.asarray([0, 1, 2, 2], dtype=np.uint8),
    }

    class _Adapter:
        def __init__(self) -> None:
            self.last_export_allele_by_variant: dict[str, str] = {}
            self.last_export_other_allele_by_variant: dict[str, str] = {}
            self.export_calls: list[list[str]] = []

        def export_genotypes_u8(self, **kwargs) -> np.ndarray:
            variant_ids = list(kwargs["variant_ids"])
            self.export_calls.append(variant_ids)
            self.last_export_allele_by_variant = {
                variant_id: "A" for variant_id in variant_ids
            }
            self.last_export_other_allele_by_variant = {
                variant_id: "G" for variant_id in variant_ids
            }
            return np.column_stack(
                [genotype_by_variant[variant_id] for variant_id in variant_ids]
            )

    adapter = _Adapter()
    records = run_simulation(
        run_cfg=RunConfig.from_dict(payload),
        run_dir=tmp_path / "run",
        adapter=adapter,
    )

    assert adapter.export_calls == [["chr1:10"], ["chr1:20", "chr1:30"]]
    assert len(records) == 1
    assert records[0]["background_grm_variant_count"] == 2
    assert records[0]["background_grm_excluded_causal_count"] == 1
    assert records[0]["component_realized"]["genetic_bg"] == pytest.approx(0.2)


def test_resolve_cohort_component_targets_uses_architecture_h2() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.25,
            h2_bg=0.10,
            covariate=0.35,
            residual=0.30,
            strict_sum_to_one=False,
        ),
        covariates=SimpleNamespace(target_variance=None),
    )

    targets = _resolve_cohort_component_targets(
        run_cfg=run_cfg,
        architecture_params={"h2": 0.30},
    )
    assert targets == {
        "target_h2_fixed": 0.30,
        "genetic_fixed": 0.30,
        "genetic_bg": 0.10,
        "genetic_total": 0.40,
        "covariate": 0.35,
        "residual": pytest.approx(0.25),
    }


def test_resolve_cohort_component_targets_allows_covariate_override() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.25,
            h2_bg=0.05,
            covariate=0.35,
            residual=0.35,
            strict_sum_to_one=True,
        ),
        covariates=SimpleNamespace(target_variance=0.10),
    )

    targets = _resolve_cohort_component_targets(
        run_cfg=run_cfg,
        architecture_params={"h2": 0.30},
    )
    assert targets["target_h2_fixed"] == 0.30
    assert abs(targets["genetic_bg"] - 0.05) < 1e-10
    assert abs(targets["covariate"] - 0.10) < 1e-10
    assert abs(targets["residual"] - 0.55) < 1e-10


def test_resolve_cohort_component_targets_rejects_impossible_budget() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.70,
            h2_bg=0.20,
            covariate=0.20,
            residual=0.0,
            strict_sum_to_one=True,
        ),
        covariates=SimpleNamespace(target_variance=None),
    )

    with pytest.raises(ValueError, match="exceed 1.0"):
        _resolve_cohort_component_targets(
            run_cfg=run_cfg,
            architecture_params={},
        )


def test_logistic_intercept_calibration_hits_target_prevalence() -> None:
    liability = np.linspace(-3.0, 3.0, 1001)
    z = (liability - liability.mean()) / liability.std()

    alpha = _calibrate_logistic_intercept(
        standardized_liability=z,
        prevalence=0.2,
        liability_logit_scale=1.25,
    )
    probabilities = 1.0 / (1.0 + np.exp(-(alpha + 1.25 * z)))

    assert float(probabilities.mean()) == pytest.approx(0.2, abs=1e-10)


def test_binary_phenotype_from_liability_supports_population_prevalence() -> None:
    liability = np.linspace(-2.0, 2.0, 200)
    sample_populations = ["EUR"] * 100 + ["AFR"] * 100
    rng = np.random.default_rng(17)

    labels, probabilities, alpha_values, summary = _binary_phenotype_from_liability(
        liability=liability,
        sample_populations=sample_populations,
        prevalence=0.1,
        liability_logit_scale=1.0,
        prevalence_by_population={"AFR": 0.3},
        alpha_by_population={},
        rng=rng,
    )

    assert set(labels.tolist()) <= {0, 1}
    assert probabilities[:100].mean() == pytest.approx(0.1, abs=1e-10)
    assert probabilities[100:].mean() == pytest.approx(0.3, abs=1e-10)
    assert np.all(np.isfinite(alpha_values))
    assert summary["binary_case_count"] + summary["binary_control_count"] == 200
    assert '"AFR"' in summary["binary_alpha_by_population"]
    assert '"EUR"' in summary["binary_alpha_by_population"]
    assert summary["binary_alpha_mode"] == "population_specific"


def test_binary_phenotype_uses_source_weighted_alpha_for_admix() -> None:
    rng = np.random.default_rng(23)
    n_per_group = 2000
    liability = np.concatenate(
        [
            rng.normal(loc=-0.2, scale=1.0, size=n_per_group),
            rng.normal(loc=0.2, scale=1.0, size=n_per_group),
            rng.normal(loc=0.0, scale=1.0, size=n_per_group),
        ]
    )
    sample_populations = (
        ["EUR"] * n_per_group
        + ["AFR"] * n_per_group
        + ["ADMIX"] * n_per_group
    )
    admix_weights = {
        row: {"EUR": 0.75, "AFR": 0.25}
        for row in range(2 * n_per_group, 3 * n_per_group)
    }

    labels, probabilities, alpha_values, summary = _binary_phenotype_from_liability(
        liability=liability,
        sample_populations=sample_populations,
        prevalence=0.2,
        liability_logit_scale=1.0,
        prevalence_by_population={"EUR": 0.1, "AFR": 0.3},
        alpha_by_population={},
        admix_alpha_weights_by_row=admix_weights,
        rng=np.random.default_rng(101),
    )

    source_alphas = json.loads(summary["binary_alpha_by_population"])
    expected_admix_alpha = 0.75 * source_alphas["EUR"] + 0.25 * source_alphas["AFR"]
    assert probabilities[:n_per_group].mean() == pytest.approx(0.1, abs=1e-10)
    assert probabilities[n_per_group : 2 * n_per_group].mean() == pytest.approx(0.3, abs=1e-10)
    assert np.allclose(alpha_values[2 * n_per_group :], expected_admix_alpha)
    admix_mean_probability = probabilities[2 * n_per_group :].mean()
    admix_realized_prevalence = labels[2 * n_per_group :].mean()
    assert 0.1 < admix_mean_probability < 0.3
    assert admix_realized_prevalence == pytest.approx(admix_mean_probability, abs=0.04)
    assert summary["binary_admix_alpha_mode"] == "causal_architecture_weighted_source_alpha"
    assert summary["binary_admix_sample_count"] == n_per_group


def test_binary_phenotype_supports_explicit_source_alpha_for_admix_only() -> None:
    liability = np.linspace(-1.0, 1.0, 10)
    sample_populations = ["ADMIX"] * 10
    admix_weights = {row: {"EUR": 0.6, "AFR": 0.4} for row in range(10)}

    _, probabilities, alpha_values, summary = _binary_phenotype_from_liability(
        liability=liability,
        sample_populations=sample_populations,
        prevalence=0.2,
        liability_logit_scale=1.0,
        prevalence_by_population={},
        alpha_by_population={"EUR": -2.0, "AFR": -0.5},
        admix_alpha_weights_by_row=admix_weights,
        rng=np.random.default_rng(19),
    )

    expected_alpha = 0.6 * -2.0 + 0.4 * -0.5
    assert np.allclose(alpha_values, expected_alpha)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert summary["binary_alpha_by_population"] == '{"AFR": -0.5, "EUR": -2.0}'
    assert summary["binary_admix_alpha_mode"] == "causal_architecture_weighted_source_alpha"


def test_admix_alpha_weights_use_squared_log_odds_architecture_not_alleles() -> None:
    codes = np.asarray(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [0, 1]],
        ],
        dtype=np.int16,
    )
    effects_detail = pd.DataFrame(
        [
            {"variant_id": "v1", "population": "EUR", "beta": 2.0},
            {"variant_id": "v1", "population": "AFR", "beta": 1.0},
            {"variant_id": "v2", "population": "EUR", "beta": 1.0},
            {"variant_id": "v2", "population": "AFR", "beta": 3.0},
        ]
    )

    weights = _admix_alpha_weights_from_local_ancestry(
        admix_row_indices=[10, 11],
        admix_local_ancestry_codes=codes,
        ancestry_order=["EUR", "AFR"],
        effects_detail=effects_detail,
        variant_ids=["v1", "v2"],
        liability=np.asarray([0.0, 1.0, 2.0, 3.0]),
        liability_logit_scale=1.5,
    )

    # Sample 10 exposure: EUR gets v1_EUR^2=4; AFR gets v1_AFR^2 + 2*v2_AFR^2 = 19.
    assert weights[10]["EUR"] == pytest.approx(4.0 / 23.0)
    assert weights[10]["AFR"] == pytest.approx(19.0 / 23.0)
    # Sample 11 exposure: EUR gets 2*v1_EUR^2 + v2_EUR^2 = 9; AFR gets v2_AFR^2 = 9.
    assert weights[11]["EUR"] == pytest.approx(0.5)
    assert weights[11]["AFR"] == pytest.approx(0.5)


def test_multi_population_cohort_uses_one_pooled_beta_scale() -> None:
    # 2 variants, 4 EUR + 4 AFR samples. EUR has lower per-variant variance.
    geno_eur = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    geno_afr = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    beta_eur = np.array([0.5, -0.2])
    beta_afr = np.array([0.5, -0.2])
    fixed_raw = np.concatenate([geno_eur @ beta_eur, geno_afr @ beta_afr])
    scale_factor = np.sqrt(0.30 / np.var(fixed_raw))
    fixed = fixed_raw * scale_factor

    var_eur = float(np.var(fixed[:4]))
    var_afr = float(np.var(fixed[4:]))

    assert np.var(fixed) == pytest.approx(0.30)
    assert var_eur > 0.0
    assert var_afr > 0.0
    assert var_afr > 2.0 * var_eur
