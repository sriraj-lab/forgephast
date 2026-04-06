"""Effect jitter covariance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgephast.phenotype.config import ArchitectureConfig, EffectModelConfig
from forgephast.phenotype.effects.betas import (
    _population_jitter_covariance,
    build_variant_scale_by_maf,
    resolve_effect_geometry,
    sample_population_specific_betas,
)


def test_correlated_population_betas_match_target_correlation() -> None:
    n_variants = 5000
    variant_ids = [f"v{i}" for i in range(n_variants)]

    arch = ArchitectureConfig(
        name="a",
        plugin="polygenic_no_pleiotropy",
        params={"sigma_beta": 1.0, "h2": 0.2},
    )
    effect_cfg = EffectModelConfig(
        population_order=["EUR", "AFR"],
        population_corr_matrix=[[1.0, 0.8], [0.8, 1.0]],
        jitter_sd_by_population={"EUR": 1.0, "AFR": 1.0},
    )
    effect_cfg.validate()

    df = sample_population_specific_betas(variant_ids, arch, effect_cfg, replicate_seed=123)
    pivot = df.pivot(index="variant_id", columns="population", values="beta")

    corr = np.corrcoef(pivot["EUR"], pivot["AFR"])[0, 1]
    assert 0.74 < corr < 0.86


def test_base_beta_sd_used_when_architecture_sigma_missing() -> None:
    variant_ids = [f"v{i}" for i in range(10000)]
    arch = ArchitectureConfig(
        name="a",
        plugin="polygenic_no_pleiotropy",
        params={"h2": 0.2},
    )
    effect_cfg = EffectModelConfig(
        base_beta_sd=2.0,
        population_order=["EUR"],
        population_corr_matrix=[[1.0]],
        jitter_sd_by_population={"EUR": 0.0},
    )
    effect_cfg.validate()

    df = sample_population_specific_betas(variant_ids, arch, effect_cfg, replicate_seed=7)
    observed_sd = float(df["beta"].std(ddof=0))
    assert 1.85 < observed_sd < 2.15


def test_student_t_base_beta_distribution_respects_target_sd() -> None:
    variant_ids = [f"v{i}" for i in range(20000)]
    arch = ArchitectureConfig(
        name="a",
        plugin="polygenic_no_pleiotropy",
        params={"h2": 0.2},
    )
    effect_cfg = EffectModelConfig(
        base_beta_distribution="student_t",
        student_t_df=6.0,
        base_beta_sd=2.0,
        population_order=["EUR"],
        population_corr_matrix=[[1.0]],
        jitter_sd_by_population={"EUR": 0.0},
    )
    effect_cfg.validate()

    df = sample_population_specific_betas(variant_ids, arch, effect_cfg, replicate_seed=11)
    observed_sd = float(df["beta"].std(ddof=0))
    assert 1.85 < observed_sd < 2.15


def test_mixture_normal_base_beta_distribution_respects_target_sd() -> None:
    variant_ids = [f"v{i}" for i in range(20000)]
    arch = ArchitectureConfig(
        name="a",
        plugin="polygenic_no_pleiotropy",
        params={"h2": 0.2},
    )
    effect_cfg = EffectModelConfig(
        base_beta_distribution="mixture_normal",
        mixture_slab_prob=0.1,
        mixture_slab_sd_ratio=10.0,
        base_beta_sd=2.0,
        population_order=["EUR"],
        population_corr_matrix=[[1.0]],
        jitter_sd_by_population={"EUR": 0.0},
    )
    effect_cfg.validate()

    df = sample_population_specific_betas(variant_ids, arch, effect_cfg, replicate_seed=13)
    observed_sd = float(df["beta"].std(ddof=0))
    assert 1.85 < observed_sd < 2.15


def test_maf_scaling_inverse_sqrt_2pq_gives_larger_scale_for_rare_variants() -> None:
    effect_cfg = EffectModelConfig(
        maf_scaling_mode="inverse_sqrt_2pq",
        population_order=["EUR"],
        population_corr_matrix=[[1.0]],
        jitter_sd_by_population={"EUR": 0.0},
    )
    effect_cfg.validate()

    membership = pd.DataFrame(
        {
            "variant_id": ["rare", "common"],
            "maf_anchor": [0.01, 0.49],
        }
    )
    scales = build_variant_scale_by_maf(
        membership_df=membership,
        variant_id_column="variant_id",
        maf_column="maf_anchor",
        effect_model=effect_cfg,
    )
    assert scales["rare"] > scales["common"]


def test_incompatible_corr_and_shared_base_raises() -> None:
    arch = ArchitectureConfig(
        name="a",
        plugin="polygenic_no_pleiotropy",
        params={"sigma_beta": 5.0, "h2": 0.2},
    )
    effect_cfg = EffectModelConfig(
        population_order=["EUR", "AFR"],
        population_corr_matrix=[[1.0, 0.0], [0.0, 1.0]],
        jitter_sd_by_population={"EUR": 0.1, "AFR": 0.1},
    )
    effect_cfg.validate()

    with pytest.raises(ValueError, match="incompatible"):
        sample_population_specific_betas(["v1", "v2"], arch, effect_cfg, replicate_seed=5)


def test_auto_fst_mode_resolves_psd_corr_and_base_jitter_floor(tmp_path) -> None:
    fst_path = tmp_path / "pairwise_fst.tsv"
    pd.DataFrame(
        {
            "population_i": ["EUR"],
            "population_j": ["AFR"],
            "fst": [0.05],
        }
    ).to_csv(fst_path, sep="\t", index=False)

    effect_cfg = EffectModelConfig(
        population_order=["EUR", "AFR"],
        population_corr_mode="auto_from_fst",
        population_fst_file=str(fst_path),
        base_jitter_sd=0.3,
        jitter_sd_by_population={"EUR": 0.5},
    )
    effect_cfg.validate()

    geometry = resolve_effect_geometry(effect_cfg)
    corr = np.asarray(geometry["population_corr_matrix"], dtype=float)

    assert geometry["resolved_jitter_sd_by_population"] == {"EUR": 0.5, "AFR": 0.3}
    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), np.ones(2))
    assert float(np.min(np.linalg.eigvalsh(corr))) >= -1e-8
    assert 0.0 < float(corr[0, 1]) < 1.0


def test_auto_fst_mode_rejects_missing_pairs(tmp_path) -> None:
    fst_path = tmp_path / "pairwise_fst.tsv"
    pd.DataFrame(
        {
            "population_i": ["EUR"],
            "population_j": ["AFR"],
            "fst": [0.05],
        }
    ).to_csv(fst_path, sep="\t", index=False)

    effect_cfg = EffectModelConfig(
        population_order=["EUR", "AFR", "EAS"],
        population_corr_mode="auto_from_fst",
        population_fst_file=str(fst_path),
        base_jitter_sd=0.5,
    )
    effect_cfg.validate()

    with pytest.raises(ValueError, match="missing pairwise FST entries"):
        resolve_effect_geometry(effect_cfg)


def test_population_jitter_covariance_repairs_tiny_negative_eigenvalue() -> None:
    corr = np.array(
        [
            [1.0, 0.9655667812123503, 0.908486463497592],
            [0.9655667812123503, 1.0, 0.9809430709143145],
            [0.908486463497592, 0.9809430709143145, 1.0],
        ],
        dtype=float,
    )
    cov = _population_jitter_covariance(
        ["EUR", "AFR", "ADMIX"],
        corr,
        {"EUR": 0.05, "AFR": 0.05, "ADMIX": 0.05},
        base_beta_sd=0.15,
    )
    assert float(np.min(np.linalg.eigvalsh(cov))) >= -1e-8
