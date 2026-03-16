"""Effect jitter covariance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgephast.phenotype.config import ArchitectureConfig, EffectModelConfig
from forgephast.phenotype.effects.betas import build_variant_scale_by_maf, sample_population_specific_betas


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
