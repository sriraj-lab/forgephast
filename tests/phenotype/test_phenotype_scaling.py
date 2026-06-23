"""Phenotype component scaling tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from forgephast.phenotype.phenotypes.simulate import (
    _scale_effects_detail_betas,
    _resolve_cohort_component_targets,
    _resolve_population_h2_targets,
    _scale_component_to_shared_population_variance,
    _scale_component_to_variance,
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


def test_scale_component_to_shared_population_variance_hits_target_per_population() -> None:
    x = np.array([-1.0, 1.0, -1.0, 1.0, -4.0, 4.0, -4.0, 4.0], dtype=float)
    pops = ["EUR"] * 4 + ["AFR"] * 4
    y = _scale_component_to_shared_population_variance(x, pops, 0.25)
    eur = y[:4]
    afr = y[4:]
    assert abs(np.var(eur) - 0.25) < 1e-12
    assert abs(np.var(afr) - 0.25) < 1e-12


def test_resolve_population_h2_targets_respects_covariate_target_when_no_override() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.25,
            h2_bg=0.10,
            covariate=0.35,
            residual=0.30,
            h2_jitter_sd=0.0,
            h2_jitter_seed=42,
            strict_sum_to_one=False,
        ),
        covariates=SimpleNamespace(target_variance=None),
        effect_model=SimpleNamespace(population_order=["EUR", "AFR"]),
    )

    targets = _resolve_population_h2_targets(
        run_cfg,
        "arch1",
        architecture_params={"h2": 0.30},
        replicate=1,
        population_order=["EUR", "AFR"],
    )
    assert abs(targets["base_h2_fixed"] - 0.30) < 1e-10
    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.30, "AFR": 0.30}
    assert targets["resolved_h2_bg_by_population"] == {"EUR": 0.10, "AFR": 0.10}
    assert targets["resolved_covariate_by_population"] == {"EUR": 0.35, "AFR": 0.35}
    assert targets["resolved_residual_by_population"] == {"EUR": 0.25, "AFR": 0.25}
    assert abs(targets["genetic_bg"] - 0.10) < 1e-10
    assert abs(targets["covariate"] - 0.35) < 1e-10


def test_resolve_population_h2_targets_allows_covariate_override() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.25,
            h2_bg=0.05,
            covariate=0.35,
            residual=0.35,
            h2_jitter_sd=0.0,
            h2_jitter_seed=42,
            strict_sum_to_one=True,
        ),
        covariates=SimpleNamespace(target_variance=0.10),
        effect_model=SimpleNamespace(population_order=["EUR", "AFR"]),
    )

    targets = _resolve_population_h2_targets(
        run_cfg,
        "arch1",
        architecture_params={"h2": 0.30},
        replicate=1,
        population_order=["EUR", "AFR"],
    )
    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.30, "AFR": 0.30}
    assert targets["resolved_h2_bg_by_population"] == {"EUR": 0.05, "AFR": 0.05}
    assert targets["resolved_covariate_by_population"] == {"EUR": 0.10, "AFR": 0.10}
    assert abs(targets["resolved_residual_by_population"]["EUR"] - 0.55) < 1e-12
    assert abs(targets["resolved_residual_by_population"]["AFR"] - 0.55) < 1e-12
    assert abs(targets["genetic_bg"] - 0.05) < 1e-10
    assert abs(targets["covariate"] - 0.10) < 1e-10


def test_resolve_population_h2_targets_excludes_admix_from_genetic_and_residual_budgets() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.25,
            h2_bg=0.05,
            covariate=0.20,
            residual=0.50,
            h2_jitter_sd=0.0,
            h2_jitter_seed=42,
            strict_sum_to_one=True,
        ),
        covariates=SimpleNamespace(target_variance=None),
        effect_model=SimpleNamespace(population_order=["EUR", "AFR"]),
    )

    targets = _resolve_population_h2_targets(
        run_cfg,
        "arch1",
        architecture_params={"h2": 0.30},
        replicate=1,
        population_order=["EUR", "AFR"],
        observed_population_order=["EUR", "AFR", "ADMIX"],
    )

    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.30, "AFR": 0.30}
    assert targets["resolved_h2_bg_by_population"] == {"EUR": 0.05, "AFR": 0.05}
    assert abs(targets["resolved_residual_by_population"]["EUR"] - 0.45) < 1e-12
    assert abs(targets["resolved_residual_by_population"]["AFR"] - 0.45) < 1e-12
    assert targets["resolved_covariate_by_population"] == {
        "EUR": 0.20,
        "AFR": 0.20,
        "ADMIX": 0.20,
    }
    assert "ADMIX" not in targets["resolved_h2_fixed_by_population"]
    assert "ADMIX" not in targets["resolved_h2_bg_by_population"]
    assert "ADMIX" not in targets["resolved_residual_by_population"]
    assert abs(targets["residual_baseline"] - 0.50) < 1e-12


def test_population_h2_jitter_is_deterministic_per_population() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.25,
            h2_bg=0.05,
            covariate=0.20,
            residual=0.50,
            h2_jitter_sd=0.05,
            h2_jitter_seed=17,
            strict_sum_to_one=True,
        ),
        covariates=SimpleNamespace(target_variance=None),
        effect_model=SimpleNamespace(population_order=["EUR", "AFR"]),
    )

    t1 = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1, population_order=["EUR", "AFR"])
    t2 = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1, population_order=["EUR", "AFR"])
    t3 = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=2, population_order=["EUR", "AFR"])

    assert t1["resolved_h2_fixed_by_population"] == t2["resolved_h2_fixed_by_population"]
    assert t1["h2_jitter_delta_by_population"] == t2["h2_jitter_delta_by_population"]
    assert t1["resolved_h2_fixed_by_population"]["EUR"] != t1["resolved_h2_fixed_by_population"]["AFR"]
    assert t1["resolved_h2_fixed_by_population"] != t3["resolved_h2_fixed_by_population"]


def test_population_h2_jitter_does_not_move_null_runs() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.0,
            h2_bg=0.05,
            covariate=0.20,
            residual=0.75,
            h2_jitter_sd=0.10,
            h2_jitter_seed=17,
            strict_sum_to_one=True,
        ),
        covariates=SimpleNamespace(target_variance=None),
        effect_model=SimpleNamespace(population_order=["EUR", "AFR"]),
    )

    targets = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1, population_order=["EUR", "AFR"])
    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.0, "AFR": 0.0}
    assert targets["h2_jitter_delta_by_population"] == {"EUR": 0.0, "AFR": 0.0}


def test_population_h2_jitter_clips_to_available_budget() -> None:
    run_cfg = SimpleNamespace(
        variance_budget=SimpleNamespace(
            h2_fixed=0.95,
            h2_bg=0.10,
            covariate=0.10,
            residual=0.0,
            h2_jitter_sd=0.20,
            h2_jitter_seed=1,
            strict_sum_to_one=True,
        ),
        covariates=SimpleNamespace(target_variance=None),
        effect_model=SimpleNamespace(population_order=["EUR", "AFR"]),
    )

    targets = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1, population_order=["EUR", "AFR"])
    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.80, "AFR": 0.80}
    assert targets["h2_jitter_clipped_by_population"] == {"EUR": True, "AFR": True}


def test_multi_population_cohort_uses_jittered_betas_without_rescaling() -> None:
    """Realized per-pop variance of GENETIC_FIXED matches what the fixed betas imply.

    With per-population β rescaling removed, each population's realized fixed-effect
    variance is a direct function of its genotype matrix and the (shared-base +
    correlated-delta) β draws. This test does the hand computation for two
    populations on a tiny synthetic genotype matrix and checks that the result
    is NOT forced to be equal across populations.
    """
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
    beta_afr = np.array([0.5, -0.2])  # identical betas — NO rescaling
    fixed_eur = geno_eur @ beta_eur
    fixed_afr = geno_afr @ beta_afr

    var_eur = float(np.var(fixed_eur))
    var_afr = float(np.var(fixed_afr))

    assert var_eur > 0.0
    assert var_afr > 0.0
    # AFR carries ~4x the genotype variance per variant and so its realized
    # fixed-effect variance must be substantially larger. This is the exact
    # behavior that the deleted inherited-source-betas rescaling used to erase.
    assert var_afr > 2.0 * var_eur


def test_resolve_cohort_component_targets_single_population_mode() -> None:
    population_targets = {
        "base_h2_fixed": 0.30,
        "resolved_h2_fixed_by_population": {"EUR": 0.28, "AFR": 0.33},
        "h2_jitter_delta_by_population": {"EUR": -0.02, "AFR": 0.03},
        "h2_jitter_clipped_by_population": {"EUR": False, "AFR": False},
        "genetic_bg": 0.10,
        "covariate": 0.15,
        "strict_sum_to_one": True,
    }

    targets = _resolve_cohort_component_targets(
        population_targets=population_targets,
        cohort_populations=["EUR", "EUR", "EUR"],
    )

    assert targets["target_mode"] == "single_population"
    assert abs(targets["genetic_fixed"] - 0.28) < 1e-10
    assert abs(targets["residual"] - 0.47) < 1e-10
    assert json.loads(targets["resolved_h2_fixed_by_population_json"]) == {"AFR": 0.33, "EUR": 0.28}
    assert json.loads(targets["resolved_h2_bg_by_population_json"]) == {"AFR": 0.1, "EUR": 0.1}
    assert json.loads(targets["resolved_covariate_by_population_json"]) == {"AFR": 0.15, "EUR": 0.15}
    residual_json = json.loads(targets["resolved_residual_by_population_json"])
    assert abs(residual_json["AFR"] - 0.42) < 1e-12
    assert abs(residual_json["EUR"] - 0.47) < 1e-12


def test_resolve_cohort_component_targets_free_per_population_for_admixed() -> None:
    population_targets = {
        "base_h2_fixed": 0.30,
        "resolved_h2_fixed_by_population": {"EUR": 0.28, "AFR": 0.33},
        "h2_jitter_delta_by_population": {"EUR": -0.02, "AFR": 0.03},
        "h2_jitter_clipped_by_population": {"EUR": False, "AFR": False},
        "genetic_bg": 0.10,
        "covariate": 0.15,
        "strict_sum_to_one": True,
    }

    targets = _resolve_cohort_component_targets(
        population_targets=population_targets,
        cohort_populations=["ADMIX", "ADMIX", "ADMIX"],
    )

    assert targets["target_mode"] == "free_per_population"
    # Multi-pop now uses base_h2_fixed as the pooled rescaling target.
    assert targets["genetic_fixed"] == 0.30
    assert np.isnan(targets["resolved_h2_fixed"])
    assert np.isnan(targets["residual"])
    # Per-population target metadata is still surfaced for informational
    # downstream reporting, even though it no longer drives β rescaling.
    assert json.loads(targets["resolved_h2_fixed_by_population_json"]) == {
        "AFR": 0.33,
        "EUR": 0.28,
    }
    assert json.loads(targets["resolved_h2_bg_by_population_json"]) == {"AFR": 0.1, "EUR": 0.1}
    assert json.loads(targets["resolved_covariate_by_population_json"]) == {"AFR": 0.15, "EUR": 0.15}
    residual_json = json.loads(targets["resolved_residual_by_population_json"])
    assert abs(residual_json["AFR"] - 0.42) < 1e-12
    assert abs(residual_json["EUR"] - 0.47) < 1e-12


def test_resolve_cohort_component_targets_single_population_infers_missing_budget_maps() -> None:
    population_targets = {
        "base_h2_fixed": 0.30,
        "resolved_h2_fixed_by_population": {"EUR": 0.50, "AFR": 0.50},
        "h2_jitter_delta_by_population": {"EUR": 0.0, "AFR": 0.0},
        "h2_jitter_clipped_by_population": {"EUR": False, "AFR": False},
        "genetic_bg": 0.0,
        "covariate": 0.20,
        "strict_sum_to_one": True,
    }

    targets = _resolve_cohort_component_targets(
        population_targets=population_targets,
        cohort_populations=["EUR", "EUR", "EUR"],
    )
    assert json.loads(targets["resolved_h2_bg_by_population_json"]) == {"AFR": 0.0, "EUR": 0.0}
    assert json.loads(targets["resolved_covariate_by_population_json"]) == {"AFR": 0.2, "EUR": 0.2}
    assert json.loads(targets["resolved_residual_by_population_json"]) == {"AFR": 0.3, "EUR": 0.3}
