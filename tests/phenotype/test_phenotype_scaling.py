"""Phenotype component scaling tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from forgephast.phenotype.phenotypes.simulate import (
    _resolve_cohort_component_targets,
    _resolve_population_h2_targets,
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
    )
    assert abs(targets["base_h2_fixed"] - 0.30) < 1e-10
    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.30, "AFR": 0.30}
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
    )
    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.30, "AFR": 0.30}
    assert abs(targets["genetic_bg"] - 0.05) < 1e-10
    assert abs(targets["covariate"] - 0.10) < 1e-10


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

    t1 = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1)
    t2 = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1)
    t3 = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=2)

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

    targets = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1)
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

    targets = _resolve_population_h2_targets(run_cfg, "arch1", architecture_params={}, replicate=1)
    assert targets["resolved_h2_fixed_by_population"] == {"EUR": 0.80, "AFR": 0.80}
    assert targets["h2_jitter_clipped_by_population"] == {"EUR": True, "AFR": True}


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
        realized_genetic_fixed_variance=0.28,
    )

    assert targets["target_mode"] == "single_population"
    assert abs(targets["genetic_fixed"] - 0.28) < 1e-10
    assert abs(targets["residual"] - 0.47) < 1e-10
    assert json.loads(targets["resolved_h2_fixed_by_population_json"]) == {"AFR": 0.33, "EUR": 0.28}


def test_resolve_cohort_component_targets_inherited_source_betas_for_admixed() -> None:
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
        realized_genetic_fixed_variance=0.31,
    )

    assert targets["target_mode"] == "inherited_source_betas"
    assert np.isnan(targets["genetic_fixed"])
    assert np.isnan(targets["resolved_h2_fixed"])
    assert abs(targets["residual"] - 0.44) < 1e-10
