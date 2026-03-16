"""Phenotype component scaling tests."""

from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from forgephast.phenotype.phenotypes.simulate import _resolve_component_targets, _scale_component_to_variance


def test_scale_component_to_variance_hits_target() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = _scale_component_to_variance(x, 0.25)
    assert abs(np.var(y) - 0.25) < 0.02


def test_scale_component_zero_target() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = _scale_component_to_variance(x, 0.0)
    assert np.allclose(y, 0.0)


def test_resolve_component_targets_respects_phenotype_covariate_target_when_no_override() -> None:
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

    targets = _resolve_component_targets(run_cfg, "arch1", architecture_params={"h2": 0.30})
    assert abs(targets["genetic_fixed"] - 0.30) < 1e-10
    assert abs(targets["genetic_bg"] - 0.10) < 1e-10
    assert abs(targets["genetic_total"] - 0.40) < 1e-10
    assert abs(targets["covariate"] - 0.35) < 1e-10
    assert abs(targets["residual"] - 0.25) < 1e-10


def test_resolve_component_targets_allows_covariate_override() -> None:
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

    targets = _resolve_component_targets(run_cfg, "arch1", architecture_params={"h2": 0.30})
    assert abs(targets["genetic_fixed"] - 0.30) < 1e-10
    assert abs(targets["genetic_bg"] - 0.05) < 1e-10
    assert abs(targets["covariate"] - 0.10) < 1e-10
    assert abs(targets["residual"] - 0.55) < 1e-10
