"""Phenotype component scaling and realized-variance diagnostics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..config import RunConfig

_COMPONENT_NAMES = ("genetic_fixed", "genetic_bg", "covariate", "residual")
_COVARIANCE_WARNING_RELATIVE_THRESHOLD = 0.05


def _scale_component_to_variance(
    component: np.ndarray,
    target_variance: float,
) -> np.ndarray:
    """Scale a component vector to desired variance; returns zeros when impossible."""
    if target_variance <= 0:
        return np.zeros_like(component)
    observed = float(np.var(component))
    if observed <= 0:
        return np.zeros_like(component)
    return component * np.sqrt(target_variance / observed)


def _scale_factor_to_variance(component: np.ndarray, target_variance: float) -> float:
    """Return multiplicative scale factor applied by _scale_component_to_variance."""
    if target_variance <= 0:
        return 0.0
    observed = float(np.var(component))
    if observed <= 0:
        return 0.0
    return float(np.sqrt(float(target_variance) / observed))


def _scale_effects_detail_betas(
    effects_detail: pd.DataFrame,
    scale_factor: float,
) -> pd.DataFrame:
    """Apply one pooled post-h2 scale factor to persisted beta columns."""
    scaled = effects_detail.copy()
    for column in ("beta_base", "beta_delta", "beta"):
        if column not in scaled.columns:
            continue
        scaled[column] = pd.to_numeric(scaled[column], errors="coerce") * float(
            scale_factor
        )
    return scaled


def _component_variance_diagnostics(
    *,
    genetic_fixed: np.ndarray,
    genetic_bg: np.ndarray,
    covariate: np.ndarray,
    residual: np.ndarray,
    liability: np.ndarray,
) -> Dict[str, object]:
    """Summarize realized component covariance without changing the simulation."""
    components = {
        "genetic_fixed": np.asarray(genetic_fixed, dtype=float),
        "genetic_bg": np.asarray(genetic_bg, dtype=float),
        "covariate": np.asarray(covariate, dtype=float),
        "residual": np.asarray(residual, dtype=float),
    }
    lengths = {values.shape for values in components.values()}
    if len(lengths) != 1 or any(
        values.ndim != 1 for values in components.values()
    ):
        raise ValueError(
            "Phenotype components must be one-dimensional vectors with equal length"
        )

    n_samples = next(iter(components.values())).shape[0]
    centered = np.column_stack(
        [
            components[name] - float(np.mean(components[name]))
            for name in _COMPONENT_NAMES
        ]
    )
    covariance_matrix = (
        (centered.T @ centered) / float(n_samples)
        if n_samples
        else np.zeros(
            (len(_COMPONENT_NAMES), len(_COMPONENT_NAMES)),
            dtype=float,
        )
    )
    liability_variance = float(np.var(np.asarray(liability, dtype=float)))
    component_variance_sum = float(np.trace(covariance_matrix))
    covariance_contribution = float(liability_variance - component_variance_sum)
    relative_contribution = (
        abs(covariance_contribution) / component_variance_sum
        if component_variance_sum > 0.0
        else 0.0
    )

    pairwise_covariances = {
        f"{left}|{right}": float(covariance_matrix[left_idx, right_idx])
        for left_idx, left in enumerate(_COMPONENT_NAMES)
        for right_idx, right in enumerate(_COMPONENT_NAMES)
        if right_idx > left_idx
    }
    realized_fractions = {
        name: (
            float(covariance_matrix[idx, idx]) / liability_variance
            if liability_variance > 0.0
            else float("nan")
        )
        for idx, name in enumerate(_COMPONENT_NAMES)
    }
    realized_fractions["genetic_total"] = (
        float(np.var(components["genetic_fixed"] + components["genetic_bg"]))
        / liability_variance
        if liability_variance > 0.0
        else float("nan")
    )

    return {
        "pairwise_covariances": pairwise_covariances,
        "component_variance_sum": component_variance_sum,
        "covariance_contribution": covariance_contribution,
        "covariance_relative_contribution": float(relative_contribution),
        "covariance_warning_threshold": _COVARIANCE_WARNING_RELATIVE_THRESHOLD,
        "covariance_warning": bool(
            relative_contribution > _COVARIANCE_WARNING_RELATIVE_THRESHOLD
        ),
        "realized_fractions": realized_fractions,
    }


def _resolve_cohort_component_targets(
    run_cfg: RunConfig,
    architecture_params: Dict[str, float],
) -> Dict[str, Any]:
    """Resolve one cohort-level variance budget for an architecture."""
    target_h2_fixed = float(run_cfg.variance_budget.h2_fixed)
    if "h2" in architecture_params:
        target_h2_fixed = float(architecture_params["h2"])
    target_h2_bg = float(run_cfg.variance_budget.h2_bg)
    target_covariate = float(
        run_cfg.covariates.target_variance
        if run_cfg.covariates.target_variance is not None
        else run_cfg.variance_budget.covariate
    )
    max_h2_fixed = 1.0 - target_h2_bg - target_covariate
    if not np.isfinite(target_h2_fixed) or target_h2_fixed < 0.0:
        raise ValueError(
            f"Cohort-level h2_fixed must be finite and non-negative; got {target_h2_fixed}"
        )
    if target_h2_fixed > max_h2_fixed + 1e-12:
        raise ValueError(
            "Cohort-level variance targets exceed 1.0: "
            f"h2_fixed={target_h2_fixed}, h2_bg={target_h2_bg}, "
            f"covariate={target_covariate}"
        )
    target_residual = max(
        0.0,
        1.0 - target_h2_fixed - target_h2_bg - target_covariate,
    )
    return {
        "target_h2_fixed": target_h2_fixed,
        "genetic_fixed": target_h2_fixed,
        "genetic_bg": target_h2_bg,
        "genetic_total": target_h2_fixed + target_h2_bg,
        "covariate": target_covariate,
        "residual": target_residual,
    }
