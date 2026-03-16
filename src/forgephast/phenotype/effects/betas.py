"""Effect size generation with calibrated population-specific jitter."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ArchitectureConfig, EffectModelConfig


def _resolve_base_beta_sd(
    architecture: ArchitectureConfig,
    effect_model: EffectModelConfig,
) -> float:
    """Resolve base SD, allowing architecture override of global default."""
    if "sigma_beta" in architecture.params:
        return float(architecture.params["sigma_beta"])
    return float(effect_model.base_beta_sd)


def _sample_base_beta(
    rng: np.random.Generator,
    *,
    effect_model: EffectModelConfig,
    sd: float,
) -> float:
    """Sample a base beta with target marginal SD=sd under the configured distribution."""
    sd = float(sd)
    if sd <= 0:
        raise ValueError("Base beta sd must be > 0")

    dist = str(effect_model.base_beta_distribution)
    if dist == "normal":
        return float(rng.normal(0.0, sd))

    if dist == "laplace":
        # Laplace(0,b) has SD = sqrt(2) * b
        b = sd / np.sqrt(2.0)
        return float(rng.laplace(0.0, float(b)))

    if dist == "student_t":
        df = float(effect_model.student_t_df)
        # Standard t has Var = df/(df-2) for df>2.
        scale = sd / np.sqrt(df / (df - 2.0))
        return float(rng.standard_t(df) * scale)

    if dist == "mixture_normal":
        p = float(effect_model.mixture_slab_prob)
        r = float(effect_model.mixture_slab_sd_ratio)
        # mixture variance: (1-p)*s^2 + p*(r*s)^2 = s^2 * ((1-p)+p*r^2) = sd^2
        denom = (1.0 - p) + p * (r**2)
        s_small = sd / np.sqrt(denom)
        s_large = r * s_small
        use_slab = bool(rng.uniform(0.0, 1.0) < p)
        return float(rng.normal(0.0, float(s_large if use_slab else s_small)))

    raise ValueError(f"Unsupported effect_model.base_beta_distribution={dist!r}")


def build_variant_scale_by_maf(
    membership_df: pd.DataFrame,
    variant_id_column: str,
    maf_column: str,
    effect_model: EffectModelConfig,
) -> Dict[str, float]:
    """Return multiplicative per-variant effect scales from configured MAF weighting."""
    dedup = membership_df[[variant_id_column, maf_column]].drop_duplicates(
        subset=[variant_id_column],
        keep="first",
    ).copy()
    maf = pd.to_numeric(dedup[maf_column], errors="coerce")
    if maf.isna().any():
        raise ValueError(f"NaN MAF values encountered in {maf_column} while building effect scales")

    if effect_model.maf_scaling_mode == "none":
        scales = np.ones(len(dedup), dtype=float)
    else:
        clipped = maf.clip(
            lower=effect_model.maf_min_clip,
            upper=effect_model.maf_max_clip,
        ).to_numpy(dtype=float)
        allele_var = 2.0 * clipped * (1.0 - clipped)
        if effect_model.maf_scaling_mode == "inverse_sqrt_2pq":
            scales = 1.0 / np.sqrt(allele_var)
        elif effect_model.maf_scaling_mode == "power_2pq":
            scales = np.power(allele_var, effect_model.maf_power_alpha)
        else:
            raise ValueError(
                f"Unsupported effect_model.maf_scaling_mode={effect_model.maf_scaling_mode}"
            )

    return dict(zip(dedup[variant_id_column].astype(str), scales.astype(float)))


def _population_jitter_covariance(
    population_order: List[str],
    correlation: np.ndarray,
    jitter_sd_by_population: Dict[str, float],
    base_beta_sd: float,
) -> np.ndarray:
    """Build delta covariance to ensure configured total inter-pop beta correlation."""
    jitter_sds = np.array(
        [jitter_sd_by_population.get(pop, 0.0) for pop in population_order],
        dtype=float,
    )
    base_var = float(base_beta_sd**2)
    total_sds = np.sqrt(base_var + np.square(jitter_sds))
    total_cov = np.diag(total_sds) @ correlation @ np.diag(total_sds)

    # beta_pop = shared_base + delta_pop
    # => Cov(delta_i, delta_j) = Cov(beta_i, beta_j) - Var(shared_base)
    delta_cov = total_cov - (base_var * np.ones_like(total_cov))
    np.fill_diagonal(delta_cov, np.square(jitter_sds))
    delta_cov = 0.5 * (delta_cov + delta_cov.T)

    tol = 1e-8
    for i in range(len(jitter_sds)):
        for j in range(i + 1, len(jitter_sds)):
            bound = jitter_sds[i] * jitter_sds[j] + tol
            if abs(delta_cov[i, j]) > bound:
                raise ValueError(
                    "Configured population_corr_matrix is incompatible with shared base variance "
                    f"and jitter SDs for pair ({population_order[i]}, {population_order[j]}). "
                    "Reduce base_beta_sd/sigma_beta, increase jitter SDs, or set a higher correlation."
                )

    evals = np.linalg.eigvalsh(delta_cov)
    min_eval = float(np.min(evals))
    if min_eval < -1e-8:
        raise ValueError(
            "Derived jitter covariance is not positive semidefinite; check correlation matrix "
            f"and effect-scale settings (min eigenvalue={min_eval:.3e})."
        )
    return delta_cov


def sample_population_specific_betas(
    variant_ids: List[str],
    architecture: ArchitectureConfig,
    effect_model: EffectModelConfig,
    replicate_seed: int,
    variant_scale_by_id: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Sample per-variant per-population betas with calibrated total correlation."""
    rng = np.random.default_rng(replicate_seed)

    base_beta_sd = _resolve_base_beta_sd(architecture, effect_model)

    pop_order = effect_model.population_order
    corr = np.asarray(effect_model.population_corr_matrix, dtype=float)
    cov = _population_jitter_covariance(
        pop_order,
        corr,
        effect_model.jitter_sd_by_population,
        base_beta_sd=base_beta_sd,
    )

    scales = (
        {variant_id: 1.0 for variant_id in variant_ids}
        if variant_scale_by_id is None
        else variant_scale_by_id
    )
    missing_scales = [variant_id for variant_id in variant_ids if variant_id not in scales]
    if missing_scales:
        raise ValueError(
            f"Missing variant effect scales for IDs: {missing_scales[:5]} (n={len(missing_scales)})"
        )

    rows = []
    for variant_id in variant_ids:
        scale = float(scales[variant_id])
        if scale < 0:
            raise ValueError(f"Negative effect scale for variant {variant_id}: {scale}")

        scaled_base_sd = base_beta_sd * scale
        base = _sample_base_beta(rng, effect_model=effect_model, sd=scaled_base_sd)
        if len(pop_order) == 1:
            delta = float(rng.normal(0.0, np.sqrt(cov[0, 0]) * scale))
            deltas = np.array([delta], dtype=float)
        else:
            deltas = rng.multivariate_normal(
                mean=np.zeros(len(pop_order), dtype=float),
                cov=cov * (scale**2),
            )

        for pop_idx, pop in enumerate(pop_order):
            beta_delta = float(deltas[pop_idx])
            beta = base + beta_delta
            rows.append(
                {
                    "variant_id": variant_id,
                    "population": pop,
                    "beta_base": base,
                    "beta_delta": beta_delta,
                    "beta": float(beta),
                    "effect_scale": scale,
                }
            )

    return pd.DataFrame(rows)
