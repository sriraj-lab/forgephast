"""Covariate loading and fixed/random/hybrid effect modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids, read_sample_ids

from ..config import CovariateConfig


@dataclass
class CovariateEffectOutput:
    """Covariate effect outputs for one cohort replicate."""

    covariate_table: pd.DataFrame
    contribution: np.ndarray
    used_coefficients: pd.DataFrame


def _load_covariates(cfg: CovariateConfig, sample_ids: List[str]) -> pd.DataFrame:
    cov = pd.read_csv(cfg.covariate_file, sep="\t")
    cov = align_by_ids(cov, cfg.id_column, sample_ids, "covariate alignment")
    cov = cov.rename(columns={cfg.id_column: "IID"})
    if "FID" not in cov.columns:
        cov.insert(0, "FID", cov["IID"])
    else:
        cov["FID"] = cov["IID"]
        ordered = ["FID", "IID"] + [c for c in cov.columns if c not in {"FID", "IID"}]
        cov = cov.loc[:, ordered]
    return cov


def _load_fixed_coefficients(cfg: CovariateConfig) -> pd.DataFrame:
    if not cfg.fixed_effects_file:
        raise ValueError("fixed effects file is required but missing")
    coeff = pd.read_csv(cfg.fixed_effects_file, sep="\t")
    required = {"covariate", "beta"}
    if not required.issubset(coeff.columns):
        raise ValueError(f"fixed effects file must contain {required}")
    coeff = coeff[["covariate", "beta"]].copy()
    coeff["beta"] = pd.to_numeric(coeff["beta"], errors="raise")
    return coeff


def build_covariate_effects(
    cfg: CovariateConfig,
    sample_ids: List[str],
    replicate_seed: int,
    runtime_cache: dict[str, Any] | None = None,
) -> CovariateEffectOutput:
    """Build covariate table and modeled contribution vector."""
    def _bump(name: str, hit: bool) -> None:
        if runtime_cache is None:
            return
        bucket_name = "cache_hits" if hit else "cache_misses"
        counters = runtime_cache.setdefault(bucket_name, {})
        counters[name] = int(counters.get(name, 0)) + 1

    cov_df: pd.DataFrame
    cov_cache = None
    if runtime_cache is not None:
        cov_cache = runtime_cache.setdefault("simulate_covariate_table_cache", {})
    cov_key = (cfg.covariate_file, cfg.id_column, tuple(sample_ids))
    if cov_cache is not None and cov_key in cov_cache:
        _bump("covariate_table", True)
        cov_df = cov_cache[cov_key].copy()
    else:
        _bump("covariate_table", False)
        cov_df = _load_covariates(cfg, sample_ids)
        if cov_cache is not None:
            cov_cache[cov_key] = cov_df.copy()

    feature_cols = [c for c in cov_df.columns if c not in {"FID", "IID"}]
    if not feature_cols:
        raise ValueError("covariate file has no feature columns")

    x = cov_df[feature_cols].to_numpy(dtype=float)
    rng = np.random.default_rng(replicate_seed)

    coeff_rows = []
    contribution = np.zeros(x.shape[0], dtype=float)

    if cfg.effects_mode in {"fixed", "hybrid"}:
        fixed_cache = None
        if runtime_cache is not None:
            fixed_cache = runtime_cache.setdefault("simulate_fixed_coeff_cache", {})
        fixed_key = str(cfg.fixed_effects_file)
        if fixed_cache is not None and fixed_key in fixed_cache:
            _bump("fixed_coefficients", True)
            fixed_df = fixed_cache[fixed_key].copy()
        else:
            _bump("fixed_coefficients", False)
            fixed_df = _load_fixed_coefficients(cfg)
            if fixed_cache is not None:
                fixed_cache[fixed_key] = fixed_df.copy()
        fixed_map = dict(zip(fixed_df["covariate"], fixed_df["beta"]))
        fixed_betas = np.array([fixed_map.get(col, 0.0) for col in feature_cols], dtype=float)
        contribution += x @ fixed_betas
        for col, beta in zip(feature_cols, fixed_betas):
            coeff_rows.append({"covariate": col, "beta": float(beta), "effect_type": "fixed"})

    if cfg.effects_mode in {"random", "hybrid"}:
        random_betas = rng.normal(0.0, cfg.random_effect_sd, size=len(feature_cols))
        contribution += x @ random_betas
        for col, beta in zip(feature_cols, random_betas):
            coeff_rows.append({"covariate": col, "beta": float(beta), "effect_type": "random"})

    coeff_df = pd.DataFrame(coeff_rows)
    return CovariateEffectOutput(covariate_table=cov_df, contribution=contribution, used_coefficients=coeff_df)
