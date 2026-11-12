"""Pleiotropic trait clusters: shared causal variants with correlated effects.

See ``PleiotropyClusterConfig`` for the model. Everything here is a pure function
of the unclustered draws, so a cluster with ``shared_fraction=1`` and an identity
``effect_correlation`` reproduces the independent architectures exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config_schema import PleiotropyClusterConfig

BETA_COLUMNS = ("beta_base", "beta_delta", "beta")
SHARED = "shared"
PRIVATE = "private"


def lower_factor(matrix: Sequence[Sequence[float]], tol: float = 1e-10) -> np.ndarray:
    """Lower-triangular L with L @ L.T == matrix, for positive semidefinite input.

    Unlike ``np.linalg.cholesky`` this accepts singular matrices (for example an
    effect correlation of exactly 1), keeping row 0 equal to ``[1, 0, ...]``.
    """
    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    out = np.zeros_like(a)
    for j in range(n):
        d = a[j, j] - float(out[j, :j] @ out[j, :j])
        if d < -tol:
            raise ValueError("correlation matrix is not positive semidefinite")
        out[j, j] = np.sqrt(max(d, 0.0))
        for i in range(j + 1, n):
            r = a[i, j] - float(out[i, :j] @ out[j, :j])
            if out[j, j] > tol:
                out[i, j] = r / out[j, j]
            elif abs(r) > 1e-8:
                raise ValueError("correlation matrix is not positive semidefinite")
    return out


@dataclass(frozen=True)
class ClusterRoles:
    """Per replicate: each trait's causal mask over the run's causal variants."""

    cluster: str
    variant_ids: tuple[str, ...]
    shared: np.ndarray                 # bool, len(variant_ids)
    private_owner: np.ndarray          # int trait index, -1 for shared
    traits: tuple[str, ...]

    def causal_mask(self, trait: str) -> np.ndarray:
        k = self.traits.index(trait)
        return self.shared | (self.private_owner == k)

    def role_of(self, trait: str) -> np.ndarray:
        """'shared' or 'private' for each variant causal for the trait (in variant order)."""
        mask = self.causal_mask(trait)
        return np.where(self.shared[mask], SHARED, PRIVATE)


def cluster_roles(
    variant_ids: Sequence[str],
    cluster: PleiotropyClusterConfig,
    replicate: int,
) -> ClusterRoles:
    """Split the replicate causal set into the shared part and one private part per trait."""
    ids = tuple(str(v) for v in variant_ids)
    m, k = len(ids), len(cluster.traits)
    n_shared = int(round(float(cluster.shared_fraction) * m))
    if n_shared + (m - n_shared) // k < 1:
        raise ValueError(f"pleiotropy cluster {cluster.name}: too few causal variants ({m}) for {k} traits")
    rng = np.random.default_rng(int(cluster.random_seed) + 1000 * int(replicate))
    order = rng.permutation(m)
    shared = np.zeros(m, dtype=bool)
    shared[order[:n_shared]] = True
    owner = np.full(m, -1, dtype=np.int64)
    owner[order[n_shared:]] = np.arange(m - n_shared) % k
    return ClusterRoles(cluster.name, ids, shared, owner, tuple(cluster.traits))


def mix_cluster_betas(
    beta_dfs: Mapping[str, pd.DataFrame],
    base_sds: Mapping[str, float],
    roles: ClusterRoles,
    effect_correlation: Sequence[Sequence[float]],
) -> Dict[str, pd.DataFrame]:
    """Correlated shared effects, own private effects, zero elsewhere.

    ``beta_dfs`` are the unclustered long tables from
    ``sample_population_specific_betas`` (variant_id, population, beta_base,
    beta_delta, beta, effect_scale), all over ``roles.variant_ids``. Rows are kept
    for every variant so genotype matrices line up; non-causal rows get zero
    effects and are dropped from the persisted tables by the caller.
    """
    traits = roles.traits
    factor = lower_factor(effect_correlation)
    index = pd.Index(roles.variant_ids, name="variant_id")
    first = beta_dfs[traits[0]]
    populations = list(dict.fromkeys(first["population"].astype(str)))
    raw: List[Dict[str, np.ndarray]] = []
    for trait in traits:
        df = beta_dfs[trait]
        if len(df) != len(index) * len(populations):
            raise ValueError(f"beta table for {trait} does not cover the cluster's variants")
        if not float(base_sds[trait]) > 0:
            raise ValueError(f"base effect SD for {trait} must be positive")
        raw.append({
            col: df.pivot(index="variant_id", columns="population", values=col)
                   .reindex(index=index, columns=populations).to_numpy(dtype=float)
            for col in BETA_COLUMNS
        })
    shared = roles.shared[:, None]
    out: Dict[str, pd.DataFrame] = {}
    for k, trait in enumerate(traits):
        own = roles.private_owner[:, None] == k
        sd = float(base_sds[trait])
        mixed_cols = {}
        for col in BETA_COLUMNS:
            # Own draw enters unscaled, so an identity factor reproduces it bit for bit.
            mixed = factor[k, k] * raw[k][col]
            for l in range(k):
                if factor[k, l] != 0.0:
                    mixed = mixed + factor[k, l] * (sd / float(base_sds[traits[l]])) * raw[l][col]
            value = np.where(shared, mixed, np.where(own, raw[k][col], 0.0))
            mixed_cols[col] = pd.DataFrame(value, index=index, columns=populations).stack()
        long = pd.DataFrame(mixed_cols)
        long.index = long.index.set_names(["variant_id", "population"])
        long = long.reset_index()
        df = beta_dfs[trait].drop(columns=list(BETA_COLUMNS)).merge(long, on=["variant_id", "population"], how="left")
        out[trait] = df[list(beta_dfs[trait].columns)]
    return out


def mix_cluster_residuals(
    raw_by_trait: Mapping[str, np.ndarray],
    traits: Sequence[str],
    residual_correlation: Sequence[Sequence[float]] | None,
) -> Dict[str, np.ndarray]:
    """Correlate the traits' raw unit-variance residual draws; identity leaves them unchanged."""
    if residual_correlation is None:
        return {t: raw_by_trait[t] for t in traits}
    factor = lower_factor(residual_correlation)
    stacked = [np.asarray(raw_by_trait[t], dtype=float) for t in traits]
    return {t: sum(factor[k, l] * stacked[l] for l in range(k + 1)) for k, t in enumerate(traits)}


def realized_genetic_correlation(
    genetic_by_trait: Mapping[str, np.ndarray],
    traits: Sequence[str],
) -> pd.DataFrame:
    """Pairwise correlation of the traits' fixed genetic components (one cohort)."""
    rows = []
    for i, a in enumerate(traits):
        for b in traits[i + 1:]:
            rows.append({"trait_a": a, "trait_b": b,
                         "realized_genetic_correlation": float(np.corrcoef(genetic_by_trait[a], genetic_by_trait[b])[0, 1])})
    return pd.DataFrame(rows)


__all__ = [
    "ClusterRoles",
    "cluster_roles",
    "lower_factor",
    "mix_cluster_betas",
    "mix_cluster_residuals",
    "realized_genetic_correlation",
]
