"""Global PCA with deterministic variant subsetting and SVD."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

_SVD_FALLBACK_MAX_ELEMENTS = 2_000_000
_SVD_FALLBACK_MAX_DIM = 2000


def _randomized_svd_compat(
    x: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD when available, otherwise bounded numpy fallback."""
    try:
        from sklearn.utils.extmath import randomized_svd  # type: ignore

        u, s, vt = randomized_svd(
            x,
            n_components=int(n_components),
            random_state=int(random_state),
        )
        return (
            np.asarray(u, dtype=np.float64),
            np.asarray(s, dtype=np.float64),
            np.asarray(vt, dtype=np.float64),
        )
    except Exception as exc:  # noqa: BLE001 - preserve fallback behavior
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size > _SVD_FALLBACK_MAX_ELEMENTS or min(x_arr.shape) > _SVD_FALLBACK_MAX_DIM:
            raise RuntimeError(
                "HapStore PCA requires scikit-learn randomized_svd for this matrix size; "
                "install scikit-learn or reduce svd_max_variants."
            ) from exc

        u, s, vt = np.linalg.svd(x_arr, full_matrices=False)
        k = min(int(n_components), int(vt.shape[0]))
        return u[:, :k], s[:k], vt[:k]


def _subset_variant_columns(
    x: np.ndarray,
    *,
    max_variants: int,
    random_seed: int,
) -> np.ndarray:
    """Deterministically subset columns when variant count exceeds cap."""
    if x.shape[1] <= int(max_variants):
        return x
    rng = np.random.default_rng(int(random_seed))
    idx = rng.choice(x.shape[1], size=int(max_variants), replace=False)
    idx.sort()
    return x[:, idx]


def _maf_filter_and_standardize(
    x: np.ndarray,
    *,
    min_maf: float,
) -> np.ndarray:
    """Apply MAF filter, mean imputation, and 2pq standardization."""
    observed = np.isfinite(x)
    ac = np.nansum(x.astype(np.float64), axis=0)
    n_obs = np.sum(observed, axis=0, dtype=np.float64)
    an = 2.0 * n_obs

    with np.errstate(divide="ignore", invalid="ignore"):
        p = ac / an
    maf = np.minimum(p, 1.0 - p)

    valid = (
        np.isfinite(p)
        & np.isfinite(maf)
        & (an >= 4.0)
        & (maf >= float(min_maf))
        & (maf <= 0.5)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("No variants remain after HapStore PCA MAF filtering")

    p = p[valid]
    mean = 2.0 * p
    scale = np.sqrt(2.0 * p * (1.0 - p))

    keep_scale = scale > 1e-12
    if np.count_nonzero(keep_scale) < 2:
        raise ValueError("No variants remain after HapStore PCA scale filtering")

    mean = mean[keep_scale]
    scale = scale[keep_scale]

    z = np.asarray(x[:, valid][:, keep_scale], dtype=np.float64)
    missing = ~np.isfinite(z)
    if np.any(missing):
        col_idx = np.where(missing)[1]
        z[missing] = mean[col_idx]

    z -= mean
    z /= scale
    return z


def compute_global_pcs_svd(
    *,
    genotypes: np.ndarray,
    sample_ids: list[str],
    n_components: int,
    max_variants: int,
    min_maf: float,
    random_seed: int,
) -> pd.DataFrame:
    """Compute PCs from dosage matrix using PCA-SVD workflow."""
    x = np.asarray(genotypes, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"genotypes must be 2D, got shape={x.shape}")
    if x.shape[0] != len(sample_ids):
        raise ValueError(
            f"genotype/sample mismatch: rows={x.shape[0]} sample_ids={len(sample_ids)}"
        )
    if int(n_components) <= 0:
        raise ValueError("n_components must be > 0")

    x = _subset_variant_columns(
        x,
        max_variants=int(max_variants),
        random_seed=int(random_seed),
    )
    x = _maf_filter_and_standardize(x, min_maf=float(min_maf))

    n_comp_eff = min(int(n_components), x.shape[0] - 1, x.shape[1])
    if n_comp_eff <= 0:
        raise ValueError("Insufficient rank to compute requested PCA components")

    _, _, vt = _randomized_svd_compat(
        x,
        n_components=n_comp_eff,
        random_state=int(random_seed),
    )
    scores = x @ vt.T

    out = pd.DataFrame({"FID": sample_ids, "IID": sample_ids})
    for i in range(1, int(n_components) + 1):
        if i <= n_comp_eff:
            out[f"PC{i}"] = scores[:, i - 1]
        else:
            out[f"PC{i}"] = 0.0
    return out
