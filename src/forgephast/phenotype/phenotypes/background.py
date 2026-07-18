"""Background-genetic covariance inputs and random-effect sampling."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import RunConfig


def _load_precomputed_kinship(
    kinship_file: str,
    sample_ids: List[str],
) -> np.ndarray:
    """Load an IID-indexed dense kinship matrix and align to sample order."""
    kin = pd.read_csv(kinship_file, sep="\t")
    id_col = (
        "IID"
        if "IID" in kin.columns
        else "#IID"
        if "#IID" in kin.columns
        else None
    )
    if id_col is None:
        raise ValueError("precomputed kinship must include IID or #IID column")
    kin = kin.set_index(id_col)

    missing_rows = [iid for iid in sample_ids if iid not in kin.index]
    if missing_rows:
        raise ValueError(
            f"Kinship matrix missing sample rows; first few: {missing_rows[:5]} "
            f"(n={len(missing_rows)})"
        )
    missing_cols = [iid for iid in sample_ids if iid not in kin.columns]
    if missing_cols:
        raise ValueError(
            f"Kinship matrix missing sample columns; first few: {missing_cols[:5]} "
            f"(n={len(missing_cols)})"
        )
    aligned = kin.loc[sample_ids, sample_ids]
    arr = aligned.to_numpy(dtype=float)
    return 0.5 * (arr + arr.T)


def _read_background_variant_ids(path: str) -> List[str]:
    """Read one variant ID from the first whitespace-delimited field per line."""
    variant_ids: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token and not token.startswith("#"):
                variant_ids.append(token.split()[0])
    if variant_ids and variant_ids[0].lower() in {"variant_id", "id"}:
        variant_ids = variant_ids[1:]
    if not variant_ids:
        raise ValueError(f"No background GRM variants found in {path}")
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError(f"Duplicate background GRM variant IDs found in {path}")
    return variant_ids


def _exclude_causal_grm_variants(
    configured_variant_ids: List[str],
    causal_variant_ids: List[str],
) -> tuple[List[str], int]:
    causal = set(causal_variant_ids)
    retained = [
        variant_id
        for variant_id in configured_variant_ids
        if variant_id not in causal
    ]
    excluded_count = len(configured_variant_ids) - len(retained)
    if not retained:
        raise ValueError(
            "No background GRM variants remain after excluding selected causal variants"
        )
    return retained, excluded_count


def _standardize_grm_genotypes(x: np.ndarray) -> np.ndarray:
    """Return standardized polymorphic genotypes as the implicit GRM factor."""
    x = np.array(x, dtype=np.float32, copy=True)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("Cannot compute GRM with zero dosage variants")
    x -= x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    keep = (std > 1e-12).reshape(-1)
    if not np.any(keep):
        raise ValueError("Cannot compute GRM: all dosage variants are monomorphic")
    x = x[:, keep]
    x /= std[:, keep]
    return x.astype(np.float32, copy=False)


def _sample_background_from_grm_factor(
    grm_factor: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample N(0, ZZ'/M) without materializing the dense GRM."""
    z = np.asarray(grm_factor, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] == 0:
        raise ValueError(
            "compute_grm mode requires standardized background genotypes"
        )
    rng = np.random.default_rng(int(seed))
    marker_effects = rng.normal(
        0.0,
        1.0 / np.sqrt(float(z.shape[1])),
        size=z.shape[1],
    ).astype(np.float32)
    return np.asarray(z @ marker_effects, dtype=float)


def _sample_mvnormal_from_cov(
    cov: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Sample one vector from N(0, cov) with eigenvalue clipping for PSD safety."""
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    rng = np.random.default_rng(int(seed))
    z = rng.normal(0.0, 1.0, size=cov.shape[0])
    return evecs @ (np.sqrt(evals) * z)


def _build_background_component(
    run_cfg: RunConfig,
    sample_ids: List[str],
    grm_factor: np.ndarray | None,
    cache_key: str,
    replicate_seed: int,
    kinship_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Build infinitesimal background genetic effect component."""
    mode = run_cfg.background_genetic.mode
    if mode == "none":
        return np.zeros(len(sample_ids), dtype=float)

    if mode == "precomputed_kinship":
        if cache_key not in kinship_cache:
            kinship_cache[cache_key] = _load_precomputed_kinship(
                kinship_file=str(run_cfg.background_genetic.kinship_file),
                sample_ids=sample_ids,
            )
        cov = kinship_cache[cache_key]
    elif mode == "compute_grm":
        if grm_factor is None:
            raise ValueError(
                "compute_grm mode requires an independent GRM variant set"
            )
        return _sample_background_from_grm_factor(
            grm_factor=grm_factor,
            seed=replicate_seed,
        )
    else:
        raise ValueError(f"Unsupported background_genetic.mode={mode}")

    return _sample_mvnormal_from_cov(cov=cov, seed=replicate_seed)
