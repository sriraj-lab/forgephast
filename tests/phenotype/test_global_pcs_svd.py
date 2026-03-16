"""SVD-based global PCA tests for HapStore mode."""

from __future__ import annotations

import numpy as np

from forgephast.phenotype.data.global_pcs_svd import compute_global_pcs_svd


def test_compute_global_pcs_svd_is_deterministic() -> None:
    x = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, np.nan],
            [1.0, 1.0, 1.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 2.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    sample_ids = ["s1", "s2", "s3", "s4"]

    pcs1 = compute_global_pcs_svd(
        genotypes=x,
        sample_ids=sample_ids,
        n_components=3,
        max_variants=5,
        min_maf=0.0,
        random_seed=123,
    )
    pcs2 = compute_global_pcs_svd(
        genotypes=x,
        sample_ids=sample_ids,
        n_components=3,
        max_variants=5,
        min_maf=0.0,
        random_seed=123,
    )

    assert list(pcs1.columns) == ["FID", "IID", "PC1", "PC2", "PC3"]
    assert pcs1["IID"].tolist() == sample_ids
    assert np.allclose(
        pcs1[["PC1", "PC2", "PC3"]].to_numpy(),
        pcs2[["PC1", "PC2", "PC3"]].to_numpy(),
    )


def test_compute_global_pcs_svd_caps_variants_with_seed() -> None:
    x = np.array(
        [
            [0.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 2.0, 1.0, 2.0],
        ],
        dtype=float,
    )

    pcs_a = compute_global_pcs_svd(
        genotypes=x,
        sample_ids=["s1", "s2", "s3", "s4"],
        n_components=2,
        max_variants=3,
        min_maf=0.0,
        random_seed=7,
    )
    pcs_b = compute_global_pcs_svd(
        genotypes=x,
        sample_ids=["s1", "s2", "s3", "s4"],
        n_components=2,
        max_variants=3,
        min_maf=0.0,
        random_seed=7,
    )

    assert np.allclose(
        pcs_a[["PC1", "PC2"]].to_numpy(),
        pcs_b[["PC1", "PC2"]].to_numpy(),
    )
