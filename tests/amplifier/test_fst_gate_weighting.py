from __future__ import annotations

import numpy as np
import pytest

from forgephast.amplifier.qc.fst_gate import (
    normalized_donor_weights,
    weighted_genotype_counts,
)


def test_normalized_donor_weights_preserves_manifest_order() -> None:
    observed = normalized_donor_weights(
        ["sample_b", "sample_a", "sample_c"],
        {"sample_a": 0.6, "sample_b": 0.3, "sample_c": 0.1},
    )
    np.testing.assert_allclose(observed, [0.3, 0.6, 0.1])


def test_weighted_genotype_counts_matches_unweighted_counts_for_uniform_weights() -> None:
    genotypes = np.asarray(
        [
            [0, 1, 2],
            [1, 1, 0],
            [2, 0, 1],
            [1, 2, 1],
        ],
        dtype=np.uint8,
    )
    allele_count, allele_number = weighted_genotype_counts(
        genotypes,
        np.full(4, 0.25),
    )
    np.testing.assert_allclose(allele_count, np.sum(genotypes, axis=0))
    np.testing.assert_allclose(allele_number, np.full(3, 8.0))


def test_weighted_genotype_counts_uses_kish_effective_sample_size() -> None:
    genotypes = np.asarray([[0, 2], [2, 0]], dtype=np.uint8)
    allele_count, allele_number = weighted_genotype_counts(
        genotypes,
        np.asarray([0.75, 0.25]),
    )
    effective_n = 1.0 / (0.75**2 + 0.25**2)
    np.testing.assert_allclose(allele_number, np.full(2, 2.0 * effective_n))
    np.testing.assert_allclose(
        allele_count / allele_number,
        [0.25, 0.75],
    )


def test_normalized_donor_weights_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match="positive value"):
        normalized_donor_weights(["a", "b"], {"a": 0.0, "b": 0.0})
