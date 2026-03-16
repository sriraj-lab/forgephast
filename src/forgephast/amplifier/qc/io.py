"""Small QC I/O helpers shared across QC metrics."""

from __future__ import annotations

import numpy as np

from forgephast._internal.qc_kernels import haplotypes_to_genotypes_f32 as _utils_haplotypes_to_genotypes_f32
from forgephast._internal.qc_kernels import haplotypes_to_genotypes_u8 as _utils_haplotypes_to_genotypes_u8

_MISSING_SENTINEL = 255


def haplotypes_to_genotypes_u8(
    haplotypes: np.ndarray,
    *,
    missing_sentinel: int = _MISSING_SENTINEL,
) -> np.ndarray:
    """Convert HapStore-style haplotypes (2N, V) -> genotypes (N, V) with missing propagation.

    HapStore convention: uint8 haplotypes with values {0, 1, 255} where 255 denotes missing.
    Output uses uint8 genotypes with values {0, 1, 2, 255}.
    """
    return _utils_haplotypes_to_genotypes_u8(haplotypes, missing_sentinel=missing_sentinel)


def haplotypes_to_genotypes_f32(
    haplotypes: np.ndarray,
    *,
    missing_sentinel: int = _MISSING_SENTINEL,
) -> np.ndarray:
    """Convert haplotypes (2N, V) -> float32 genotype matrix (N, V) with NaN missing."""
    return _utils_haplotypes_to_genotypes_f32(haplotypes, missing_sentinel=missing_sentinel)
