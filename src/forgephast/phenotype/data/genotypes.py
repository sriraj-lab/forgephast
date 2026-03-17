"""Canonical genotype access utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids

logger = logging.getLogger(__name__)


@runtime_checkable
class GenotypeAdapter(Protocol):
    """Protocol for genotype adapters used by phenotype simulation."""

    last_export_allele_by_variant: dict[str, str]

    def validate_files(self) -> None:
        """Validate required input artifacts."""

    def export_dosages(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Export additive dosage matrix for requested samples/variants."""

    def export_haplotypes(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Export phased haplotypes as shape (n_samples, n_variants, 2) array."""

    def compute_global_pcs(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        n_components: int,
        out_dir: Path,
        mode_cfg: object,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Compute global PCs for requested samples."""

    def export_to_pgen(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        out_prefix: Path,
        chromosomes: Sequence[int] | None = None,
        variant_ids: List[str] | None = None,
        export_cfg: object | None = None,
    ) -> str:
        """Export selected samples/variants to PGEN and return output prefix."""


def check_id_alignment(sample_ids: List[str], dosage_df: pd.DataFrame) -> None:
    """Check dosage table IID order against expected sample IDs."""
    got = dosage_df["IID"].astype(str).tolist()
    if len(got) != len(sample_ids):
        raise ValueError(f"ID alignment failed: expected {len(sample_ids)} rows, got {len(got)}")
    mismatches = [
        (i, exp, obs)
        for i, (exp, obs) in enumerate(zip(sample_ids, got))
        if exp != obs
    ]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"ID alignment mismatch at row {first[0]}: expected {first[1]} got {first[2]}"
        )


def build_genotype_adapter(
    *,
    hapstore_cfg: object,
) -> GenotypeAdapter:
    """Factory for simulation genotype adapter."""
    if hapstore_cfg is None:
        raise ValueError("hapstore_cfg is required")
    from .hapstore_adapter import HapStoreAdapter

    return HapStoreAdapter(hapstore_cfg=hapstore_cfg)
