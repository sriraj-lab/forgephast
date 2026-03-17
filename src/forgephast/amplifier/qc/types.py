"""Shared QC report, input contracts, and type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QcReport:
    chrom: int
    vcf_path: Path | None
    sample_manifest_path: Path
    pairwise_fst: dict[str, float]
    within_split_fst: dict[str, float]
    gate_pass: bool
    diagnostics_path: Path
    mode: str
    hapstore_path: Path | None = None
    population_hapstore_paths: dict[str, Path] | None = None
    donor_pairwise_fst: dict[str, float] | None = None
    ld_decay: dict[str, Any] | None = None
    ld_decay_table_path: Path | None = None
    elapsed_seconds: float = 0.0
    stage_timings: dict[str, float] | None = None


@dataclass
class DonorWithinPopRoundRobinReport:
    chrom: int
    n_folds: int
    fold_fraction: float
    seed: int
    workers: int
    diagnostics_path: Path
    table_path: Path
    summary_by_population: dict[str, dict[str, float | int]]
    n_records: int


@dataclass
class QcInputs:
    mode: str
    sample_manifest: Path
    combined_vcf: Path | None
    combined_hapstore: Path | None
    population_hapstores: dict[str, Path]


__all__ = [
    "QcReport",
    "DonorWithinPopRoundRobinReport",
    "QcInputs",
]
