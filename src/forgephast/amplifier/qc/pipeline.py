"""QC and population differentiation diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics.ld_decay import (
    ld_decay_metrics_from_curves as local_ld_decay_metrics_from_curves,
    compute_ld_decay_qc_impl as _metric_compute_ld_decay_qc_impl,
)
from . import data_support
from . import donor_round_robin_support
from . import run_support
from .types import (
    DonorWithinPopRoundRobinReport,
    QcInputs,
    QcReport,
)
from . import round_robin_support as rr_support
from .round_robin_support import (
    round_robin_folds_from_fraction as _round_robin_folds_from_fraction,
)

from ..config import RunConfig
from ..donors import load_manifest
from .metrics.fst import (
    hudson_fst_from_counts as matrix_hudson_fst_from_counts,
)
from ..utils import ensure_dir, write_json

logger = logging.getLogger(__name__)


def _resolve_requested_populations(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    if not requested:
        return sorted(available)
    available_set = set(available)
    out: list[str] = []
    for pop in requested:
        pop_s = str(pop)
        if pop_s not in available_set:
            raise ValueError(f"Requested population {pop_s!r} is not present in donor manifest")
        if pop_s not in out:
            out.append(pop_s)
    return out


def _ld_decay_metrics_from_curves(
    *,
    donor_curve: pd.DataFrame,
    synthetic_curve: pd.DataFrame,
    min_pairs_per_bin: int,
    half_decay_r2_threshold: float,
) -> dict[str, float | int | str]:
    # Compatibility wrapper retained for existing tests/import sites.
    return local_ld_decay_metrics_from_curves(
        donor_curve=donor_curve,
        synthetic_curve=synthetic_curve,
        min_pairs_per_bin=min_pairs_per_bin,
        half_decay_r2_threshold=half_decay_r2_threshold,
    )


def _compute_ld_decay_qc_python(
    *,
    config: RunConfig,
    chrom: int,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
) -> tuple[dict[str, Any], Path]:
    return _metric_compute_ld_decay_qc_impl(
        config=config,
        chrom=int(chrom),
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=numba_threads,
        resolve_donor_panel_source_for_population_fn=lambda c, pop: data_support.resolve_donor_panel_source(
            config,
            int(c),
            str(pop),
        ),
        resolve_map_path_for_population_fn=lambda c, pop: data_support.resolve_map_path(
            config,
            int(c),
            str(pop),
        ),
        load_sample_haplotypes_from_panel_fn=lambda *, source_path, chrom, sample_ids, source_kind: (
            data_support.load_sample_haplotypes_from_panel(
                config=config,
                source_path=Path(source_path),
                chrom=int(chrom),
                sample_ids=list(sample_ids),
                source_kind=str(source_kind),
            )
        ),
        load_synthetic_population_haplotypes_fn=lambda *, inputs, chrom, pop, sample_ids: (
            data_support.load_synthetic_population_haplotypes(
                inputs=inputs,
                chrom=int(chrom),
                pop=str(pop),
                sample_ids=list(sample_ids),
            )
        ),
    )


def _compute_ld_decay_qc(
    *,
    config: RunConfig,
    chrom: int,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
) -> tuple[dict[str, Any], Path]:
    return _compute_ld_decay_qc_python(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=numba_threads,
    )


def _even_folds(sample_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(sample_ids) < n_folds:
        raise ValueError(f"n_folds={n_folds} exceeds available donors={len(sample_ids)}")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sample_ids, dtype=object)
    rng.shuffle(shuffled)
    return [chunk.tolist() for chunk in np.array_split(shuffled, n_folds) if len(chunk) > 0]


def donor_baseline_fst(
    config: RunConfig,
    chrom: int,
    *,
    pairs: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    return run_support.donor_baseline_fst(config, chrom=int(chrom), pairs=pairs)


def _round_robin_for_population_chrom(
    *,
    config: RunConfig,
    pop: str,
    chrom: int,
    sample_ids: list[str],
    n_folds: int,
    seed: int,
    numba_threads: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    source_kind, source_path = data_support.resolve_donor_panel_source(
        config,
        chrom,
        pop,
    )
    return rr_support.round_robin_for_population_chrom(
        pop=pop,
        chrom=chrom,
        sample_ids=sample_ids,
        n_folds=n_folds,
        seed=seed,
        numba_threads=numba_threads,
        even_folds_fn=_even_folds,
        load_sample_haplotypes_fn=lambda fold_ids: data_support.load_sample_haplotypes_from_panel(
            config=config,
            source_path=source_path,
            chrom=chrom,
            sample_ids=list(fold_ids),
            source_kind=source_kind,
        ),
        matrix_hudson_fst_from_counts_fn=matrix_hudson_fst_from_counts,
    )


def run_donor_within_pop_round_robin(
    config: RunConfig,
    chrom: int,
    fold_fraction: float = 0.2,
    seed: int = 42,
    workers: int = 1,
    populations: list[str] | None = None,
) -> DonorWithinPopRoundRobinReport:
    return donor_round_robin_support.run_donor_within_pop_round_robin_impl(
        config=config,
        chrom=int(chrom),
        fold_fraction=float(fold_fraction),
        seed=int(seed),
        workers=int(workers),
        populations=populations,
        round_robin_folds_from_fraction_fn=_round_robin_folds_from_fraction,
        load_manifest_fn=load_manifest,
        resolve_requested_populations_fn=_resolve_requested_populations,
        ensure_dir_fn=ensure_dir,
        round_robin_for_population_fn=lambda **kwargs: _round_robin_for_population_chrom(
            config=config,
            **kwargs,
        ),
        write_json_fn=write_json,
    )


def run_qc(config: RunConfig, chrom: int, n_shards: int = 1) -> QcReport:
    return run_support.run_qc_impl(
        config=config,
        chrom=int(chrom),
        n_shards=int(n_shards),
        logger=logger,
    )
