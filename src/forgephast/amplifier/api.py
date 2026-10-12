"""Public Python interface for the FORGEPHAST amplification workflow.

The workflow resolves donor manifests, prepares reusable donor HapStore inputs,
generates synthetic chromosome shards, merges shard HapStores, and runs QC.
Stages can be invoked independently once their required upstream artifacts
exist. This module backs both the CLI and downstream Python callers.
"""

from __future__ import annotations

from pathlib import Path

from .config import from_yaml
from .donor_cache import DonorCacheResult, build_donor_cache
from .donors import Manifest, build_manifest_from_config
from .merge import MergeResult, merge_chromosome
from .qc import (
    DonorWithinPopRoundRobinReport,
    QcReport,
    run_donor_within_pop_round_robin,
    run_qc,
)
from .simulation import SimulationBatchResult, simulate_chromosome_population_shard_pool


def build_manifest(config_path: str | Path) -> Manifest:
    """Write donor lists, weights, and a manifest resolved from YAML metadata.

    Mosaic populations are resolved from their configured selectors. For
    ``msprime_tracts`` populations, this stage validates the donor-pool
    populations named as ancestry sources.
    """
    cfg = from_yaml(config_path)
    return build_manifest_from_config(cfg)


def build_cache(
    config_path: str | Path,
    chroms: list[int] | None = None,
    force: bool = False,
) -> DonorCacheResult:
    """Prepare or reuse donor HapStore inputs for the requested chromosomes.

    VCF donor panels are converted to fingerprinted HapStore caches, while
    HapStore inputs are validated and reused directly. ``chroms`` defaults to
    the configured simulation chromosomes; ``force`` rebuilds VCF-backed
    caches even when a matching cache receipt exists.
    """
    cfg = from_yaml(config_path)
    return build_donor_cache(cfg, chroms=chroms, force=force)


def simulate(
    config_path: str | Path,
    chrom: int,
    n_shards: int | None = None,
    workers: int | None = None,
    threads_per_task: int | None = None,
    task_start_method: str | None = None,
) -> SimulationBatchResult:
    """Generate synthetic samples for one chromosome.

    Explicit shard and runtime arguments override ``simulation.*`` YAML
    settings. A single-shard run exposes configured combined outputs directly;
    multi-shard runs require :func:`merge`. Configurations containing an
    ``msprime_tracts`` population use the current shard-sequential path.
    """
    cfg = from_yaml(config_path)
    resolved_n_shards = n_shards if n_shards is not None else cfg.simulation.n_shards
    resolved_task_workers = workers if workers is not None else cfg.simulation.task_workers
    if resolved_n_shards is None:
        raise ValueError(
            "simulate requires n_shards via CLI/API or simulation.n_shards in config"
        )
    if resolved_task_workers is None:
        raise ValueError(
            "simulate requires workers via CLI/API or simulation.task_workers in config"
        )
    resolved_task_start_method = (
        task_start_method
        if task_start_method is not None
        else cfg.simulation.task_start_method
    )
    resolved_threads_per_task = (
        threads_per_task if threads_per_task is not None else cfg.simulation.threads_per_task
    )
    return simulate_chromosome_population_shard_pool(
        cfg,
        chrom=chrom,
        n_shards=int(resolved_n_shards),
        task_workers=int(resolved_task_workers),
        threads_per_task=int(resolved_threads_per_task),
        task_start_method=str(resolved_task_start_method),
    )


def merge(
    config_path: str | Path,
    chrom: int,
    n_shards: int,
    merge_combined: bool | None = None,
    input_root: str | Path | None = None,
    populations: list[str] | None = None,
) -> MergeResult:
    """Merge per-population shard HapStores along the sample axis.

    The stage writes a merged sample manifest, optionally combines population
    stores, and writes PGEN from the combined HapStore when configured.
    ``input_root`` changes only where shard inputs are read; outputs remain
    under the run root specified by the configuration.
    """
    cfg = from_yaml(config_path)
    return merge_chromosome(
        cfg,
        chrom=chrom,
        n_shards=n_shards,
        merge_combined=merge_combined,
        input_root=input_root,
        populations=populations,
    )


def qc(config_path: str | Path, chrom: int, n_shards: int = 1) -> QcReport:
    """Run configured QC metrics and hard gates for one chromosome.

    ``n_shards`` identifies the synthetic artifacts to evaluate. Metric
    selection, diagnostics, and which comparisons act as gates are defined by
    the configuration.
    """
    cfg = from_yaml(config_path)
    return run_qc(cfg, chrom=chrom, n_shards=n_shards)


def qc_donor_round_robin(
    config_path: str | Path,
    chrom: int,
    fold_fraction: float = 0.2,
    seed: int = 42,
    workers: int = 1,
    populations: list[str] | None = None,
) -> DonorWithinPopRoundRobinReport:
    """Measure within-population donor FST variation across seeded folds.

    This donor-only diagnostic estimates the baseline dispersion expected from
    finite donor sampling; it does not compare synthetic cohorts or donor
    populations against one another.
    """
    cfg = from_yaml(config_path)
    return run_donor_within_pop_round_robin(
        cfg,
        chrom=chrom,
        fold_fraction=fold_fraction,
        seed=seed,
        workers=workers,
        populations=populations,
    )
