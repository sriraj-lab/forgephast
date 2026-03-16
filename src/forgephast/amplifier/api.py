"""Public Python API surface."""

from __future__ import annotations

from pathlib import Path

from .config import from_yaml
from .donor_cache import DonorCacheResult, build_donor_cache
from .donors import Manifest, build_manifest_from_config
from .merge import MergeResult, merge_chromosome
from .qc import (
    DonorWithinPopRoundRobinReport,
    QcReport,
    SyntheticPremergeWithinPopRoundRobinReport,
    run_donor_within_pop_round_robin,
    run_qc,
    run_synthetic_premerge_within_pop_round_robin,
)
from .simulation import SimulationBatchResult, SimulationResult, simulate_chromosome, simulate_chromosome_population_shard_pool


def build_manifest(config_path: str | Path) -> Manifest:
    cfg = from_yaml(config_path)
    return build_manifest_from_config(cfg)


def build_cache(
    config_path: str | Path,
    chroms: list[int] | None = None,
    force: bool = False,
) -> DonorCacheResult:
    cfg = from_yaml(config_path)
    return build_donor_cache(cfg, chroms=chroms, force=force)


def simulate(
    config_path: str | Path,
    chrom: int,
    shard_id: int = 0,
    n_shards: int = 1,
) -> SimulationResult:
    cfg = from_yaml(config_path)
    return simulate_chromosome(cfg, chrom=chrom, shard_id=shard_id, n_shards=n_shards)


def simulate_pool(
    config_path: str | Path,
    chrom: int,
    n_shards: int | None = None,
    workers: int | None = None,
    threads_per_task: int | None = None,
    task_start_method: str | None = None,
) -> SimulationBatchResult:
    cfg = from_yaml(config_path)
    resolved_n_shards = n_shards if n_shards is not None else cfg.simulation.n_shards
    resolved_task_workers = workers if workers is not None else cfg.simulation.task_workers
    if resolved_n_shards is None:
        raise ValueError(
            "simulate_pool requires n_shards via CLI/API or simulation.n_shards in config"
        )
    if resolved_task_workers is None:
        raise ValueError(
            "simulate_pool requires workers via CLI/API or simulation.task_workers in config"
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
) -> MergeResult:
    cfg = from_yaml(config_path)
    return merge_chromosome(
        cfg,
        chrom=chrom,
        n_shards=n_shards,
        merge_combined=merge_combined,
        input_root=input_root,
    )


def qc(config_path: str | Path, chrom: int, n_shards: int = 1) -> QcReport:
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
    cfg = from_yaml(config_path)
    return run_donor_within_pop_round_robin(
        cfg,
        chrom=chrom,
        fold_fraction=fold_fraction,
        seed=seed,
        workers=workers,
        populations=populations,
    )


def qc_synthetic_premerge_round_robin(
    config_path: str | Path,
    chrom: int,
    n_shards: int,
    n_folds: int | None = None,
    seed: int = 42,
    workers: int = 1,
    populations: list[str] | None = None,
) -> SyntheticPremergeWithinPopRoundRobinReport:
    cfg = from_yaml(config_path)
    return run_synthetic_premerge_within_pop_round_robin(
        cfg,
        chrom=chrom,
        n_shards=n_shards,
        n_folds=n_folds,
        seed=seed,
        workers=workers,
        populations=populations,
    )
