"""Simulation engine for synthetic population amplification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

from hapstore import ingest_arrays as _hapstore_ingest_arrays
import numpy as np
import pandas as pd

from .._internal.resources import (
    environment_scratch_root,
    execution_token,
    require_cpu_budget,
)
from ..local_ancestry import write_local_ancestry_source_from_tract_rows
from .config import RunConfig
from .donors import load_manifest
from .donor_panels import load_donor_haplotypes_from_source, subset_panel_donors
from .merge import merge_chromosomes_sample_axis
from .maps import load_genetic_map, switch_probs_from_positions
from .msprime_tracts import simulate_msprime_tract_haplotypes
from forgephast._internal.mp_logging import install_queue_handler, setup_queue_listener_for_logger
from .telemetry import RateLimitedProgress, telemetry_span
from .utils import derived_seed, ensure_dir, shard_bounds
from .vcfio import (
    convert_hapstore_to_vcf,
    convert_hapstore_to_pgen,
)

logger = logging.getLogger(__name__)


def _resolve_scratch_root(config: RunConfig) -> Path | None:
    if not bool(config.outputs.scratch_first):
        return None
    if config.outputs.scratch_root is not None:
        return Path(config.outputs.scratch_root)
    return environment_scratch_root()


def _scratch_dir(
    config: RunConfig,
    *,
    chrom: int,
    shard_id: int,
    n_shards: int,
    population: str | None = None,
) -> Path | None:
    root = _resolve_scratch_root(config)
    if root is None:
        return None
    job_token = execution_token()
    path = (
        root
        / "forgephast"
        / config.name
        / str(job_token)
        / f"chr{chrom}"
        / f"shard_{shard_id}_of_{n_shards}"
    )
    if population is not None:
        path = path / str(population)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_scratch_tree(source: Path, destination: Path) -> float:
    started = perf_counter()
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return perf_counter() - started


def _is_stale_hapstore_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("already exists" in msg) and (
        ("chromosome" in msg)
        or ("chrom" in msg)
        or ("group" in msg)
        or ("zarr" in msg)
    )


def _ingest_arrays_with_retry(*args, **kwargs):
    """Ingest arrays with stale-store overwrite retry.

    If a prior failed run left a stale chromosome/group in the destination Zarr
    store, the hapstore library may raise an "already exists" style error. In
    that case we remove the destination store and retry once.
    """
    output_zarr = kwargs.get("output_zarr", args[0] if args else None)
    output_path = Path(output_zarr) if output_zarr is not None else None

    try:
        return _hapstore_ingest_arrays(*args, **kwargs)
    except Exception as exc:
        if output_path is None or not output_path.exists() or not _is_stale_hapstore_error(exc):
            raise
        logger.warning(
            "Detected stale HapStore at %s after ingest error; removing and retrying once: %s",
            output_path,
            exc,
        )
        shutil.rmtree(output_path, ignore_errors=True)
        return _hapstore_ingest_arrays(*args, **kwargs)


def _write_pgen_from_hapstore(
    *,
    out_prefix: Path,
    chrom: int,
    hapstore_path: Path | None,
) -> Path:
    if hapstore_path is None:
        raise RuntimeError("HapStore-backed PGEN export requires HapStore output")
    convert_hapstore_to_pgen(
        hapstore_path=hapstore_path,
        out_prefix=out_prefix,
        chrom=chrom,
    )
    return out_prefix


def _write_vcf_from_hapstore(
    *,
    output_vcf: Path,
    chrom: int,
    hapstore_path: Path | None,
    plink2_location: str | None,
    keep_intermediate_pgen: bool,
    n_shards: int,
) -> Path:
    if hapstore_path is None:
        raise RuntimeError("HapStore-backed phased VCF export requires HapStore output")
    return convert_hapstore_to_vcf(
        hapstore_path=hapstore_path,
        output_vcf=output_vcf,
        chrom=int(chrom),
        variant_block_size=1024,
        plink2_location=plink2_location,
        keep_intermediate_files=bool(keep_intermediate_pgen),
        n_shards=int(n_shards),
    )

try:
    from numba import njit, prange, set_num_threads

    HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback path tested without numba dependency
    HAS_NUMBA = False
    njit = None
    prange = range
    set_num_threads = None


if HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _simulate_kernel_parallel(
        donor_haps: np.ndarray,
        switch_probs: np.ndarray,
        init_donors: np.ndarray,
        u: np.ndarray,
        replacement: np.ndarray,
    ) -> np.ndarray:
        n_haps = init_donors.shape[0]
        n_markers = donor_haps.shape[1]
        n_donor_haps = donor_haps.shape[0]
        out = np.empty((n_haps, n_markers), dtype=np.uint8)

        for h in prange(n_haps):
            donor_idx = init_donors[h]
            out[h, 0] = donor_haps[donor_idx, 0]
            for j in range(1, n_markers):
                if u[h, j - 1] < switch_probs[j - 1]:
                    cand = replacement[h, j - 1]
                    if cand == donor_idx:
                        cand = (cand + 1) % n_donor_haps
                    donor_idx = cand
                out[h, j] = donor_haps[donor_idx, j]
        return out


    @njit(fastmath=True)
    def _simulate_kernel_serial(
        donor_haps: np.ndarray,
        switch_probs: np.ndarray,
        init_donors: np.ndarray,
        u: np.ndarray,
        replacement: np.ndarray,
    ) -> np.ndarray:
        n_haps = init_donors.shape[0]
        n_markers = donor_haps.shape[1]
        n_donor_haps = donor_haps.shape[0]
        out = np.empty((n_haps, n_markers), dtype=np.uint8)

        for h in range(n_haps):
            donor_idx = init_donors[h]
            out[h, 0] = donor_haps[donor_idx, 0]
            for j in range(1, n_markers):
                if u[h, j - 1] < switch_probs[j - 1]:
                    cand = replacement[h, j - 1]
                    if cand == donor_idx:
                        cand = (cand + 1) % n_donor_haps
                    donor_idx = cand
                out[h, j] = donor_haps[donor_idx, j]
        return out



def _simulate_kernel_numpy(
    donor_haps: np.ndarray,
    switch_probs: np.ndarray,
    init_donors: np.ndarray,
    u: np.ndarray,
    replacement: np.ndarray,
) -> np.ndarray:
    n_haps = init_donors.shape[0]
    n_markers = donor_haps.shape[1]
    n_donor_haps = donor_haps.shape[0]
    out = np.empty((n_haps, n_markers), dtype=np.uint8)

    for h in range(n_haps):
        donor_idx = int(init_donors[h])
        out[h, 0] = donor_haps[donor_idx, 0]
        for j in range(1, n_markers):
            if u[h, j - 1] < switch_probs[j - 1]:
                cand = int(replacement[h, j - 1])
                if cand == donor_idx:
                    cand = (cand + 1) % n_donor_haps
                donor_idx = cand
            out[h, j] = donor_haps[donor_idx, j]
    return out



def _simulate_dispatch(
    donor_haps: np.ndarray,
    switch_probs: np.ndarray,
    init_donors: np.ndarray,
    u: np.ndarray,
    replacement: np.ndarray,
    use_parallel_kernel: bool,
    threads: int,
) -> np.ndarray:
    if HAS_NUMBA:
        if set_num_threads is not None:
            set_num_threads(max(1, threads))
        if use_parallel_kernel:
            return _simulate_kernel_parallel(donor_haps, switch_probs, init_donors, u, replacement)
        return _simulate_kernel_serial(donor_haps, switch_probs, init_donors, u, replacement)
    return _simulate_kernel_numpy(donor_haps, switch_probs, init_donors, u, replacement)



def simulate_haplotypes(
    donor_haps: np.ndarray,
    switch_probs: np.ndarray,
    n_haps: int,
    seed: int,
    threads: int,
    donor_hap_sampling_weights: np.ndarray | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_donor_haps = donor_haps.shape[0]
    n_markers = donor_haps.shape[1]
    donor_choice_probs = None
    if donor_hap_sampling_weights is not None:
        donor_choice_probs = np.asarray(donor_hap_sampling_weights, dtype=np.float64)
        if donor_choice_probs.shape != (n_donor_haps,):
            raise ValueError(
                "donor_hap_sampling_weights must have one value per donor haplotype; "
                f"got {donor_choice_probs.shape} expected ({n_donor_haps},)"
            )
        if np.any(donor_choice_probs < 0.0):
            raise ValueError("donor_hap_sampling_weights must be non-negative")
        total_weight = float(donor_choice_probs.sum())
        if total_weight <= 0.0:
            raise ValueError("donor_hap_sampling_weights must sum to a positive value")
        donor_choice_probs = donor_choice_probs / total_weight

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    out = np.empty((n_haps, n_markers), dtype=np.uint8)
    for start in range(0, n_haps, int(batch_size)):
        end = min(n_haps, start + int(batch_size))
        batch_n = end - start
        init_donors = rng.choice(
            n_donor_haps, size=batch_n, replace=True, p=donor_choice_probs
        ).astype(np.int32)
        if n_markers > 1:
            u = rng.random((batch_n, n_markers - 1), dtype=np.float32)
            replacement = rng.choice(
                n_donor_haps,
                size=(batch_n, n_markers - 1),
                replace=True,
                p=donor_choice_probs,
            ).astype(np.int32)
        else:
            u = np.zeros((batch_n, 0), dtype=np.float32)
            replacement = np.zeros((batch_n, 0), dtype=np.int32)

        out[start:end] = _simulate_dispatch(
            donor_haps=donor_haps,
            switch_probs=switch_probs,
            init_donors=init_donors,
            u=u,
            replacement=replacement,
            use_parallel_kernel=True,
            threads=max(1, int(threads)),
        )
    return out


@dataclass
class PopulationSimulationArtifacts:
    population: str
    sample_ids: list[str]
    vcf_path: Path | None
    hapstore_path: Path | None
    pgen_prefix: Path | None
    engine: str = "mosaic"
    tract_summary_path: Path | None = None
    tracts_tsv_path: Path | None = None
    ancestry_source_path: Path | None = None
    individual_ancestry_path: Path | None = None


@dataclass
class SimulationResult:
    chrom: int
    shard_id: int
    n_shards: int
    combined_vcf: Path | None
    combined_hapstore: Path | None
    combined_pgen_prefix: Path | None
    sample_manifest_path: Path
    population_outputs: list[PopulationSimulationArtifacts]
    elapsed_seconds: float
    stage_timings: dict[str, float]
    population_stage_timings: dict[str, dict[str, float]]


@dataclass
class SimulationBatchResult:
    chrom: int
    n_shards: int
    sample_manifest_paths: list[Path]
    shard_results: list[SimulationResult]
    elapsed_seconds: float
    stage_timings: dict[str, float]
    task_count: int

    @property
    def sample_manifest_path(self) -> Path:
        if len(self.shard_results) != 1:
            raise AttributeError("sample_manifest_path is only available for n_shards=1")
        return self.shard_results[0].sample_manifest_path

    @property
    def combined_vcf(self) -> Path | None:
        if len(self.shard_results) != 1:
            raise AttributeError("combined_vcf is only available for n_shards=1")
        return self.shard_results[0].combined_vcf

    @property
    def combined_hapstore(self) -> Path | None:
        if len(self.shard_results) != 1:
            raise AttributeError("combined_hapstore is only available for n_shards=1")
        return self.shard_results[0].combined_hapstore

    @property
    def combined_pgen_prefix(self) -> Path | None:
        if len(self.shard_results) != 1:
            raise AttributeError("combined_pgen_prefix is only available for n_shards=1")
        return self.shard_results[0].combined_pgen_prefix

    @property
    def population_outputs(self) -> list[PopulationSimulationArtifacts]:
        if len(self.shard_results) != 1:
            raise AttributeError("population_outputs is only available for n_shards=1")
        return self.shard_results[0].population_outputs

    @property
    def population_stage_timings(self) -> dict[str, dict[str, float]]:
        if len(self.shard_results) != 1:
            raise AttributeError("population_stage_timings is only available for n_shards=1")
        return self.shard_results[0].population_stage_timings


@dataclass
class _PopulationWorkItem:
    pop_name: str
    donors: list[str]
    panel_source_kind: str
    panel_source_path: Path
    map_path: Path
    sample_ids: list[str]
    pop_n: int
    seed: int
    shard_id: int
    n_shards: int
    engine: str = "mosaic"
    msprime_source_populations: list[str] | None = None
    msprime_source_weights: list[float] | None = None
    msprime_admixture_time_generations: float | None = None
    msprime_end_time_generations: float = 10.0
    msprime_write_full_tract_tsv: bool = False
    msprime_write_ancestry_source_zarr: bool = False


_TASK_PANEL_BY_POP: dict[str, Any] = {}
_TASK_SWITCH_PROBS_BY_POP: dict[str, np.ndarray] = {}
_TASK_THREADS_PER_TASK: int = 1
_TASK_HAPLOTYPE_BATCH_SIZE: int = 128


def _init_pool_worker_logging(queue: Any, level: int) -> None:
    install_queue_handler(queue=queue, level=int(level), reset_handlers=True, logger_name=None)


def _init_pool_worker_with_state(
    queue: Any,
    level: int,
    panel_by_pop: dict[str, Any],
    switch_probs_by_pop: dict[str, np.ndarray],
    threads_per_task: int,
    haplotype_batch_size: int,
) -> None:
    _init_pool_worker_logging(queue, level)
    _init_task_worker(
        panel_by_pop=panel_by_pop,
        switch_probs_by_pop=switch_probs_by_pop,
        threads_per_task=threads_per_task,
        haplotype_batch_size=haplotype_batch_size,
    )


def _init_task_worker(
    panel_by_pop: dict[str, Any],
    switch_probs_by_pop: dict[str, np.ndarray],
    threads_per_task: int,
    haplotype_batch_size: int,
) -> None:
    global _TASK_PANEL_BY_POP
    global _TASK_SWITCH_PROBS_BY_POP
    global _TASK_THREADS_PER_TASK
    global _TASK_HAPLOTYPE_BATCH_SIZE

    _TASK_PANEL_BY_POP = panel_by_pop
    _TASK_SWITCH_PROBS_BY_POP = switch_probs_by_pop
    _TASK_THREADS_PER_TASK = int(threads_per_task)
    _TASK_HAPLOTYPE_BATCH_SIZE = int(haplotype_batch_size)


def _simulate_population_shard_task(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        pop_name,
        sample_ids,
        pop_n,
        seed,
        chrom_dir_str,
        work_dir_str,
        chrom,
        shard_id,
        n_shards,
        write_phased_vcf,
        write_hapstore,
        hapstore_chunk_variant_size,
        hapstore_chunk_haplotype_size,
        hapstore_chunk_sample_size,
        shard_hapstore_compression_level,
        hapstore_store_genotypes,
        write_pgen,
        hapstore_vcf_plink2_location,
        hapstore_vcf_keep_intermediate_pgen,
        hapstore_vcf_n_shards,
    ) = args

    panel = _TASK_PANEL_BY_POP[pop_name]
    switch_probs = _TASK_SWITCH_PROBS_BY_POP[pop_name]
    final_chrom_dir = Path(chrom_dir_str)
    chrom_dir = Path(work_dir_str)

    timings: dict[str, float] = {}
    with telemetry_span(
        logger,
        "simulate-pool.task",
        chrom=int(chrom),
        pop=str(pop_name),
        shard=int(shard_id),
        n_shards=int(n_shards),
        n_diploids=int(pop_n),
    ):
        with telemetry_span(logger, "simulate_haplotypes", chrom=int(chrom), pop=str(pop_name), shard=int(shard_id)):
            t_sim = perf_counter()
            synthetic_haps = simulate_haplotypes(
                donor_haps=panel.donor_haps,
                switch_probs=switch_probs,
                n_haps=2 * pop_n,
                seed=seed,
                threads=_TASK_THREADS_PER_TASK,
                donor_hap_sampling_weights=panel.donor_hap_sampling_weights,
                batch_size=_TASK_HAPLOTYPE_BATCH_SIZE,
            )
            timings["simulate_haplotypes_sec"] = perf_counter() - t_sim

        pop_vcf: Path | None = None

        pop_hapstore = None
        if write_hapstore:
            with telemetry_span(logger, "write_pop_hapstore", chrom=int(chrom), pop=str(pop_name), shard=int(shard_id)):
                t_hapstore = perf_counter()
                pop_hapstore = chrom_dir / f"{pop_name}.chr{chrom}.hapstore.zarr"
                _ingest_arrays_with_retry(
                    output_zarr=pop_hapstore,
                    chrom=f"chr{chrom}",
                    haplotypes=synthetic_haps,
                    variant_pos=panel.variants.pos,
                    sample_ids=sample_ids,
                    ref=panel.variants.ref,
                    alt=panel.variants.alt,
                    vid=panel.variants.vid,
                    store_genotypes=bool(hapstore_store_genotypes),
                    chunk_variant_size=int(hapstore_chunk_variant_size),
                    chunk_haplotype_size=(
                        None if hapstore_chunk_haplotype_size is None else int(hapstore_chunk_haplotype_size)
                    ),
                    chunk_sample_size=(None if hapstore_chunk_sample_size is None else int(hapstore_chunk_sample_size)),
                    compression_level=int(shard_hapstore_compression_level),
                    attrs={"source": "population_amplifier"},
                )
                timings["write_pop_hapstore_sec"] = perf_counter() - t_hapstore

        if write_phased_vcf:
            pop_vcf = chrom_dir / f"{pop_name}.chr{chrom}.vcf.gz"
            with telemetry_span(logger, "write_pop_vcf", chrom=int(chrom), pop=str(pop_name), shard=int(shard_id)):
                t_write = perf_counter()
                pop_vcf = _write_vcf_from_hapstore(
                    output_vcf=pop_vcf,
                    chrom=int(chrom),
                    hapstore_path=pop_hapstore,
                    plink2_location=str(hapstore_vcf_plink2_location)
                    if hapstore_vcf_plink2_location is not None
                    else None,
                    keep_intermediate_pgen=bool(hapstore_vcf_keep_intermediate_pgen),
                    n_shards=int(hapstore_vcf_n_shards),
                )
                timings["write_pop_vcf_sec"] = perf_counter() - t_write

        pop_pgen_prefix = None
        if write_pgen:
            with telemetry_span(logger, "write_pop_pgen", chrom=int(chrom), pop=str(pop_name), shard=int(shard_id)):
                t_pgen = perf_counter()
                pop_pgen_prefix = chrom_dir / f"{pop_name}.chr{chrom}"
                pop_pgen_prefix = _write_pgen_from_hapstore(out_prefix=pop_pgen_prefix, chrom=int(chrom), hapstore_path=pop_hapstore)
                timings["write_pop_pgen_sec"] = perf_counter() - t_pgen

    if chrom_dir != final_chrom_dir:
        timings["sync_scratch_outputs_sec"] = _copy_scratch_tree(chrom_dir, final_chrom_dir)

        def _remap(path: Path | None) -> Path | None:
            if path is None:
                return None
            return final_chrom_dir / path.relative_to(chrom_dir)

        pop_vcf = _remap(pop_vcf)
        pop_hapstore = _remap(pop_hapstore)
        pop_pgen_prefix = _remap(pop_pgen_prefix)
        shutil.rmtree(chrom_dir)

    return {
        "population": pop_name,
        "sample_ids": sample_ids,
        "shard_id": int(shard_id),
        "n_shards": int(n_shards),
        "vcf_path": str(pop_vcf) if (write_phased_vcf and pop_vcf is not None) else None,
        "hapstore_path": str(pop_hapstore) if pop_hapstore is not None else None,
        "pgen_prefix": str(pop_pgen_prefix) if pop_pgen_prefix is not None else None,
        "timings": timings,
    }


def _preload_panel_unions(
    chrom: int,
    union_donors_by_source: dict[tuple[str, Path], set[str]],
    donor_io_workers: int,
    cache_dir: Path,
) -> tuple[dict[tuple[str, Path], Any], int, int]:
    panel_union_cache: dict[tuple[str, Path], Any] = {}
    cache_hits = 0
    cache_misses = 0
    if not union_donors_by_source:
        return panel_union_cache, cache_hits, cache_misses

    jobs = [
        (source_kind, source_path, sorted(donors))
        for (source_kind, source_path), donors in union_donors_by_source.items()
    ]

    def _load_one(source_kind: str, source_path: Path, donor_ids: list[str]) -> tuple[Any, bool]:
        panel, _, cache_hit = load_donor_haplotypes_from_source(
            source_path=source_path,
            source_kind=source_kind,
            chrom=chrom,
            donor_ids=donor_ids,
            cache_dir=cache_dir,
            ingest_workers=max(1, int(donor_io_workers)),
        )
        return panel, cache_hit

    max_workers = min(max(1, donor_io_workers), len(jobs))
    if max_workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_source = {
                pool.submit(_load_one, source_kind, source_path, donor_ids): (source_kind, source_path)
                for source_kind, source_path, donor_ids in jobs
            }
            for future in as_completed(future_to_source):
                source_key = future_to_source[future]
                panel, cache_hit = future.result()
                panel_union_cache[source_key] = panel
                if cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
    else:
        for source_kind, source_path, donor_ids in jobs:
            panel, cache_hit = _load_one(source_kind, source_path, donor_ids)
            panel_union_cache[(source_kind, source_path)] = panel
            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
    return panel_union_cache, cache_hits, cache_misses


def _build_shard_result(
    *,
    config: RunConfig,
    chrom: int,
    shard_id: int,
    n_shards: int,
    sample_manifest_path: Path,
    population_outputs: list[PopulationSimulationArtifacts],
    stage_timings: dict[str, float],
    population_stage_timings: dict[str, dict[str, float]],
    elapsed_seconds: float,
) -> SimulationResult:
    shard_dir = config.run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}"
    combined_hapstore = shard_dir / f"combined.chr{chrom}.hapstore.zarr"
    combined_vcf = shard_dir / f"combined.chr{chrom}.vcf.gz"
    combined_pgen_prefix = shard_dir / f"combined.chr{chrom}"
    return SimulationResult(
        chrom=int(chrom),
        shard_id=int(shard_id),
        n_shards=int(n_shards),
        combined_vcf=combined_vcf if combined_vcf.exists() else None,
        combined_hapstore=combined_hapstore if combined_hapstore.exists() else None,
        combined_pgen_prefix=combined_pgen_prefix if Path(f"{combined_pgen_prefix}.pgen").exists() else None,
        sample_manifest_path=sample_manifest_path,
        population_outputs=population_outputs,
        elapsed_seconds=float(elapsed_seconds),
        stage_timings=stage_timings,
        population_stage_timings=population_stage_timings,
    )


def _build_batch_shard_results(
    *,
    config: RunConfig,
    chrom: int,
    n_shards: int,
    sample_manifest_by_shard: dict[int, Path],
    task_results: list[dict[str, Any]],
    batch_stage_timings: dict[str, float],
    batch_elapsed_seconds: float,
) -> list[SimulationResult]:
    grouped_results: dict[int, list[dict[str, Any]]] = {}
    for result in task_results:
        grouped_results.setdefault(int(result["shard_id"]), []).append(result)

    shard_results: list[SimulationResult] = []
    for shard_id in range(int(n_shards)):
        manifest_path = sample_manifest_by_shard.get(int(shard_id))
        shard_task_results = grouped_results.get(int(shard_id), [])
        if manifest_path is None and not shard_task_results:
            continue

        population_outputs = [
            PopulationSimulationArtifacts(
                population=str(result["population"]),
                sample_ids=list(result["sample_ids"]),
                vcf_path=None if result.get("vcf_path") is None else Path(str(result["vcf_path"])),
                hapstore_path=None
                if result.get("hapstore_path") is None
                else Path(str(result["hapstore_path"])),
                pgen_prefix=None
                if result.get("pgen_prefix") is None
                else Path(str(result["pgen_prefix"])),
            )
            for result in sorted(shard_task_results, key=lambda x: str(x["population"]))
        ]
        population_stage_timings = {
            str(result["population"]): {str(k): float(v) for k, v in dict(result.get("timings", {})).items()}
            for result in shard_task_results
        }
        shard_stage_timings: dict[str, float] = {}
        for result in shard_task_results:
            for key, elapsed in dict(result.get("timings", {})).items():
                shard_stage_timings[str(key)] = shard_stage_timings.get(str(key), 0.0) + float(elapsed)
        if int(n_shards) == 1:
            for key, value in batch_stage_timings.items():
                shard_stage_timings.setdefault(str(key), float(value))

        shard_elapsed = float(batch_elapsed_seconds if int(n_shards) == 1 else sum(shard_stage_timings.values()))
        if manifest_path is None:
            raise RuntimeError(f"missing sample manifest for shard {shard_id} on chrom {chrom}")
        shard_results.append(
            _build_shard_result(
                config=config,
                chrom=int(chrom),
                shard_id=int(shard_id),
                n_shards=int(n_shards),
                sample_manifest_path=manifest_path,
                population_outputs=population_outputs,
                stage_timings=shard_stage_timings,
                population_stage_timings=population_stage_timings,
                elapsed_seconds=shard_elapsed,
            )
        )

    return shard_results



def _simulate_chromosome_shard(
    config: RunConfig,
    chrom: int,
    shard_id: int = 0,
    n_shards: int = 1,
) -> SimulationResult:
    t0 = perf_counter()
    stage_timings: dict[str, float] = {}
    population_stage_timings: dict[str, dict[str, float]] = {}

    def _add_timing(key: str, elapsed: float) -> None:
        stage_timings[key] = stage_timings.get(key, 0.0) + elapsed

    t_manifest = perf_counter()
    manifest = load_manifest(config)
    _add_timing("load_manifest_sec", perf_counter() - t_manifest)
    threads = int(config.simulation.threads_per_task)
    donor_io_workers = int(config.simulation.donor_io_workers)
    cache_dir = config.donor_cache_root

    final_chrom_dir = ensure_dir(
        config.run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}"
    )
    scratch_chrom_dir = _scratch_dir(
        config,
        chrom=int(chrom),
        shard_id=int(shard_id),
        n_shards=int(n_shards),
    )
    chrom_dir = scratch_chrom_dir or final_chrom_dir
    manifest_dir = ensure_dir(config.run_root / "manifests")

    combined_sample_ids: list[str] = []
    variant_table = None
    sample_rows: list[dict[str, str | int]] = []
    population_outputs: list[PopulationSimulationArtifacts] = []
    write_combined = config.outputs.write_combined_shard_output
    write_hapstore = bool(config.outputs.write_hapstore)

    work_items: list[_PopulationWorkItem] = []
    union_donors_by_source: dict[tuple[str, Path], set[str]] = {}
    for pop in config.populations:
        if pop.donor_only:
            continue
        panel_source_kind, panel_source_path = config.resolve_donor_panel_source(
            chrom=int(chrom),
            population=str(pop.name),
        )
        map_path = Path(config.input.genetic_map_pattern.format(chrom=chrom, population=pop.name))
        donors = manifest.donors_by_population.get(pop.name, [])
        msprime_source_populations: list[str] | None = None
        msprime_source_weights: list[float] | None = None
        msprime_admixture_time_generations: float | None = None
        msprime_end_time_generations = 10.0
        msprime_write_full_tract_tsv = False
        msprime_write_ancestry_source_zarr = False
        if pop.engine == "msprime_tracts":
            assert pop.msprime is not None
            msprime_source_populations = [str(source.population) for source in pop.msprime.sources]
            msprime_source_weights = [float(source.weight) for source in pop.msprime.sources]
            msprime_admixture_time_generations = (
                float(pop.msprime.admixture_time_generations)
                if pop.msprime.admixture_time_generations is not None
                else None
            )
            msprime_end_time_generations = float(pop.msprime.end_time_generations)
            msprime_write_full_tract_tsv = bool(pop.msprime.write_full_tract_tsv)
            msprime_write_ancestry_source_zarr = bool(pop.msprime.write_ancestry_source_zarr)
            if msprime_source_populations:
                first_source = msprime_source_populations[0]
                panel_source_kind, panel_source_path = config.resolve_donor_panel_source(
                    chrom=int(chrom),
                    population=str(first_source),
                )
                map_path = Path(config.input.genetic_map_pattern.format(chrom=chrom, population=first_source))

        pop_start, pop_n = shard_bounds(pop.n_diploids, shard_id=shard_id, n_shards=n_shards)
        sample_ids = [f"{pop.name}_syn{pop_start + i + 1}" for i in range(pop_n)]
        seed = derived_seed(config.simulation.random_seed, pop.name, chrom, shard_id)
        work_items.append(
            _PopulationWorkItem(
                pop_name=pop.name,
                donors=donors,
                panel_source_kind=panel_source_kind,
                panel_source_path=panel_source_path,
                map_path=map_path,
                sample_ids=sample_ids,
                pop_n=pop_n,
                seed=seed,
                shard_id=shard_id,
                n_shards=n_shards,
                engine=str(pop.engine),
                msprime_source_populations=msprime_source_populations,
                msprime_source_weights=msprime_source_weights,
                msprime_admixture_time_generations=msprime_admixture_time_generations,
                msprime_end_time_generations=msprime_end_time_generations,
                msprime_write_full_tract_tsv=msprime_write_full_tract_tsv,
                msprime_write_ancestry_source_zarr=msprime_write_ancestry_source_zarr,
            )
        )
        if pop_n > 0:
            if pop.engine == "msprime_tracts":
                assert msprime_source_populations is not None
                for source_name in msprime_source_populations:
                    source_kind, source_path = config.resolve_donor_panel_source(
                        chrom=int(chrom),
                        population=str(source_name),
                    )
                    source_donors = manifest.donors_by_population.get(source_name, [])
                    union_donors_by_source.setdefault((source_kind, source_path), set()).update(source_donors)
            else:
                union_donors_by_source.setdefault((panel_source_kind, panel_source_path), set()).update(donors)

    t_preload_panels = perf_counter()
    logger.info(
        "simulate_chromosome preload donor panels start chrom=%s shard=%s/%s panel_files=%s donor_counts_by_panel=%s",
        chrom,
        shard_id,
        n_shards,
        len(union_donors_by_source),
        json.dumps(
            {f"{kind}:{path}": len(donors) for (kind, path), donors in union_donors_by_source.items()},
            sort_keys=True,
        ),
    )
    panel_union_cache, cache_hits, cache_misses = _preload_panel_unions(
        chrom=chrom,
        union_donors_by_source=union_donors_by_source,
        donor_io_workers=donor_io_workers,
        cache_dir=cache_dir,
    )
    _add_timing("preload_panel_unions_sec", perf_counter() - t_preload_panels)
    stage_timings["preload_panel_cache_hits"] = float(cache_hits)
    stage_timings["preload_panel_cache_misses"] = float(cache_misses)
    logger.info(
        "simulate_chromosome preload donor panels done chrom=%s shard=%s/%s elapsed=%.2fs cache_hits=%s cache_misses=%s",
        chrom,
        shard_id,
        n_shards,
        stage_timings.get("preload_panel_unions_sec", 0.0),
        cache_hits,
        cache_misses,
    )

    map_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    msprime_panel_cache: dict[str, Any] = {}
    logger.info(
        "simulate_chromosome plan chrom=%s shard=%s/%s configured_pops=%s active_pops=%s",
        chrom,
        shard_id,
        n_shards,
        len(work_items),
        sum(1 for x in work_items if x.pop_n > 0),
    )
    for pop_idx, item in enumerate(work_items, start=1):
        pop_t0 = perf_counter()
        pop_stage: dict[str, float] = {}
        if item.pop_n == 0:
            population_stage_timings[item.pop_name] = pop_stage
            logger.info(
                "simulate_chromosome population skip chrom=%s shard=%s/%s pop=%s index=%s/%s reason=empty_shard",
                chrom,
                shard_id,
                n_shards,
                item.pop_name,
                pop_idx,
                len(work_items),
            )
            continue

        logger.info(
            "simulate_chromosome population start chrom=%s shard=%s/%s pop=%s index=%s/%s engine=%s n_diploids=%s donors=%s",
            chrom,
            shard_id,
            n_shards,
            item.pop_name,
            pop_idx,
            len(work_items),
            item.engine,
            item.pop_n,
            len(item.donors),
        )

        pop_stage["engine_is_msprime_tracts"] = 1.0 if item.engine == "msprime_tracts" else 0.0
        pop_stage["engine_is_mosaic"] = 1.0 if item.engine == "mosaic" else 0.0
        if item.engine == "msprime_tracts":
            _add_timing("engine_msprime_tracts_population_count", 1.0)
        else:
            _add_timing("engine_mosaic_population_count", 1.0)
        variants = None
        tract_summary_path: Path | None = None
        tracts_tsv_path: Path | None = None
        ancestry_source_path: Path | None = None
        individual_ancestry_path: Path | None = None
        if item.engine == "msprime_tracts":
            t_source = perf_counter()
            assert item.msprime_source_populations is not None
            assert item.msprime_source_weights is not None
            source_order = [str(x) for x in item.msprime_source_populations]
            logger.info(
                "simulate_chromosome msprime source load start chrom=%s shard=%s/%s pop=%s sources=%s",
                chrom,
                shard_id,
                n_shards,
                item.pop_name,
                json.dumps(source_order),
            )
            source_panels: dict[str, Any] = {}
            for source_name in source_order:
                if source_name in msprime_panel_cache:
                    source_panels[source_name] = msprime_panel_cache[source_name]
                    continue
                source_kind, source_path = config.resolve_donor_panel_source(
                    chrom=int(chrom),
                    population=str(source_name),
                )
                source_donors = manifest.donors_by_population[source_name]
                source_weights = manifest.donor_weights_by_population.get(source_name, {})
                source_panels[source_name] = subset_panel_donors(
                    panel_union_cache[(source_kind, source_path)],
                    source_donors,
                    donor_weights=[float(source_weights.get(sid, 0.0)) for sid in source_donors],
                )
                msprime_panel_cache[source_name] = source_panels[source_name]
            elapsed_source = perf_counter() - t_source
            pop_stage["load_msprime_source_haps_sec"] = elapsed_source
            _add_timing("load_msprime_source_haps_sec", elapsed_source)

            first_source = source_order[0]
            first_source_panel = source_panels[first_source]
            variants = first_source_panel.variants
            source_donor_counts = {
                source_name: len(manifest.donors_by_population.get(source_name, []))
                for source_name in source_order
            }
            logger.info(
                "simulate_chromosome msprime source load done chrom=%s shard=%s/%s pop=%s elapsed=%.2fs markers=%s source_donor_counts=%s",
                chrom,
                shard_id,
                n_shards,
                item.pop_name,
                elapsed_source,
                int(variants.pos.shape[0]),
                json.dumps(source_donor_counts, sort_keys=True),
            )

            t_map = perf_counter()
            if item.map_path not in map_cache:
                map_cache[item.map_path] = load_genetic_map(item.map_path)
            map_bp, map_cm = map_cache[item.map_path]
            elapsed_map = perf_counter() - t_map
            pop_stage["load_msprime_map_sec"] = elapsed_map
            _add_timing("load_msprime_map_sec", elapsed_map)

            t_simulate = perf_counter()
            logger.info(
                "simulate_chromosome msprime tract simulation start chrom=%s shard=%s/%s pop=%s n_haps=%s seed=%s",
                chrom,
                shard_id,
                n_shards,
                item.pop_name,
                2 * item.pop_n,
                item.seed,
            )
            msprime_result = simulate_msprime_tract_haplotypes(
                source_order=source_order,
                source_weights=[float(x) for x in item.msprime_source_weights],
                source_panels=source_panels,
                marker_positions=variants.pos,
                map_bp=map_bp,
                map_cm=map_cm,
                n_haps=2 * item.pop_n,
                seed=int(item.seed),
                chrom=int(chrom),
                population_name=str(item.pop_name),
                admixture_time_generations=item.msprime_admixture_time_generations,
                end_time_generations=float(item.msprime_end_time_generations),
                copy_workers=max(1, int(threads)),
            )
            synthetic_haps = msprime_result.haplotypes
            elapsed_simulate = perf_counter() - t_simulate
            pop_stage["simulate_msprime_tracts_sec"] = elapsed_simulate
            _add_timing("simulate_msprime_tracts_sec", elapsed_simulate)
            logger.info(
                "simulate_chromosome msprime tract simulation done chrom=%s shard=%s/%s pop=%s elapsed=%.2fs tract_count=%s mean_tract_length_bp=%s",
                chrom,
                shard_id,
                n_shards,
                item.pop_name,
                elapsed_simulate,
                int(msprime_result.tract_summary.get("tract_count", 0)),
                msprime_result.tract_summary.get("mean_tract_length_bp"),
            )

            t_diag = perf_counter()
            tract_summary_path = chrom_dir / f"{item.pop_name}.chr{chrom}.tract_summary.json"
            summary_payload = dict(msprime_result.tract_summary)
            summary_payload["shard_id"] = int(item.shard_id)
            summary_payload["n_shards"] = int(item.n_shards)
            summary_payload["n_diploids"] = int(item.pop_n)
            with open(tract_summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary_payload, handle, indent=2, sort_keys=True)
            individual_ancestry_path = chrom_dir / f"{item.pop_name}.chr{chrom}.individual_ancestry.tsv"
            individual_rows: list[dict[str, Any]] = []
            for row in msprime_result.individual_ancestry_rows:
                enriched = dict(row)
                individual_idx = int(enriched.get("individual_index", -1))
                if 0 <= individual_idx < len(item.sample_ids):
                    enriched["sample_id"] = str(item.sample_ids[individual_idx])
                individual_rows.append(enriched)
            pd.DataFrame.from_records(individual_rows).to_csv(
                individual_ancestry_path,
                sep="\t",
                index=False,
            )
            if item.msprime_write_full_tract_tsv:
                tracts_tsv_path = chrom_dir / f"{item.pop_name}.chr{chrom}.tracts.tsv"
                pd.DataFrame.from_records(msprime_result.tract_rows).to_csv(
                    tracts_tsv_path,
                    sep="\t",
                    index=False,
                )
            if item.msprime_write_ancestry_source_zarr:
                ancestry_source_path = chrom_dir / f"{item.pop_name}.chr{chrom}.local_ancestry.zarr"
                write_local_ancestry_source_from_tract_rows(
                    ancestry_source_path,
                    sample_ids=item.sample_ids,
                    tract_rows=msprime_result.tract_rows,
                    ancestry_order=source_order,
                    chrom=int(chrom),
                    metadata={
                        "population": str(item.pop_name),
                        "shard_id": int(item.shard_id),
                        "n_shards": int(item.n_shards),
                        "n_diploids": int(item.pop_n),
                        "engine": "msprime_tracts",
                    },
                )
            elapsed_diag = perf_counter() - t_diag
            pop_stage["write_msprime_tract_diagnostics_sec"] = elapsed_diag
            _add_timing("write_msprime_tract_diagnostics_sec", elapsed_diag)
        else:
            t_subset = perf_counter()
            panel = subset_panel_donors(
                panel_union_cache[(item.panel_source_kind, item.panel_source_path)],
                item.donors,
                donor_weights=[
                    float(manifest.donor_weights_by_population.get(item.pop_name, {}).get(sid, 0.0))
                    for sid in item.donors
                ],
            )
            elapsed_subset = perf_counter() - t_subset
            pop_stage["load_donor_haps_sec"] = elapsed_subset
            _add_timing("load_donor_haps_sec", elapsed_subset)
            variants = panel.variants
            logger.info(
                "simulate_chromosome mosaic donor load done chrom=%s shard=%s/%s pop=%s elapsed=%.2fs donor_haps=%s markers=%s",
                chrom,
                shard_id,
                n_shards,
                item.pop_name,
                elapsed_subset,
                int(panel.donor_haps.shape[0]),
                int(panel.donor_haps.shape[1]),
            )

            t_map = perf_counter()
            if item.map_path not in map_cache:
                map_cache[item.map_path] = load_genetic_map(item.map_path)
            map_bp, map_cm = map_cache[item.map_path]
            switch_probs = switch_probs_from_positions(
                marker_positions=panel.variants.pos,
                map_bp=map_bp,
                map_cm=map_cm,
                mean_tract_cm=config.simulation.mean_tract_cm,
            )
            elapsed_map = perf_counter() - t_map
            pop_stage["load_map_and_switch_probs_sec"] = elapsed_map
            _add_timing("load_map_and_switch_probs_sec", elapsed_map)

            t_simulate = perf_counter()
            synthetic_haps = simulate_haplotypes(
                donor_haps=panel.donor_haps,
                switch_probs=switch_probs,
                n_haps=2 * item.pop_n,
                seed=item.seed,
                threads=threads,
                donor_hap_sampling_weights=panel.donor_hap_sampling_weights,
                batch_size=int(config.simulation.haplotype_batch_size),
            )
            elapsed_simulate = perf_counter() - t_simulate
            pop_stage["simulate_haplotypes_sec"] = elapsed_simulate
            _add_timing("simulate_haplotypes_sec", elapsed_simulate)

        if variants is None:
            raise RuntimeError(f"Failed to resolve variants for population {item.pop_name}")
        if variant_table is None:
            variant_table = variants
        elif not all(
            (
                np.array_equal(variants.pos, variant_table.pos),
                np.array_equal(variants.ref, variant_table.ref),
                np.array_equal(variants.alt, variant_table.alt),
                np.array_equal(variants.vid, variant_table.vid),
            )
        ):
            raise ValueError(
                f"Population {item.pop_name} has marker identity mismatch on chr{chrom}; "
                "positions, REF, ALT, and source IDs must align"
            )

        pop_vcf: Path | None = None

        pop_hapstore: Path | None = None
        if write_hapstore:
            t_pop_hapstore = perf_counter()
            pop_hapstore = chrom_dir / f"{item.pop_name}.chr{chrom}.hapstore.zarr"
            _ingest_arrays_with_retry(
                output_zarr=pop_hapstore,
                chrom=f"chr{chrom}",
                haplotypes=synthetic_haps,
                variant_pos=variants.pos,
                sample_ids=item.sample_ids,
                ref=variants.ref,
                alt=variants.alt,
                vid=variants.vid,
                store_genotypes=bool(config.outputs.hapstore_store_genotypes),
                chunk_variant_size=int(config.outputs.hapstore_chunk_variant_size),
                chunk_haplotype_size=(
                    None
                    if config.outputs.hapstore_chunk_haplotype_size is None
                    else int(config.outputs.hapstore_chunk_haplotype_size)
                ),
                chunk_sample_size=(
                    None
                    if config.outputs.hapstore_chunk_sample_size is None
                    else int(config.outputs.hapstore_chunk_sample_size)
                ),
                compression_level=int(
                    config.outputs.shard_hapstore_compression_level
                    if config.outputs.shard_hapstore_compression_level is not None
                    else config.outputs.hapstore_compression_level
                ),
                attrs={"source": "population_amplifier"},
            )
            elapsed_pop_hapstore = perf_counter() - t_pop_hapstore
            pop_stage["write_pop_hapstore_sec"] = elapsed_pop_hapstore
            _add_timing("write_pop_hapstore_sec", elapsed_pop_hapstore)

        if config.outputs.write_phased_vcf:
            pop_vcf = chrom_dir / f"{item.pop_name}.chr{chrom}.vcf.gz"
            t_write_pop_vcf = perf_counter()
            pop_vcf = _write_vcf_from_hapstore(
                output_vcf=pop_vcf,
                chrom=int(chrom),
                hapstore_path=pop_hapstore,
                plink2_location=config.outputs.hapstore_vcf_plink2_location,
                keep_intermediate_pgen=bool(config.outputs.hapstore_vcf_keep_intermediate_pgen),
                n_shards=int(config.outputs.hapstore_vcf_n_shards),
            )
            elapsed_write_pop_vcf = perf_counter() - t_write_pop_vcf
            pop_stage["write_pop_vcf_sec"] = elapsed_write_pop_vcf
            _add_timing("write_pop_vcf_sec", elapsed_write_pop_vcf)

        pop_pgen_prefix = None
        if config.outputs.write_pgen:
            t_pop_pgen = perf_counter()
            pop_pgen_prefix = chrom_dir / f"{item.pop_name}.chr{chrom}"
            pop_pgen_prefix = _write_pgen_from_hapstore(
                out_prefix=pop_pgen_prefix,
                chrom=int(chrom),
                hapstore_path=pop_hapstore,
            )
            elapsed_pop_pgen = perf_counter() - t_pop_pgen
            pop_stage["write_pop_pgen_sec"] = elapsed_pop_pgen
            _add_timing("write_pop_pgen_sec", elapsed_pop_pgen)

        pop_vcf_out: Path | None = pop_vcf if (config.outputs.write_phased_vcf and pop_vcf is not None) else None

        population_outputs.append(
            PopulationSimulationArtifacts(
                population=item.pop_name,
                sample_ids=item.sample_ids,
                vcf_path=pop_vcf_out,
                hapstore_path=pop_hapstore,
                pgen_prefix=pop_pgen_prefix,
                engine=str(item.engine),
                tract_summary_path=tract_summary_path,
                tracts_tsv_path=tracts_tsv_path,
                ancestry_source_path=ancestry_source_path,
                individual_ancestry_path=individual_ancestry_path,
            )
        )

        combined_sample_ids.extend(item.sample_ids)
        for sid in item.sample_ids:
            sample_rows.append(
                {
                    "sample_order": len(sample_rows),
                    "sample_id": sid,
                    "population": item.pop_name,
                    "chrom": chrom,
                    "shard_id": shard_id,
                    "n_shards": n_shards,
                }
            )
        population_stage_timings[item.pop_name] = pop_stage
        logger.info(
            "simulate_chromosome population done chrom=%s shard=%s/%s pop=%s index=%s/%s elapsed=%.2fs outputs=%s timings=%s",
            chrom,
            shard_id,
            n_shards,
            item.pop_name,
            pop_idx,
            len(work_items),
            perf_counter() - pop_t0,
            json.dumps(
                {
                    "vcf": str(pop_vcf_out) if pop_vcf_out is not None else None,
                    "hapstore": str(pop_hapstore) if pop_hapstore is not None else None,
                    "pgen_prefix": str(pop_pgen_prefix) if pop_pgen_prefix is not None else None,
                    "tract_summary": str(tract_summary_path) if tract_summary_path is not None else None,
                    "ancestry_source": (
                        str(ancestry_source_path) if ancestry_source_path is not None else None
                    ),
                    "individual_ancestry": (
                        str(individual_ancestry_path) if individual_ancestry_path is not None else None
                    ),
                },
                sort_keys=True,
            ),
            json.dumps(pop_stage, sort_keys=True),
        )

    if not combined_sample_ids or variant_table is None:
        raise ValueError("No samples produced for this shard; check n_diploids/shard settings")

    combined_pgen_prefix = None
    combined_vcf_out: Path | None = None
    combined_hapstore_out: Path | None = None
    if write_combined:
        combined_vcf: Path | None = None

        if write_hapstore:
            t_write_combined_hapstore = perf_counter()
            combined_hapstore = chrom_dir / f"combined.chr{chrom}.hapstore.zarr"
            population_hapstores = [
                output.hapstore_path
                for output in population_outputs
                if output.hapstore_path is not None
            ]
            merge_chromosomes_sample_axis(
                input_stores=population_hapstores,
                output_store=combined_hapstore,
                chrom=f"chr{chrom}",
                output_mode="w",
                store_genotypes=bool(config.outputs.hapstore_store_genotypes),
                chunk_variant_size=int(config.outputs.hapstore_chunk_variant_size),
                chunk_haplotype_size=(
                    None
                    if config.outputs.hapstore_chunk_haplotype_size is None
                    else int(config.outputs.hapstore_chunk_haplotype_size)
                ),
                chunk_sample_size=(
                    None
                    if config.outputs.hapstore_chunk_sample_size is None
                    else int(config.outputs.hapstore_chunk_sample_size)
                ),
                compression_level=int(
                    config.outputs.shard_hapstore_compression_level
                    if config.outputs.shard_hapstore_compression_level is not None
                    else config.outputs.hapstore_compression_level
                ),
                attrs={"source": "population_amplifier"},
                n_workers=max(1, int(config.simulation.threads_per_task)),
            )
            _add_timing("write_combined_hapstore_sec", perf_counter() - t_write_combined_hapstore)
            combined_hapstore_out = combined_hapstore

        if config.outputs.write_phased_vcf:
            combined_vcf = chrom_dir / f"combined.chr{chrom}.vcf.gz"
            t_write_combined = perf_counter()
            combined_vcf = _write_vcf_from_hapstore(
                output_vcf=combined_vcf,
                chrom=int(chrom),
                hapstore_path=combined_hapstore_out,
                plink2_location=config.outputs.hapstore_vcf_plink2_location,
                keep_intermediate_pgen=bool(config.outputs.hapstore_vcf_keep_intermediate_pgen),
                n_shards=int(config.outputs.hapstore_vcf_n_shards),
            )
            _add_timing("write_combined_vcf_sec", perf_counter() - t_write_combined)
            combined_vcf_out = combined_vcf

        if config.outputs.write_pgen:
            t_combined_pgen = perf_counter()
            combined_pgen_prefix = chrom_dir / f"combined.chr{chrom}"
            combined_pgen_prefix = _write_pgen_from_hapstore(
                out_prefix=combined_pgen_prefix,
                chrom=int(chrom),
                hapstore_path=combined_hapstore_out,
            )
            _add_timing("write_combined_pgen_sec", perf_counter() - t_combined_pgen)

    if scratch_chrom_dir is not None:
        sync_elapsed = _copy_scratch_tree(scratch_chrom_dir, final_chrom_dir)
        _add_timing("sync_scratch_outputs_sec", sync_elapsed)

        def _remap(path: Path | None) -> Path | None:
            if path is None:
                return None
            return final_chrom_dir / path.relative_to(scratch_chrom_dir)

        for output in population_outputs:
            output.vcf_path = _remap(output.vcf_path)
            output.hapstore_path = _remap(output.hapstore_path)
            output.pgen_prefix = _remap(output.pgen_prefix)
            output.tract_summary_path = _remap(output.tract_summary_path)
            output.tracts_tsv_path = _remap(output.tracts_tsv_path)
            output.ancestry_source_path = _remap(output.ancestry_source_path)
            output.individual_ancestry_path = _remap(output.individual_ancestry_path)
        combined_vcf_out = _remap(combined_vcf_out)
        combined_hapstore_out = _remap(combined_hapstore_out)
        combined_pgen_prefix = _remap(combined_pgen_prefix)
        shutil.rmtree(scratch_chrom_dir)
        logger.info(
            "simulate_chromosome scratch sync complete chrom=%s shard=%s/%s source=%s destination=%s elapsed=%.2fs",
            chrom,
            shard_id,
            n_shards,
            scratch_chrom_dir,
            final_chrom_dir,
            sync_elapsed,
        )

    t_manifest_write = perf_counter()
    sample_manifest = pd.DataFrame(sample_rows)
    sample_manifest_path = manifest_dir / f"synthetic_samples_chr{chrom}_shard_{shard_id}_of_{n_shards}.tsv"
    sample_manifest.to_csv(sample_manifest_path, sep="\t", index=False)
    _add_timing("write_sample_manifest_sec", perf_counter() - t_manifest_write)

    elapsed = perf_counter() - t0
    stage_timings["total_sec"] = elapsed
    logger.info(
        "simulate_chromosome complete chrom=%s shard=%s/%s samples=%s elapsed=%.2fs write_combined=%s timings=%s",
        chrom,
        shard_id,
        n_shards,
        len(combined_sample_ids),
        elapsed,
        write_combined,
        json.dumps(stage_timings, sort_keys=True),
    )
    logger.info(
        "simulate_chromosome population timings chrom=%s shard=%s/%s timings=%s",
        chrom,
        shard_id,
        n_shards,
        json.dumps(population_stage_timings, sort_keys=True),
    )

    return SimulationResult(
        chrom=chrom,
        shard_id=shard_id,
        n_shards=n_shards,
        combined_vcf=combined_vcf_out,
        combined_hapstore=combined_hapstore_out,
        combined_pgen_prefix=combined_pgen_prefix,
        sample_manifest_path=sample_manifest_path,
        population_outputs=population_outputs,
        elapsed_seconds=elapsed,
        stage_timings=stage_timings,
        population_stage_timings=population_stage_timings,
    )


def simulate_chromosome_population_shard_pool(
    config: RunConfig,
    chrom: int,
    n_shards: int,
    task_workers: int,
    threads_per_task: int | None = None,
    task_start_method: str = "fork",
) -> SimulationBatchResult:
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")
    if task_workers <= 0:
        raise ValueError("task_workers must be > 0")
    if threads_per_task is None:
        threads_per_task = int(config.simulation.threads_per_task)
    if threads_per_task <= 0:
        raise ValueError("threads_per_task must be > 0")
    active_population_count = sum(not pop.donor_only for pop in config.populations)
    require_cpu_budget(
        workers=min(int(task_workers), max(1, int(n_shards) * active_population_count)),
        threads_per_worker=int(threads_per_task),
        context="amplifier simulation",
    )

    t0 = perf_counter()
    stage_timings: dict[str, float] = {}

    def _add_timing(key: str, elapsed: float) -> None:
        stage_timings[key] = stage_timings.get(key, 0.0) + elapsed

    with telemetry_span(logger, "simulate-pool.load_manifest", chrom=int(chrom), n_shards=int(n_shards)):
        t_manifest = perf_counter()
        manifest = load_manifest(config)
        _add_timing("load_manifest_sec", perf_counter() - t_manifest)

    donor_io_workers = int(config.simulation.donor_io_workers)
    cache_dir = config.donor_cache_root

    if any(pop.engine == "msprime_tracts" for pop in config.populations):
        logger.info(
            "simulate-pool detected msprime_tracts population(s) on chrom=%s; "
            "using shard-sequential simulate_chromosome fallback path",
            chrom,
        )
        sample_manifest_paths: list[Path] = []
        shard_results: list[SimulationResult] = []
        task_count = 0
        for pop in config.populations:
            if pop.donor_only:
                continue
            for shard_id in range(n_shards):
                _, pop_n = shard_bounds(pop.n_diploids, shard_id=shard_id, n_shards=n_shards)
                if pop_n > 0:
                    task_count += 1
        t_fallback = perf_counter()
        logger.info(
            "simulate-pool msprime fallback start chrom=%s n_shards=%s expected_tasks=%s",
            chrom,
            n_shards,
            task_count,
        )
        fallback_done = 0
        for shard_id in range(n_shards):
            t_shard = perf_counter()
            logger.info(
                "simulate-pool msprime fallback shard start chrom=%s shard=%s/%s",
                chrom,
                shard_id,
                n_shards,
            )
            shard_result = _simulate_chromosome_shard(
                config=config,
                chrom=int(chrom),
                shard_id=int(shard_id),
                n_shards=int(n_shards),
            )
            fallback_done += 1
            sample_manifest_paths.append(shard_result.sample_manifest_path)
            shard_results.append(shard_result)
            for key, elapsed in shard_result.stage_timings.items():
                if key == "total_sec":
                    continue
                _add_timing(key, float(elapsed))
            logger.info(
                "simulate-pool msprime fallback shard done chrom=%s shard=%s/%s elapsed=%.2fs completed=%s/%s shard_manifest=%s",
                chrom,
                shard_id,
                n_shards,
                perf_counter() - t_shard,
                fallback_done,
                n_shards,
                shard_result.sample_manifest_path,
            )
        _add_timing("simulate_pool_msprime_fallback_sec", perf_counter() - t_fallback)
        elapsed = perf_counter() - t0
        stage_timings["simulate_pool_msprime_fallback_used"] = 1.0
        stage_timings["total_sec"] = elapsed
        return SimulationBatchResult(
            chrom=int(chrom),
            n_shards=int(n_shards),
            sample_manifest_paths=sample_manifest_paths,
            shard_results=shard_results,
            elapsed_seconds=elapsed,
            stage_timings=stage_timings,
            task_count=int(task_count),
        )

    # Build full (population, shard) task grid up front.
    with telemetry_span(logger, "simulate-pool.build_task_grid", chrom=int(chrom), n_shards=int(n_shards)):
        t_build = perf_counter()
        work_items: list[_PopulationWorkItem] = []
        union_donors_by_source: dict[tuple[str, Path], set[str]] = {}
        for pop in config.populations:
            if pop.donor_only:
                continue
            donors = manifest.donors_by_population[pop.name]
            panel_source_kind, panel_source_path = config.resolve_donor_panel_source(
                chrom=int(chrom),
                population=str(pop.name),
            )
            map_path = Path(config.input.genetic_map_pattern.format(chrom=chrom, population=pop.name))
            union_donors_by_source.setdefault((panel_source_kind, panel_source_path), set()).update(donors)
            for shard_id in range(n_shards):
                pop_start, pop_n = shard_bounds(pop.n_diploids, shard_id=shard_id, n_shards=n_shards)
                sample_ids = [f"{pop.name}_syn{pop_start + i + 1}" for i in range(pop_n)]
                seed = derived_seed(config.simulation.random_seed, pop.name, chrom, shard_id)
                work_items.append(
                    _PopulationWorkItem(
                        pop_name=pop.name,
                        donors=donors,
                        panel_source_kind=panel_source_kind,
                        panel_source_path=panel_source_path,
                        map_path=map_path,
                        sample_ids=sample_ids,
                        pop_n=pop_n,
                        seed=seed,
                        shard_id=shard_id,
                        n_shards=n_shards,
                    )
                )
        _add_timing("build_task_grid_sec", perf_counter() - t_build)
    logger.info(
        "simulate-pool task grid chrom=%s n_populations=%s n_shards=%s task_count=%s",
        chrom,
        len(config.populations),
        n_shards,
        len(work_items),
    )

    # Load donor unions once and pre-subset per population once.
    with telemetry_span(logger, "simulate-pool.preload_panel_unions", chrom=int(chrom)):
        t_preload = perf_counter()
        panel_union_cache, cache_hits, cache_misses = _preload_panel_unions(
            chrom=chrom,
            union_donors_by_source=union_donors_by_source,
            donor_io_workers=donor_io_workers,
            cache_dir=cache_dir,
        )
        _add_timing("preload_panel_unions_sec", perf_counter() - t_preload)
    stage_timings["preload_panel_cache_hits"] = float(cache_hits)
    stage_timings["preload_panel_cache_misses"] = float(cache_misses)
    logger.info(
        "simulate-pool preload complete chrom=%s donor_panel_files=%s cache_hits=%s cache_misses=%s",
        chrom,
        len(panel_union_cache),
        cache_hits,
        cache_misses,
    )

    with telemetry_span(logger, "simulate-pool.subset_and_switch_probs", chrom=int(chrom)):
        t_subset = perf_counter()
        panel_by_pop: dict[str, Any] = {}
        switch_probs_by_pop: dict[str, np.ndarray] = {}
        exemplars_by_pop: dict[str, _PopulationWorkItem] = {}
        for pop in config.populations:
            if pop.donor_only:
                continue
            exemplar = next((x for x in work_items if x.pop_name == pop.name), None)
            if exemplar is not None:
                exemplars_by_pop[pop.name] = exemplar

        def _prepare_pop(pop_name: str, exemplar: _PopulationWorkItem) -> tuple[str, Any, np.ndarray]:
            donor_weights = manifest.donor_weights_by_population.get(pop_name, {})
            panel = subset_panel_donors(
                panel_union_cache[(exemplar.panel_source_kind, exemplar.panel_source_path)],
                exemplar.donors,
                donor_weights=[float(donor_weights.get(sid, 0.0)) for sid in exemplar.donors],
            )
            map_bp, map_cm = load_genetic_map(exemplar.map_path)
            switch_probs = switch_probs_from_positions(
                marker_positions=panel.variants.pos,
                map_bp=map_bp,
                map_cm=map_cm,
                mean_tract_cm=config.simulation.mean_tract_cm,
            )
            return pop_name, panel, switch_probs

        prep_jobs = list(exemplars_by_pop.items())
        prep_workers = min(len(prep_jobs), max(1, min(task_workers, 2)))
        if prep_workers > 1:
            with ThreadPoolExecutor(max_workers=prep_workers) as pool:
                future_to_pop = {
                    pool.submit(_prepare_pop, pop_name, exemplar): pop_name
                    for pop_name, exemplar in prep_jobs
                }
                for future in as_completed(future_to_pop):
                    pop_name_out, panel, switch_probs = future.result()
                    panel_by_pop[pop_name_out] = panel
                    switch_probs_by_pop[pop_name_out] = switch_probs
        else:
            for pop_name, exemplar in prep_jobs:
                pop_name_out, panel, switch_probs = _prepare_pop(pop_name, exemplar)
                panel_by_pop[pop_name_out] = panel
                switch_probs_by_pop[pop_name_out] = switch_probs
        _add_timing("subset_and_switch_probs_sec", perf_counter() - t_subset)
    logger.info(
        "simulate-pool donor prep complete chrom=%s prepared_pops=%s",
        chrom,
        len(panel_by_pop),
    )

    task_items = [item for item in work_items if item.pop_n > 0]
    if not task_items:
        raise ValueError("No (population, shard) tasks produced samples; check n_diploids and n_shards")

    rr = RateLimitedProgress(logger=logger, name="simulate-pool.dispatch", every_sec=60.0)
    task_results: list[dict[str, Any]] = []
    with telemetry_span(logger, "simulate-pool.dispatch_tasks", chrom=int(chrom), task_workers=int(task_workers)):
        t_tasks = perf_counter()
        task_args: list[tuple[Any, ...]] = []
        for item in task_items:
            chrom_dir = ensure_dir(
                config.run_root / "synthetic" / f"chr{chrom}" / f"shard_{item.shard_id}_of_{n_shards}"
            )
            scratch_task_dir = _scratch_dir(
                config,
                chrom=int(chrom),
                shard_id=int(item.shard_id),
                n_shards=int(n_shards),
                population=str(item.pop_name),
            )
            task_args.append(
                (
                    item.pop_name,
                    item.sample_ids,
                    item.pop_n,
                    item.seed,
                    str(chrom_dir),
                    str(scratch_task_dir or chrom_dir),
                    chrom,
                    item.shard_id,
                    n_shards,
                    config.outputs.write_phased_vcf,
                    config.outputs.write_hapstore,
                    config.outputs.hapstore_chunk_variant_size,
                    config.outputs.hapstore_chunk_haplotype_size,
                    config.outputs.hapstore_chunk_sample_size,
                    (
                        config.outputs.shard_hapstore_compression_level
                        if config.outputs.shard_hapstore_compression_level is not None
                        else config.outputs.hapstore_compression_level
                    ),
                    config.outputs.hapstore_store_genotypes,
                    config.outputs.write_pgen,
                    config.outputs.hapstore_vcf_plink2_location,
                    config.outputs.hapstore_vcf_keep_intermediate_pgen,
                    config.outputs.hapstore_vcf_n_shards,
                )
            )

        process_count = min(task_workers, len(task_args))
        progress_interval = max(1, len(task_args) // 10)
        tasks_total = len(task_args)
        completed = 0
        tasks_done_by_pop: dict[str, int] = {pop.name: 0 for pop in config.populations}

        def _record_task_result(result: dict[str, Any]) -> None:
            nonlocal completed
            task_results.append(result)
            completed += 1
            pop_name = str(result.get("population", ""))
            if pop_name in tasks_done_by_pop:
                tasks_done_by_pop[pop_name] += 1

            should_log = completed == 1 or completed == tasks_total or (completed % progress_interval == 0)
            if should_log:
                elapsed = perf_counter() - t_tasks
                tasks_per_min = (completed / elapsed) * 60.0 if elapsed > 0 else float("nan")
                logger.info(
                    "simulate-pool progress chrom=%s completed=%s/%s (%.1f%%) elapsed=%.1fs tasks_per_min=%.2f by_pop=%s",
                    chrom,
                    completed,
                    tasks_total,
                    (100.0 * completed) / tasks_total if tasks_total > 0 else float("nan"),
                    elapsed,
                    tasks_per_min,
                    json.dumps(tasks_done_by_pop, sort_keys=True),
                )
                rr.log(
                    chrom=int(chrom),
                    done=int(completed),
                    total=int(tasks_total),
                    rate_per_min=float(tasks_per_min),
                )

        if process_count == 1:
            _init_task_worker(
                panel_by_pop,
                switch_probs_by_pop,
                threads_per_task=int(threads_per_task),
                haplotype_batch_size=int(config.simulation.haplotype_batch_size),
            )
            for args in task_args:
                _record_task_result(_simulate_population_shard_task(args))
        else:
            ctx = mp.get_context(task_start_method)
            root_logger = logging.getLogger()
            queue_listener = None
            log_queue = None
            log_level = int(root_logger.getEffectiveLevel() or logging.INFO)
            if any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
                log_queue = ctx.Queue()
                queue_listener = setup_queue_listener_for_logger(root_logger, queue=log_queue)
                logger.info(
                    "simulate-pool mp logging enabled chrom=%s start_method=%s",
                    chrom,
                    task_start_method,
                )
            if task_start_method == "fork":
                # For fork workers, preload globals once in parent to maximize copy-on-write sharing.
                _init_task_worker(
                    panel_by_pop,
                    switch_probs_by_pop,
                    threads_per_task=int(threads_per_task),
                    haplotype_batch_size=int(config.simulation.haplotype_batch_size),
                )
                try:
                    with ctx.Pool(
                        processes=process_count,
                        initializer=(_init_pool_worker_logging if log_queue is not None else None),
                        initargs=((log_queue, log_level) if log_queue is not None else ()),
                    ) as pool:
                        for result in pool.imap_unordered(_simulate_population_shard_task, task_args, chunksize=1):
                            _record_task_result(result)
                finally:
                    if queue_listener is not None:
                        queue_listener.stop()
            else:
                try:
                    initializer = _init_task_worker
                    initargs: tuple[Any, ...] = (
                        panel_by_pop,
                        switch_probs_by_pop,
                        int(threads_per_task),
                        int(config.simulation.haplotype_batch_size),
                    )
                    if log_queue is not None:
                        initializer = _init_pool_worker_with_state
                        initargs = (
                            log_queue,
                            log_level,
                            panel_by_pop,
                            switch_probs_by_pop,
                            int(threads_per_task),
                            int(config.simulation.haplotype_batch_size),
                        )
                    with ctx.Pool(
                        processes=process_count,
                        initializer=initializer,
                        initargs=initargs,
                    ) as pool:
                        for result in pool.imap_unordered(_simulate_population_shard_task, task_args, chunksize=1):
                            _record_task_result(result)
                finally:
                    if queue_listener is not None:
                        queue_listener.stop()
        _add_timing("run_population_shard_tasks_sec", perf_counter() - t_tasks)
        rr.log(force=True, chrom=int(chrom), done=int(completed), total=int(tasks_total))

    for result in task_results:
        for key, elapsed in result["timings"].items():
            _add_timing(key, float(elapsed))

    # Write shard manifests in canonical order expected by merge.
    manifest_dir = ensure_dir(config.run_root / "manifests")
    sample_manifest_paths: list[Path] = []
    sample_manifest_by_shard: dict[int, Path] = {}
    with telemetry_span(logger, "simulate-pool.write_sample_manifests", chrom=int(chrom), n_shards=int(n_shards)):
        t_manifest_write = perf_counter()
        by_shard_pop: dict[tuple[int, str], _PopulationWorkItem] = {
            (item.shard_id, item.pop_name): item for item in task_items
        }
        for shard_id in range(n_shards):
            rows: list[dict[str, str | int]] = []
            sample_order = 0
            for pop in config.populations:
                item = by_shard_pop.get((shard_id, pop.name))
                if item is None:
                    continue
                for sid in item.sample_ids:
                    rows.append(
                        {
                            "sample_order": sample_order,
                            "sample_id": sid,
                            "population": pop.name,
                            "chrom": chrom,
                            "shard_id": shard_id,
                            "n_shards": n_shards,
                        }
                    )
                    sample_order += 1
            if not rows:
                continue
            manifest_path = manifest_dir / f"synthetic_samples_chr{chrom}_shard_{shard_id}_of_{n_shards}.tsv"
            pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
            sample_manifest_paths.append(manifest_path)
            sample_manifest_by_shard[int(shard_id)] = manifest_path
        _add_timing("write_sample_manifests_sec", perf_counter() - t_manifest_write)

    if config.outputs.write_combined_shard_output and config.outputs.write_hapstore:
        with telemetry_span(logger, "simulate-pool.write_combined_shard_hapstores", chrom=int(chrom)):
            t_combined_hapstore = perf_counter()
            hapstore_concat_workers = max(1, int(threads_per_task))
            for shard_id in range(n_shards):
                shard_dir = config.run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}"
                pop_hapstores = [
                    shard_dir / f"{pop.name}.chr{chrom}.hapstore.zarr"
                    for pop in config.populations
                    if (shard_dir / f"{pop.name}.chr{chrom}.hapstore.zarr").exists()
                ]
                if not pop_hapstores:
                    continue
                out_hapstore = shard_dir / f"combined.chr{chrom}.hapstore.zarr"
                merge_chromosomes_sample_axis(
                    input_stores=pop_hapstores,
                    output_store=out_hapstore,
                    chrom=f"chr{chrom}",
                    output_mode="w",
                    chunk_variant_size=int(config.outputs.hapstore_chunk_variant_size),
                    chunk_haplotype_size=(
                        None
                        if config.outputs.hapstore_chunk_haplotype_size is None
                        else int(config.outputs.hapstore_chunk_haplotype_size)
                    ),
                    chunk_sample_size=(
                        None
                        if config.outputs.hapstore_chunk_sample_size is None
                        else int(config.outputs.hapstore_chunk_sample_size)
                    ),
                    compression_level=int(
                        config.outputs.shard_hapstore_compression_level
                        if config.outputs.shard_hapstore_compression_level is not None
                        else config.outputs.hapstore_compression_level
                    ),
                    store_genotypes=bool(config.outputs.hapstore_store_genotypes),
                    attrs={"source": "population_amplifier"},
                    n_workers=hapstore_concat_workers,
                )
            _add_timing("write_combined_shard_hapstores_sec", perf_counter() - t_combined_hapstore)

    if (
        config.outputs.write_combined_shard_output
        and config.outputs.write_hapstore
        and config.outputs.write_phased_vcf
    ):
        with telemetry_span(logger, "simulate-pool.write_combined_shard_vcf_from_hapstore", chrom=int(chrom)):
            t_combined_vcf_from_hapstore = perf_counter()
            for shard_id in range(n_shards):
                shard_dir = config.run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}"
                combined_hapstore = shard_dir / f"combined.chr{chrom}.hapstore.zarr"
                if not combined_hapstore.exists():
                    continue
                combined_vcf = shard_dir / f"combined.chr{chrom}.vcf.gz"
                convert_hapstore_to_vcf(
                    hapstore_path=combined_hapstore,
                    output_vcf=combined_vcf,
                    chrom=int(chrom),
                    variant_block_size=1024,
                    plink2_location=config.outputs.hapstore_vcf_plink2_location,
                    keep_intermediate_files=bool(config.outputs.hapstore_vcf_keep_intermediate_pgen),
                    n_shards=int(config.outputs.hapstore_vcf_n_shards),
                )
            _add_timing(
                "write_combined_shard_vcf_from_hapstore_sec",
                perf_counter() - t_combined_vcf_from_hapstore,
            )

    if config.outputs.write_combined_shard_output and config.outputs.write_pgen:
        with telemetry_span(logger, "simulate-pool.write_combined_shard_pgen", chrom=int(chrom)):
            t_combined_pgen = perf_counter()
            for shard_id in range(n_shards):
                shard_dir = config.run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}"
                combined_prefix = shard_dir / f"combined.chr{chrom}"
                combined_hapstore = shard_dir / f"combined.chr{chrom}.hapstore.zarr"
                _write_pgen_from_hapstore(
                    out_prefix=combined_prefix,
                    chrom=int(chrom),
                    hapstore_path=combined_hapstore if combined_hapstore.exists() else None,
                )
            _add_timing("write_combined_shard_pgen_sec", perf_counter() - t_combined_pgen)

    elapsed = perf_counter() - t0
    stage_timings["total_sec"] = elapsed
    shard_results = _build_batch_shard_results(
        config=config,
        chrom=int(chrom),
        n_shards=int(n_shards),
        sample_manifest_by_shard=sample_manifest_by_shard,
        task_results=task_results,
        batch_stage_timings=stage_timings,
        batch_elapsed_seconds=elapsed,
    )
    logger.info(
        "simulate_chromosome_population_shard_pool complete chrom=%s n_shards=%s task_count=%s task_workers=%s threads_per_task=%s timings=%s",
        chrom,
        n_shards,
        len(task_items),
        task_workers,
        int(threads_per_task),
        json.dumps(stage_timings, sort_keys=True),
    )
    return SimulationBatchResult(
        chrom=chrom,
        n_shards=n_shards,
        sample_manifest_paths=sample_manifest_paths,
        shard_results=shard_results,
        elapsed_seconds=elapsed,
        stage_timings=stage_timings,
        task_count=len(task_items),
    )
