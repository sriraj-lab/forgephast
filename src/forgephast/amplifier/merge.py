"""Merge utilities for combining sample-sharded outputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import json
import logging
from pathlib import Path
import shutil
from time import perf_counter
from typing import Sequence

from hapstore import (
    HapStore,
    concat_hapstores as _pkg_concat_hapstores,
)
import pandas as pd

from .config import RunConfig
from .telemetry import telemetry_span
from .utils import ensure_dir
from .vcfio import convert_hapstore_to_pgen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HapStore merge helpers (inlined from former hapstore_ops.py)
# ---------------------------------------------------------------------------


def _read_sample_manifests(manifest_paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, sep="\t", dtype=str) for path in manifest_paths]
    for path, frame in zip(manifest_paths, frames):
        missing_columns = [column for column in ("sample_id", "shard_id") if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"sample manifest {path} is missing required columns: {missing_columns}")
        if frame["sample_id"].astype("string").fillna("").str.strip().eq("").any():
            raise ValueError(f"sample manifest {path} contains blank sample_id values")

    sample_order_presence = {"sample_order" in frame.columns for frame in frames}
    if len(sample_order_presence) > 1:
        raise ValueError("sample_order must be present in either all shard manifests or none")

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df["sample_id"] = merged_df["sample_id"].astype("string").str.strip()
    duplicate_samples = merged_df["sample_id"].duplicated(keep=False)
    if duplicate_samples.any():
        duplicate_ids = merged_df.loc[duplicate_samples, "sample_id"].unique().tolist()
        raise ValueError(f"duplicate sample_id values across shard manifests: {duplicate_ids[:10]}")

    if "sample_order" in merged_df.columns:
        sample_order = pd.to_numeric(merged_df["sample_order"], errors="raise")
        if (sample_order % 1 != 0).any():
            raise ValueError("sample manifest sample_order values must be integers")
        merged_df["sample_order"] = sample_order.astype(int)
    shard_ids = pd.to_numeric(merged_df["shard_id"], errors="raise")
    if (shard_ids % 1 != 0).any():
        raise ValueError("sample manifest shard_id values must be integers")
    merged_df["shard_id"] = shard_ids.astype(int)

    sort_cols = ["shard_id", "sample_order"] if "sample_order" in merged_df.columns else ["shard_id", "sample_id"]
    return merged_df.sort_values(sort_cols)


def _cleanup_hapstore_dirs(paths: Sequence[Path], *, chrom: int, pop: str) -> None:
    removed = 0
    for path in paths:
        if not path.exists():
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    logger.info(
        "merge shard cleanup chrom=%s pop=%s removed_hapstores=%s",
        chrom,
        pop,
        removed,
    )


def _merge_chromosomes_sample_axis_flat(
    input_stores: Sequence[str | Path],
    output_store: str | Path,
    chrom: str | int,
    output_mode: str = "w",
    chunk_variant_size: int | None = None,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int | None = None,
    store_genotypes: bool = False,
    attrs: dict | None = None,
    n_workers: int = 1,
) -> HapStore:
    """Merge stores along sample axis through the canonical HapStore concat API."""
    if int(n_workers) <= 0:
        raise ValueError("n_workers must be > 0")
    if output_mode != "w":
        raise ValueError("output_mode='w' is required; legacy append-mode HapStore merge was removed")

    return _pkg_concat_hapstores(
        input_stores=input_stores,
        output_store=output_store,
        chrom=chrom,
        axis="samples",
        validate="full",
        n_workers=int(n_workers),
        compression_level=compression_level,
        chunk_variant_size=chunk_variant_size,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        store_genotypes=store_genotypes,
        attrs=attrs,
        overwrite=True,
    )


def merge_chromosomes_sample_axis(
    input_stores: Sequence[str | Path],
    output_store: str | Path,
    chrom: str | int,
    output_mode: str = "w",
    chunk_variant_size: int | None = None,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int | None = None,
    store_genotypes: bool = False,
    attrs: dict | None = None,
    n_workers: int = 1,
    batch_size: int = 20,
) -> HapStore:
    """Merge stores along sample axis using a hierarchical (tree) approach.

    This avoids running out of memory when merging a large number of stores
    simultaneously by recursively merging batches.
    """
    if len(input_stores) <= batch_size:
        return _merge_chromosomes_sample_axis_flat(
            input_stores=input_stores,
            output_store=output_store,
            chrom=chrom,
            output_mode=output_mode,
            chunk_variant_size=chunk_variant_size,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            store_genotypes=store_genotypes,
            attrs=attrs,
            n_workers=n_workers,
        )

    out_path = Path(output_store)
    # create a temp dir near output_store for intermediates
    temp_dir = out_path.with_name(f"{out_path.name}.tmp_tree")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        current_stores = list(input_stores)
        level = 0
        while len(current_stores) > batch_size:
            next_stores = []
            for i in range(0, len(current_stores), batch_size):
                batch = current_stores[i:i+batch_size]
                if len(batch) == 1:
                    # pass through single stores without merging
                    next_stores.append(batch[0])
                    continue

                batch_out = temp_dir / f"level_{level}_batch_{i}.hapstore.zarr"
                _merge_chromosomes_sample_axis_flat(
                    input_stores=batch,
                    output_store=batch_out,
                    chrom=chrom,
                    output_mode="w",
                    chunk_variant_size=chunk_variant_size,
                    chunk_haplotype_size=chunk_haplotype_size,
                    chunk_sample_size=chunk_sample_size,
                    compression_level=compression_level,
                    store_genotypes=store_genotypes,
                    attrs=attrs,
                    n_workers=n_workers,
                )
                next_stores.append(batch_out)

                # force GC to reclaim chunk memory
                gc.collect()

            current_stores = next_stores
            level += 1

        # final merge
        return _merge_chromosomes_sample_axis_flat(
            input_stores=current_stores,
            output_store=output_store,
            chrom=chrom,
            output_mode=output_mode,
            chunk_variant_size=chunk_variant_size,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            store_genotypes=store_genotypes,
            attrs=attrs,
            n_workers=n_workers,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@dataclass
class MergeResult:
    chrom: int
    combined_hapstore: Path | None
    merged_pgen_prefix: Path | None
    merged_manifest: Path
    population_merged_hapstores: dict[str, Path]
    stage_timings: dict[str, float]
    thread_plan: dict[str, int]


def _resolve_thread_plan(config: RunConfig, n_populations: int) -> dict[str, int]:
    total_threads = int(config.simulation.threads_per_task)
    if config.simulation.task_workers is not None:
        total_threads = total_threads * int(config.simulation.task_workers)
    total_threads = max(1, int(total_threads))

    population_workers = max(1, int(getattr(config.outputs, "merge_population_workers", 1)))
    population_workers = min(population_workers, max(1, int(n_populations)))
    threads_per_population = max(1, total_threads // population_workers)
    return {
        "total_threads": int(total_threads),
        "population_workers": int(population_workers),
        "threads_per_population": int(threads_per_population),
    }


def _write_merged_pgen(
    *,
    out_prefix: Path,
    chrom: int,
    merged_hapstore: Path | None,
) -> Path:
    if merged_hapstore is None:
        raise FileNotFoundError(
            "Merged PGEN export requires a merged combined HapStore; set outputs.merge_combined_populations=true."
        )
    convert_hapstore_to_pgen(
        hapstore_path=merged_hapstore,
        out_prefix=out_prefix,
        chrom=chrom,
    )
    return out_prefix


def merge_chromosome(
    config: RunConfig,
    chrom: int,
    n_shards: int,
    merge_combined: bool | None = None,
    input_root: str | Path | None = None,
    populations: list[str] | None = None,
) -> MergeResult:
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")

    t0 = perf_counter()
    stage_timings: dict[str, float] = {}

    input_run_root = Path(input_root) if input_root is not None else config.run_root

    configured_populations = [pop.name for pop in config.populations if not pop.donor_only]
    if not configured_populations:
        raise ValueError("No populations configured")
    if populations is None:
        populations = configured_populations
    else:
        configured_set = set(configured_populations)
        deduped: list[str] = []
        for pop in populations:
            pop_name = str(pop)
            if pop_name not in configured_set:
                raise ValueError(f"Unknown merge population: {pop_name}")
            if pop_name not in deduped:
                deduped.append(pop_name)
        if not deduped:
            raise ValueError("No merge populations requested")
        populations = deduped

    thread_plan = _resolve_thread_plan(config, n_populations=len(populations))

    with telemetry_span(
        logger,
        "merge.collect_inputs",
        chrom=int(chrom),
        n_shards=int(n_shards),
        n_populations=int(len(populations)),
    ):
        t_collect = perf_counter()
        shard_hapstores_by_population: dict[str, list[Path]] = {pop: [] for pop in populations}
        manifest_paths: list[Path] = []
        for shard_id in range(n_shards):
            shard_dir = input_run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}"
            manifest = (
                input_run_root / "manifests" / f"synthetic_samples_chr{chrom}_shard_{shard_id}_of_{n_shards}.tsv"
            )
            if not manifest.exists():
                raise FileNotFoundError(f"Missing shard sample manifest: {manifest}")
            manifest_paths.append(manifest)

            for pop in populations:
                pop_hapstore = shard_dir / f"{pop}.chr{chrom}.hapstore.zarr"
                if config.outputs.write_hapstore and not pop_hapstore.exists():
                    raise FileNotFoundError(f"Missing shard population HapStore: {pop_hapstore}")
                if pop_hapstore.exists():
                    shard_hapstores_by_population[pop].append(pop_hapstore)
        merged_df = _read_sample_manifests(manifest_paths)
        stage_timings["collect_inputs_sec"] = perf_counter() - t_collect
    logger.info(
        "merge input summary chrom=%s n_shards=%s populations=%s hapstore_counts=%s",
        chrom,
        n_shards,
        len(populations),
        json.dumps({pop: len(shard_hapstores_by_population.get(pop, [])) for pop in populations}, sort_keys=True),
    )

    out_dir = ensure_dir(config.run_root / "merged" / f"chr{chrom}")
    pop_dir = ensure_dir(out_dir / "populations")

    population_merged_hapstores: dict[str, Path] = {}

    def _merge_population(pop: str) -> tuple[str, Path | None, float]:
        logger.info(
            "merge population start chrom=%s pop=%s shard_hapstores=%s",
            chrom,
            pop,
            len(shard_hapstores_by_population.get(pop, [])),
        )
        with telemetry_span(logger, "merge.population", chrom=int(chrom), pop=str(pop), n_shards=int(n_shards)):
            pop_hapstore_out: Path | None = None
            elapsed_hapstore = 0.0
            if config.outputs.write_hapstore and shard_hapstores_by_population.get(pop):
                t_pop_hapstore = perf_counter()
                pop_hapstore_out = pop_dir / f"{pop}.chr{chrom}.merged.hapstore.zarr"
                merge_chromosomes_sample_axis(
                    input_stores=shard_hapstores_by_population[pop],
                    output_store=pop_hapstore_out,
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
                    compression_level=int(config.outputs.hapstore_compression_level),
                    store_genotypes=bool(config.outputs.hapstore_store_genotypes),
                    attrs={"source": "population_amplifier"},
                    n_workers=int(thread_plan["threads_per_population"]),
                    batch_size=int(config.outputs.hapstore_merge_batch_size),
                )
                elapsed_hapstore = perf_counter() - t_pop_hapstore
                if bool(config.outputs.cleanup_shard_hapstores_after_merge):
                    _cleanup_hapstore_dirs(
                        list(shard_hapstores_by_population.get(pop, [])),
                        chrom=int(chrom),
                        pop=str(pop),
                    )

            logger.info(
                "merge population done chrom=%s pop=%s hapstore_elapsed=%.2fs outputs=%s",
                chrom,
                pop,
                elapsed_hapstore,
                json.dumps(
                    {
                        "merged_hapstore": str(pop_hapstore_out) if pop_hapstore_out is not None else None,
                    },
                    sort_keys=True,
                ),
            )
            return pop, pop_hapstore_out, elapsed_hapstore

    t_merge_pop_all = perf_counter()
    population_workers = int(thread_plan["population_workers"])
    if population_workers <= 1:
        for idx, pop in enumerate(populations, start=1):
            t_pop_wall = perf_counter()
            pop, pop_hapstore_out, elapsed_hapstore = _merge_population(pop)
            if pop_hapstore_out is not None:
                population_merged_hapstores[pop] = pop_hapstore_out
                stage_timings[f"merge_population_{pop}_hapstore_sec"] = elapsed_hapstore
            logger.info(
                "merge population progress chrom=%s completed=%s/%s pop=%s wall_elapsed=%.2fs",
                chrom,
                idx,
                len(populations),
                pop,
                perf_counter() - t_pop_wall,
            )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=int(population_workers)) as pool:
            future_starts: dict[object, float] = {}
            futures = {}
            for pop in populations:
                fut = pool.submit(_merge_population, pop)
                futures[fut] = pop
                future_starts[fut] = perf_counter()
            for fut in as_completed(futures):
                pop, pop_hapstore_out, elapsed_hapstore = fut.result()
                if pop_hapstore_out is not None:
                    population_merged_hapstores[pop] = pop_hapstore_out
                    stage_timings[f"merge_population_{pop}_hapstore_sec"] = elapsed_hapstore
                completed += 1
                logger.info(
                    "merge population progress chrom=%s completed=%s/%s pop=%s wall_elapsed=%.2fs",
                    chrom,
                    completed,
                    len(populations),
                    pop,
                    perf_counter() - float(future_starts.get(fut, t_merge_pop_all)),
                )
    stage_timings["merge_population_hapstores_sec"] = perf_counter() - t_merge_pop_all

    with telemetry_span(logger, "merge.write_manifest", chrom=int(chrom), n_shards=int(n_shards)):
        merged_manifest_dir = ensure_dir(config.run_root / "manifests")
        merged_manifest = merged_manifest_dir / f"merged_samples_chr{chrom}.tsv"
        t_manifest = perf_counter()
        merged_df.to_csv(merged_manifest, sep="\t", index=False)
        stage_timings["merge_manifest_sec"] = perf_counter() - t_manifest
    logger.info(
        "merge manifest written chrom=%s path=%s rows=%s",
        chrom,
        merged_manifest,
        len(merged_df),
    )

    combine_requested = config.outputs.merge_combined_populations if merge_combined is None else merge_combined
    combined_hapstore: Path | None = None
    if combine_requested:
        if config.outputs.write_hapstore and population_merged_hapstores:
            logger.info(
                "merge combined-hapstore start chrom=%s input_populations=%s",
                chrom,
                len(population_merged_hapstores),
            )
            with telemetry_span(logger, "merge.combined_hapstore", chrom=int(chrom), n_inputs=int(len(population_merged_hapstores))):
                t_merge_combined_hapstore = perf_counter()
                combined_hapstore = out_dir / f"combined.chr{chrom}.merged.hapstore.zarr"
                ordered_pop_hapstores = [
                    population_merged_hapstores[pop]
                    for pop in populations
                    if pop in population_merged_hapstores
                ]
                merge_chromosomes_sample_axis(
                    input_stores=ordered_pop_hapstores,
                    output_store=combined_hapstore,
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
                    compression_level=int(config.outputs.hapstore_compression_level),
                    store_genotypes=bool(config.outputs.hapstore_store_genotypes),
                    attrs={"source": "population_amplifier"},
                    n_workers=int(thread_plan["threads_per_population"]),
                    batch_size=int(config.outputs.hapstore_merge_batch_size),
                )
                stage_timings["merge_combined_hapstore_sec"] = perf_counter() - t_merge_combined_hapstore
            logger.info(
                "merge combined-hapstore done chrom=%s output=%s elapsed=%.2fs",
                chrom,
                combined_hapstore,
                stage_timings["merge_combined_hapstore_sec"],
            )

    merged_pgen_prefix = None
    if config.outputs.write_pgen:
        with telemetry_span(logger, "merge.write_pgen", chrom=int(chrom), threads=int(thread_plan["total_threads"])):
            t_pgen = perf_counter()
            merged_pgen_prefix = _write_merged_pgen(
                out_prefix=out_dir / f"combined.chr{chrom}.merged",
                chrom=int(chrom),
                merged_hapstore=combined_hapstore,
            )
            elapsed_pgen = perf_counter() - t_pgen
            stage_timings["write_merged_pgen_sec"] = elapsed_pgen
            stage_timings["write_merged_pgen_hapstore_direct_sec"] = elapsed_pgen

    stage_timings["total_sec"] = perf_counter() - t0
    logger.info(
        "Merged chr=%s n_shards=%s combine_requested=%s thread_plan=%s timings=%s",
        chrom,
        n_shards,
        combine_requested,
        json.dumps(thread_plan, sort_keys=True),
        json.dumps(stage_timings, sort_keys=True),
    )
    return MergeResult(
        chrom=chrom,
        combined_hapstore=combined_hapstore,
        merged_pgen_prefix=merged_pgen_prefix,
        merged_manifest=merged_manifest,
        population_merged_hapstores=population_merged_hapstores,
        stage_timings=stage_timings,
        thread_plan=thread_plan,
    )
