"""Merge utilities for combining sample-sharded outputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
from time import perf_counter

import pandas as pd

from .config import RunConfig
from .hapstore_ops import merge_chromosomes_sample_axis
from .telemetry import telemetry_span
from .utils import ensure_dir
from .vcfio import convert_hapstore_to_pgen, convert_vcf_to_pgen, merge_shard_vcfs, merge_shard_vcfs_bcftools

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    chrom: int
    merged_vcf: Path | None
    combined_hapstore: Path | None
    merged_pgen_prefix: Path | None
    merged_manifest: Path
    merge_backend: str
    population_merged_vcfs: dict[str, Path]
    population_merged_hapstores: dict[str, Path]
    population_merge_backends: dict[str, str]
    combined_merge_backend: str | None
    stage_timings: dict[str, float]
    thread_plan: dict[str, int]


def _resolve_merge_backend(config: RunConfig) -> str:
    backend = config.outputs.merge_backend
    if backend != "auto":
        return backend
    if shutil.which(config.outputs.bcftools_path):
        return "bcftools"
    return "python"


def _resolve_thread_plan(config: RunConfig, n_populations: int) -> dict[str, int]:
    total_threads = int(config.simulation.threads_per_task)
    if config.simulation.task_workers is not None:
        total_threads = total_threads * int(config.simulation.task_workers)
    total_threads = max(1, int(total_threads))

    population_workers = max(1, int(getattr(config.outputs, "merge_population_workers", 1)))
    population_workers = min(population_workers, max(1, int(n_populations)))
    threads_per_population = max(1, total_threads // population_workers)
    # Default behavior: consume available per-pop merge threads.
    # If user sets outputs.bcftools_threads > 1, treat it as an upper bound.
    bcftools_threads_per_population = threads_per_population
    if config.outputs.bcftools_threads > 1:
        bcftools_threads_per_population = min(config.outputs.bcftools_threads, threads_per_population)

    return {
        "total_threads": int(total_threads),
        "population_workers": int(population_workers),
        "threads_per_population": int(threads_per_population),
        "bcftools_threads_per_population": int(bcftools_threads_per_population),
    }


def _resolve_pgen_mode(preference: str, hapstore_path: Path | None) -> str:
    if preference == "vcf_bridge":
        return "vcf_bridge"
    if preference == "hapstore_direct":
        if hapstore_path is None:
            raise RuntimeError(
                "outputs.pgen_source_preference='hapstore_direct' requires merged combined HapStore; "
                "set outputs.write_hapstore=true and merge_combined_populations=true, or use "
                "pgen_source_preference='vcf_bridge'"
            )
        return "hapstore_direct"
    # auto
    return "hapstore_direct" if hapstore_path is not None else "vcf_bridge"


def _write_merged_pgen_with_preference(
    *,
    out_prefix: Path,
    chrom: int,
    preference: str,
    merged_hapstore: Path | None,
    merged_vcf: Path | None,
    plink2_path: str,
    threads: int,
) -> tuple[Path, str]:
    mode = _resolve_pgen_mode(preference=preference, hapstore_path=merged_hapstore)
    if mode == "hapstore_direct":
        try:
            convert_hapstore_to_pgen(
                hapstore_path=merged_hapstore,
                out_prefix=out_prefix,
                chrom=chrom,
            )
            return out_prefix, "hapstore_direct"
        except Exception:
            if preference != "auto":
                raise
            logger.warning(
                "Direct merged HapStore->PGEN failed for prefix=%s; falling back to VCF bridge in auto mode",
                out_prefix,
            )
            mode = "vcf_bridge"

    if merged_vcf is None:
        raise FileNotFoundError(
            "VCF bridge selected for merged PGEN output, but merged combined VCF is unavailable; "
            "ensure combined VCF output exists or use direct HapStore PGEN mode with merged HapStore enabled"
        )
    convert_vcf_to_pgen(
        vcf_path=merged_vcf,
        out_prefix=out_prefix,
        plink2_path=plink2_path,
        threads=threads,
    )
    return out_prefix, "vcf_bridge"


def _merge_vcf_paths(
    shard_vcfs: list[Path],
    output_vcf: Path,
    backend: str,
    config: RunConfig,
    bcftools_threads: int,
    allow_auto_fallback: bool,
) -> str:
    if backend == "bcftools":
        try:
            merge_shard_vcfs_bcftools(
                shard_vcfs=shard_vcfs,
                output_vcf_gz=output_vcf,
                bcftools_path=config.outputs.bcftools_path,
                threads=bcftools_threads,
                no_index=config.outputs.bcftools_no_index,
            )
            return "bcftools"
        except Exception:
            if not allow_auto_fallback:
                raise
            logger.warning(
                "bcftools merge failed in auto mode for output=%s; falling back to python merge",
                output_vcf,
            )
            merge_shard_vcfs(shard_vcfs=shard_vcfs, output_vcf_gz=output_vcf)
            return "python"

    merge_shard_vcfs(shard_vcfs=shard_vcfs, output_vcf_gz=output_vcf)
    return "python"


def merge_chromosome(
    config: RunConfig,
    chrom: int,
    n_shards: int,
    merge_combined: bool | None = None,
    input_root: str | Path | None = None,
) -> MergeResult:
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")

    t0 = perf_counter()
    stage_timings: dict[str, float] = {}

    input_run_root = Path(input_root) if input_root is not None else config.run_root

    populations = [pop.name for pop in config.populations]
    if not populations:
        raise ValueError("No populations configured")

    merge_backend_preference = _resolve_merge_backend(config)
    allow_auto_fallback = config.outputs.merge_backend == "auto"
    thread_plan = _resolve_thread_plan(config, n_populations=len(populations))

    with telemetry_span(
        logger,
        "merge.collect_inputs",
        chrom=int(chrom),
        n_shards=int(n_shards),
        n_populations=int(len(populations)),
    ):
        t_collect = perf_counter()
        shard_vcfs_by_population: dict[str, list[Path]] = {pop: [] for pop in populations}
        shard_hapstores_by_population: dict[str, list[Path]] = {pop: [] for pop in populations}
        manifest_paths: list[Path] = []
        want_vcfs = bool(config.outputs.write_phased_vcf)
        for shard_id in range(n_shards):
            shard_dir = input_run_root / "synthetic" / f"chr{chrom}" / f"shard_{shard_id}_of_{n_shards}"
            manifest = (
                input_run_root / "manifests" / f"synthetic_samples_chr{chrom}_shard_{shard_id}_of_{n_shards}.tsv"
            )
            if not manifest.exists():
                raise FileNotFoundError(f"Missing shard sample manifest: {manifest}")
            manifest_paths.append(manifest)

            for pop in populations:
                if want_vcfs:
                    pop_vcf = shard_dir / f"{pop}.chr{chrom}.vcf.gz"
                    if pop_vcf.exists():
                        shard_vcfs_by_population[pop].append(pop_vcf)
                pop_hapstore = shard_dir / f"{pop}.chr{chrom}.hapstore.zarr"
                if config.outputs.write_hapstore and not pop_hapstore.exists():
                    raise FileNotFoundError(f"Missing shard population HapStore: {pop_hapstore}")
                if pop_hapstore.exists():
                    shard_hapstores_by_population[pop].append(pop_hapstore)
        for pop, vcf_paths in shard_vcfs_by_population.items():
            if vcf_paths and len(vcf_paths) != n_shards:
                raise FileNotFoundError(
                    f"Incomplete shard population VCF set for {pop}: found {len(vcf_paths)} expected {n_shards}"
                )
        stage_timings["collect_inputs_sec"] = perf_counter() - t_collect
    logger.info(
        "merge input summary chrom=%s n_shards=%s populations=%s vcf_counts=%s hapstore_counts=%s",
        chrom,
        n_shards,
        len(populations),
        json.dumps({pop: len(shard_vcfs_by_population.get(pop, [])) for pop in populations}, sort_keys=True),
        json.dumps({pop: len(shard_hapstores_by_population.get(pop, [])) for pop in populations}, sort_keys=True),
    )

    out_dir = ensure_dir(config.run_root / "merged" / f"chr{chrom}")
    pop_dir = ensure_dir(out_dir / "populations")

    population_merged_vcfs: dict[str, Path] = {}
    population_merged_hapstores: dict[str, Path] = {}
    population_merge_backends: dict[str, str] = {}

    def _merge_population(pop: str) -> tuple[str, Path | None, str, float, Path | None, float]:
        logger.info(
            "merge population start chrom=%s pop=%s shard_vcfs=%s shard_hapstores=%s",
            chrom,
            pop,
            len(shard_vcfs_by_population.get(pop, [])),
            len(shard_hapstores_by_population.get(pop, [])),
        )
        with telemetry_span(logger, "merge.population", chrom=int(chrom), pop=str(pop), n_shards=int(n_shards)):
            pop_out: Path | None = None
            backend_used = "none"
            elapsed_vcf = 0.0
            if shard_vcfs_by_population.get(pop):
                pop_out = pop_dir / f"{pop}.chr{chrom}.merged.vcf.gz"
                t_pop = perf_counter()
                backend_used = _merge_vcf_paths(
                    shard_vcfs=shard_vcfs_by_population[pop],
                    output_vcf=pop_out,
                    backend=merge_backend_preference,
                    config=config,
                    bcftools_threads=thread_plan["bcftools_threads_per_population"],
                    allow_auto_fallback=allow_auto_fallback,
                )
                elapsed_vcf = perf_counter() - t_pop

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
                    merge_strategy=str(config.outputs.hapstore_merge_strategy),
                    n_workers=int(thread_plan["threads_per_population"]),
                    batch_size=int(config.outputs.hapstore_merge_batch_size),
                )
                elapsed_hapstore = perf_counter() - t_pop_hapstore

            logger.info(
                "merge population done chrom=%s pop=%s vcf_elapsed=%.2fs hapstore_elapsed=%.2fs backend=%s outputs=%s",
                chrom,
                pop,
                elapsed_vcf,
                elapsed_hapstore,
                backend_used,
                json.dumps(
                    {
                        "merged_vcf": str(pop_out) if pop_out is not None else None,
                        "merged_hapstore": str(pop_hapstore_out) if pop_hapstore_out is not None else None,
                    },
                    sort_keys=True,
                ),
            )
            return pop, pop_out, backend_used, elapsed_vcf, pop_hapstore_out, elapsed_hapstore

    t_merge_pop_all = perf_counter()
    population_workers = int(thread_plan["population_workers"])
    if population_workers <= 1:
        for idx, pop in enumerate(populations, start=1):
            t_pop_wall = perf_counter()
            pop, pop_out, backend_used, elapsed, pop_hapstore_out, elapsed_hapstore = _merge_population(pop)
            if pop_out is not None:
                population_merged_vcfs[pop] = pop_out
                population_merge_backends[pop] = backend_used
                stage_timings[f"merge_population_{pop}_vcf_sec"] = elapsed
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
                pop, pop_out, backend_used, elapsed, pop_hapstore_out, elapsed_hapstore = fut.result()
                if pop_out is not None:
                    population_merged_vcfs[pop] = pop_out
                    population_merge_backends[pop] = backend_used
                    stage_timings[f"merge_population_{pop}_vcf_sec"] = elapsed
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
    stage_timings["merge_population_vcfs_sec"] = perf_counter() - t_merge_pop_all

    with telemetry_span(logger, "merge.write_manifest", chrom=int(chrom), n_shards=int(n_shards)):
        merged_manifest_dir = ensure_dir(config.run_root / "manifests")
        merged_manifest = merged_manifest_dir / f"merged_samples_chr{chrom}.tsv"
        t_manifest = perf_counter()
        frames = [pd.read_csv(path, sep="\t", dtype=str) for path in manifest_paths]
        merged_df = pd.concat(frames, ignore_index=True)
        if "sample_order" in merged_df.columns:
            merged_df["sample_order"] = pd.to_numeric(merged_df["sample_order"], errors="coerce").fillna(0).astype(int)
        merged_df["shard_id"] = pd.to_numeric(merged_df["shard_id"], errors="coerce").fillna(0).astype(int)
        sort_cols = ["shard_id", "sample_order"] if "sample_order" in merged_df.columns else ["shard_id", "sample_id"]
        merged_df = merged_df.drop_duplicates(subset=["sample_id"]).sort_values(sort_cols)
        merged_df.to_csv(merged_manifest, sep="\t", index=False)
        stage_timings["merge_manifest_sec"] = perf_counter() - t_manifest
    logger.info(
        "merge manifest written chrom=%s path=%s rows=%s",
        chrom,
        merged_manifest,
        len(merged_df),
    )

    combine_requested = config.outputs.merge_combined_populations if merge_combined is None else merge_combined
    combined_merge_backend: str | None = None
    merged_vcf: Path | None = None
    combined_hapstore: Path | None = None
    if combine_requested:
        if population_merged_vcfs:
            logger.info(
                "merge combined-vcf start chrom=%s input_populations=%s",
                chrom,
                len(population_merged_vcfs),
            )
            with telemetry_span(logger, "merge.combined_vcf", chrom=int(chrom), n_inputs=int(len(population_merged_vcfs))):
                t_merge_combined = perf_counter()
                merged_vcf = out_dir / f"combined.chr{chrom}.merged.vcf.gz"
                ordered_pop_vcfs = [
                    population_merged_vcfs[pop] for pop in populations if pop in population_merged_vcfs
                ]
                combined_merge_backend = _merge_vcf_paths(
                    shard_vcfs=ordered_pop_vcfs,
                    output_vcf=merged_vcf,
                    backend=merge_backend_preference,
                    config=config,
                    bcftools_threads=thread_plan["bcftools_threads_per_population"],
                    allow_auto_fallback=allow_auto_fallback,
                )
                stage_timings["merge_combined_vcf_sec"] = perf_counter() - t_merge_combined
            logger.info(
                "merge combined-vcf done chrom=%s output=%s backend=%s elapsed=%.2fs",
                chrom,
                merged_vcf,
                combined_merge_backend,
                stage_timings["merge_combined_vcf_sec"],
            )

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
                    merge_strategy=str(config.outputs.hapstore_merge_strategy),
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
            merged_pgen_prefix, pgen_mode = _write_merged_pgen_with_preference(
                out_prefix=out_dir / f"combined.chr{chrom}.merged",
                chrom=int(chrom),
                preference=str(config.outputs.pgen_source_preference),
                merged_hapstore=combined_hapstore,
                merged_vcf=merged_vcf,
                plink2_path=config.outputs.plink2_path,
                threads=int(thread_plan["total_threads"]),
            )
            elapsed_pgen = perf_counter() - t_pgen
            stage_timings["write_merged_pgen_sec"] = elapsed_pgen
            stage_timings[f"write_merged_pgen_{pgen_mode}_sec"] = elapsed_pgen

    if combined_merge_backend is not None:
        merge_backend = combined_merge_backend
    else:
        unique_backends = sorted(set(population_merge_backends.values()))
        if not unique_backends:
            merge_backend = "none"
        else:
            merge_backend = unique_backends[0] if len(unique_backends) == 1 else "mixed"

    stage_timings["total_sec"] = perf_counter() - t0
    logger.info(
        "Merged chr=%s n_shards=%s backend=%s combine_requested=%s thread_plan=%s timings=%s",
        chrom,
        n_shards,
        merge_backend,
        combine_requested,
        json.dumps(thread_plan, sort_keys=True),
        json.dumps(stage_timings, sort_keys=True),
    )
    return MergeResult(
        chrom=chrom,
        merged_vcf=merged_vcf,
        combined_hapstore=combined_hapstore,
        merged_pgen_prefix=merged_pgen_prefix,
        merged_manifest=merged_manifest,
        merge_backend=merge_backend,
        population_merged_vcfs=population_merged_vcfs,
        population_merged_hapstores=population_merged_hapstores,
        population_merge_backends=population_merge_backends,
        combined_merge_backend=combined_merge_backend,
        stage_timings=stage_timings,
        thread_plan=thread_plan,
    )
