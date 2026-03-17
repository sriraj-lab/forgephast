"""Stage-0 donor HapStore cache builder."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

from .config import RunConfig
from .donor_panels import ensure_donor_panel_hapstore
from .donors import Manifest, load_manifest
from .telemetry import telemetry_span
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class DonorCacheEntry:
    chrom: int
    panel_source_kind: str
    panel_source_path: Path
    donor_count: int
    cache_path: Path
    cache_hit: bool


@dataclass
class DonorCacheResult:
    cache_root: Path
    entries: list[DonorCacheEntry]
    elapsed_seconds: float


def _jobs_for_chrom(
    config: RunConfig,
    manifest: Manifest,
    chrom: int,
) -> list[tuple[str, Path, list[str]]]:
    union_donors_by_source: dict[tuple[str, Path], set[str]] = {}
    for pop in config.populations:
        source_kind, source_path = config.resolve_donor_panel_source(
            chrom=int(chrom),
            population=str(pop.name),
        )
        union_donors_by_source.setdefault((source_kind, source_path), set()).update(
            manifest.donors_by_population[pop.name]
        )
    return [
        (source_kind, source_path, sorted(donors))
        for (source_kind, source_path), donors in union_donors_by_source.items()
    ]


def build_donor_cache(
    config: RunConfig,
    chroms: list[int] | None = None,
    force: bool = False,
) -> DonorCacheResult:
    t0 = perf_counter()
    cache_root = ensure_dir(config.donor_cache_root)
    work_chroms = sorted(set(chroms if chroms else config.simulation.chromosomes))
    donor_io_workers = config.simulation.donor_io_workers
    entries: list[DonorCacheEntry] = []
    with telemetry_span(logger, "donor_cache.load_manifest"):
        manifest = load_manifest(config)

    for chrom in work_chroms:
        jobs = _jobs_for_chrom(config, manifest=manifest, chrom=chrom)
        if not jobs:
            continue

        with telemetry_span(
            logger,
            "donor_cache.build_chrom",
            chrom=int(chrom),
            jobs=int(len(jobs)),
            donor_io_workers=int(donor_io_workers),
            force=bool(force),
        ):
            max_workers = min(max(1, donor_io_workers), len(jobs))
            if max_workers > 1 and len(jobs) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_job = {
                        pool.submit(
                            ensure_donor_panel_hapstore,
                            source_path=source_path,
                            source_kind=source_kind,
                            chrom=chrom,
                            donor_ids=donor_ids,
                            cache_dir=cache_root,
                            force_refresh=force,
                            ingest_workers=max(1, int(donor_io_workers)),
                        ): (source_kind, source_path, donor_ids)
                        for source_kind, source_path, donor_ids in jobs
                    }
                    for future in as_completed(future_to_job):
                        source_kind, source_path, donor_ids = future_to_job[future]
                        cache_path, cache_hit = future.result()
                        entries.append(
                            DonorCacheEntry(
                                chrom=chrom,
                                panel_source_kind=source_kind,
                                panel_source_path=source_path,
                                donor_count=len(donor_ids),
                                cache_path=cache_path,
                                cache_hit=cache_hit,
                            )
                        )
            else:
                for source_kind, source_path, donor_ids in jobs:
                    cache_path, cache_hit = ensure_donor_panel_hapstore(
                        source_path=source_path,
                        source_kind=source_kind,
                        chrom=chrom,
                        donor_ids=donor_ids,
                        cache_dir=cache_root,
                        force_refresh=force,
                        ingest_workers=max(1, int(donor_io_workers)),
                    )
                    entries.append(
                        DonorCacheEntry(
                            chrom=chrom,
                            panel_source_kind=source_kind,
                            panel_source_path=source_path,
                            donor_count=len(donor_ids),
                            cache_path=cache_path,
                            cache_hit=cache_hit,
                        )
                    )
            chrom_hits = sum(1 for e in entries if e.chrom == chrom and e.cache_hit)
            chrom_misses = sum(1 for e in entries if e.chrom == chrom and not e.cache_hit)
            logger.info(
                "donor_cache chrom summary chrom=%s entries=%s hits=%s misses=%s",
                chrom,
                chrom_hits + chrom_misses,
                chrom_hits,
                chrom_misses,
            )

    elapsed = perf_counter() - t0
    total_hits = sum(1 for e in entries if e.cache_hit)
    total_misses = sum(1 for e in entries if not e.cache_hit)
    logger.info(
        "donor_cache complete chroms=%s entries=%s hits=%s misses=%s elapsed_sec=%.3f cache_root=%s",
        len(work_chroms),
        len(entries),
        total_hits,
        total_misses,
        elapsed,
        cache_root,
    )
    return DonorCacheResult(cache_root=cache_root, entries=entries, elapsed_seconds=elapsed)
