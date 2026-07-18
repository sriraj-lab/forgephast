"""Cohort-level genotype and ancestry input preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from hapstore.ids import read_sample_ids

from ...local_ancestry import LocalAncestrySource
from ..config import CohortConfig, RunConfig
from ..data.cohort_support import load_population_labels
from ..data.genotypes import GenotypeAdapter, check_id_alignment
from ..local_ancestry.tracts import (
    load_tracts_tsv_gz,
    rasterize_tracts_to_variant_ancestry,
)
from .background import _exclude_causal_grm_variants, _standardize_grm_genotypes

TimingCallback = Callable[..., None]


@dataclass
class CohortInputCaches:
    runtime_cache: dict[str, Any] | None
    tracts: Dict[tuple[str, int, str], dict]
    ancestry_sources: Dict[tuple[str, int, str], LocalAncestrySource]
    sample_ids: Dict[str, List[str]]
    sample_populations: Dict[str, List[str]]
    genotypes: dict[str, dict[str, Any]]
    max_bytes: int | None

    @classmethod
    def from_runtime_cache(
        cls,
        runtime_cache: dict[str, Any] | None,
    ) -> CohortInputCaches:
        if runtime_cache is None:
            return cls(
                runtime_cache=None,
                tracts={},
                ancestry_sources={},
                sample_ids={},
                sample_populations={},
                genotypes={},
                max_bytes=0,
            )

        max_bytes = runtime_cache.get("max_cache_bytes")
        return cls(
            runtime_cache=runtime_cache,
            tracts=runtime_cache.setdefault("simulate_tracts_cache", {}),
            ancestry_sources=runtime_cache.setdefault(
                "simulate_ancestry_source_cache", {}
            ),
            sample_ids=runtime_cache.setdefault("simulate_sample_ids_cache", {}),
            sample_populations=runtime_cache.setdefault(
                "simulate_sample_pops_cache", {}
            ),
            genotypes=runtime_cache.setdefault("simulate_genotype_cache", {}),
            max_bytes=None if max_bytes is None else int(max_bytes),
        )

    def bump(self, name: str, *, hit: bool) -> None:
        if self.runtime_cache is None:
            return
        bucket_name = "cache_hits" if hit else "cache_misses"
        counters = self.runtime_cache.setdefault(bucket_name, {})
        counters[name] = int(counters.get(name, 0)) + 1

    def store_genotypes(self, key: str, entry: dict[str, Any]) -> None:
        entry_bytes = _cache_entry_bytes(entry)
        if self.max_bytes is None:
            self.genotypes[key] = entry
            return

        used_bytes = (
            int(self.runtime_cache.get("simulate_genotype_cache_bytes", 0))
            if self.runtime_cache is not None
            else 0
        )
        if used_bytes + entry_bytes > self.max_bytes:
            return
        self.genotypes[key] = entry
        if self.runtime_cache is not None:
            self.runtime_cache["simulate_genotype_cache_bytes"] = used_bytes + entry_bytes


@dataclass
class PreparedCohortInputs:
    sample_ids: List[str]
    sample_populations: List[str]
    sample_frame: pd.DataFrame
    genotype_matrix: np.ndarray | None
    truth_admix_indices: List[int] | None
    truth_admix_sample_ids: List[str] | None
    admix_haplotypes: np.ndarray | None
    admix_local_ancestry_codes: np.ndarray | None
    donor_fallback_indices: List[int] | None
    donor_fallback_sample_ids: List[str] | None
    donor_fallback_sample_pops: List[str] | None
    la_genotypes_by_population: dict[str, np.ndarray]
    dosage_effect_allele: dict[str, str]
    dosage_other_allele: dict[str, str]
    background_grm_factor: np.ndarray | None
    background_grm_variant_count: int
    background_grm_excluded_causal_count: int
    la_enabled: bool
    cohort_tmp: Path


def _sample_id_prefix(sample_id: str) -> str:
    token = str(sample_id).strip()
    prefix, sep, _ = token.rpartition("_syn")
    return prefix if sep else token


def _sample_frame(sample_ids: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"FID": sample_ids, "IID": sample_ids}, copy=False)


def _cache_entry_bytes(entry: dict[str, Any]) -> int:
    total = 0
    sample_frame = entry.get("sample_frame")
    if isinstance(sample_frame, pd.DataFrame):
        total += int(sample_frame.memory_usage(deep=True).sum())
    for name in (
        "genotype_matrix",
        "admix_haplotypes",
        "admix_local_ancestry_codes",
    ):
        value = entry.get(name)
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    by_population = entry.get("la_genotypes_by_population")
    if isinstance(by_population, dict):
        for value in by_population.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
    return total


def _load_sample_metadata(
    cohort: CohortConfig,
    *,
    default_population: str,
    caches: CohortInputCaches,
) -> tuple[List[str], List[str]]:
    if cohort.name in caches.sample_ids:
        sample_ids = caches.sample_ids[cohort.name]
    else:
        sample_ids = read_sample_ids(cohort.sample_ids_file)
        caches.sample_ids[cohort.name] = sample_ids

    if cohort.name in caches.sample_populations:
        sample_populations = caches.sample_populations[cohort.name]
    else:
        sample_populations = load_population_labels(
            cohort=cohort,
            sample_ids=sample_ids,
            default_population=default_population,
        )
        caches.sample_populations[cohort.name] = sample_populations
    return sample_ids, sample_populations


def _local_ancestry_pattern(run_cfg: RunConfig, cohort: CohortConfig) -> str | None:
    if str(run_cfg.local_ancestry.mode) == "source_zarr":
        return cohort.local_ancestry_source_pattern
    return cohort.local_ancestry_tracts_pattern


def _prepare_local_ancestry_inputs(
    *,
    run_cfg: RunConfig,
    cohort: CohortConfig,
    replicate: int,
    adapter: GenotypeAdapter,
    sample_ids: List[str],
    sample_populations: List[str],
    variant_ids: List[str],
    variant_indices_by_chrom: dict[int, np.ndarray],
    variant_pos_bp: np.ndarray,
    local_ancestry_pattern: str,
    cohort_tmp: Path,
    caches: CohortInputCaches,
    timed: TimingCallback,
) -> dict[str, Any]:
    ancestry_order = list(run_cfg.local_ancestry.ancestry_order)
    ancestry_to_index = {ancestry: idx for idx, ancestry in enumerate(ancestry_order)}
    fallback_prefixes = {
        str(rule.sample_id_prefix)
        for rule in run_cfg.local_ancestry.donor_mixture_fallbacks
    }

    truth_admix_indices: List[int] = []
    truth_admix_sample_ids: List[str] = []
    truth_admix_sample_pops: List[str] = []
    donor_fallback_indices: List[int] = []
    donor_fallback_sample_ids: List[str] = []
    donor_fallback_sample_pops: List[str] = []
    for idx, population in enumerate(sample_populations):
        if str(population).upper() != "ADMIX":
            continue
        sample_id = str(sample_ids[idx])
        sample_population = str(population)
        if _sample_id_prefix(sample_id) in fallback_prefixes:
            donor_fallback_indices.append(int(idx))
            donor_fallback_sample_ids.append(sample_id)
            donor_fallback_sample_pops.append(sample_population)
        else:
            truth_admix_indices.append(int(idx))
            truth_admix_sample_ids.append(sample_id)
            truth_admix_sample_pops.append(sample_population)

    admix_haplotypes: np.ndarray | None = None
    ancestry_codes: np.ndarray | None = None
    if truth_admix_sample_ids:
        started = time.perf_counter()
        admix_haplotypes = adapter.export_haplotypes(
            sample_ids=truth_admix_sample_ids,
            sample_populations=truth_admix_sample_pops,
            variant_ids=variant_ids,
            out_dir=cohort_tmp,
            chromosomes=run_cfg.chromosomes,
        )
        timed(
            "simulate.export_haplotypes_admix",
            started,
            replicate=int(replicate),
            cohort=str(cohort.name),
            n_samples=int(len(truth_admix_sample_ids)),
            n_variants=int(len(variant_ids)),
        )
        expected_shape = (len(truth_admix_sample_ids), len(variant_ids), 2)
        if admix_haplotypes.shape != expected_shape:
            raise ValueError(
                f"Unexpected ADMIX haplotype array shape: {admix_haplotypes.shape} "
                f"(expected {expected_shape})"
            )
        ancestry_codes = np.full(expected_shape, -1, dtype=np.int8)

        pattern = str(local_ancestry_pattern)
        for chrom in run_cfg.chromosomes:
            idx_for_chrom = variant_indices_by_chrom[int(chrom)]
            if int(idx_for_chrom.shape[0]) == 0:
                continue
            positions_bp = variant_pos_bp[idx_for_chrom]
            ancestry_path = (
                pattern.format(chrom=int(chrom)) if "{chrom}" in pattern else pattern
            )
            cache_key = (cohort.name, int(chrom), ancestry_path)

            if str(run_cfg.local_ancestry.mode) == "source_zarr":
                if cache_key in caches.ancestry_sources:
                    ancestry_source = caches.ancestry_sources[cache_key]
                else:
                    started = time.perf_counter()
                    ancestry_source = LocalAncestrySource.open(ancestry_path)
                    caches.ancestry_sources[cache_key] = ancestry_source
                    timed(
                        "simulate.load_ancestry_source",
                        started,
                        replicate=int(replicate),
                        cohort=str(cohort.name),
                        chrom=int(chrom),
                        n_admix_samples=int(len(truth_admix_sample_ids)),
                    )

                started = time.perf_counter()
                codes_subset = ancestry_source.rasterize(
                    sample_ids=truth_admix_sample_ids,
                    positions_bp=positions_bp,
                    ancestry_to_index=ancestry_to_index,
                    chunk_n_samples=int(run_cfg.local_ancestry.chunk_n_samples),
                )
                timed(
                    "simulate.rasterize_ancestry_source",
                    started,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    chrom=int(chrom),
                    n_admix_samples=int(len(truth_admix_sample_ids)),
                    n_variants=int(len(positions_bp)),
                )
            else:
                if cache_key in caches.tracts:
                    tracts_by_sample_hap = caches.tracts[cache_key]
                else:
                    started = time.perf_counter()
                    tracts_by_sample_hap = load_tracts_tsv_gz(
                        ancestry_path,
                        sample_ids=truth_admix_sample_ids,
                        label_map=dict(run_cfg.local_ancestry.label_map),
                    )
                    caches.tracts[cache_key] = tracts_by_sample_hap
                    timed(
                        "simulate.load_tracts",
                        started,
                        replicate=int(replicate),
                        cohort=str(cohort.name),
                        chrom=int(chrom),
                        n_admix_samples=int(len(truth_admix_sample_ids)),
                    )

                started = time.perf_counter()
                codes_subset = rasterize_tracts_to_variant_ancestry(
                    sample_ids=truth_admix_sample_ids,
                    tracts_by_sample_hap=tracts_by_sample_hap,
                    variant_positions_bp=positions_bp,
                    ancestry_to_index=ancestry_to_index,
                    chunk_n_samples=int(run_cfg.local_ancestry.chunk_n_samples),
                    use_numba=bool(run_cfg.local_ancestry.use_numba),
                )
                timed(
                    "simulate.rasterize_tracts",
                    started,
                    replicate=int(replicate),
                    cohort=str(cohort.name),
                    chrom=int(chrom),
                    n_admix_samples=int(len(truth_admix_sample_ids)),
                    n_variants=int(len(positions_bp)),
                )
            ancestry_codes[:, idx_for_chrom, :] = codes_subset.astype(
                np.int8, copy=False
            )

        if np.any(ancestry_codes < 0):
            bad = np.where(ancestry_codes < 0)
            first = (int(bad[0][0]), int(bad[1][0]), int(bad[2][0]))
            raise ValueError(
                "ADMIX local ancestry codes are incomplete after tracts rasterization. "
                f"First missing admix_sample_idx={first[0]} "
                f"variant_idx={first[1]} hap_idx={first[2]}"
            )

    return {
        "truth_admix_indices": truth_admix_indices,
        "truth_admix_sample_ids": truth_admix_sample_ids,
        "admix_haplotypes": admix_haplotypes,
        "admix_local_ancestry_codes": ancestry_codes,
        "donor_fallback_indices": donor_fallback_indices,
        "donor_fallback_sample_ids": donor_fallback_sample_ids,
        "donor_fallback_sample_pops": donor_fallback_sample_pops,
    }


def _prepare_genotype_inputs(
    *,
    run_cfg: RunConfig,
    cohort: CohortConfig,
    replicate: int,
    adapter: GenotypeAdapter,
    sample_ids: List[str],
    sample_populations: List[str],
    variant_ids: List[str],
    variant_indices_by_chrom: dict[int, np.ndarray],
    variant_pos_bp: np.ndarray,
    la_enabled: bool,
    local_ancestry_pattern: str | None,
    cohort_tmp: Path,
    caches: CohortInputCaches,
    timed: TimingCallback,
) -> dict[str, Any]:
    cache_key = (
        f"{replicate}|{cohort.name}|{run_cfg.local_ancestry.mode}|{la_enabled}|"
        f"{'|'.join(variant_ids)}"
    )
    if cache_key in caches.genotypes:
        caches.bump("simulate_genotype", hit=True)
        return caches.genotypes[cache_key]

    caches.bump("simulate_genotype", hit=False)
    sample_frame = _sample_frame(sample_ids)
    entry: dict[str, Any] = {
        "sample_frame": sample_frame,
        "genotype_matrix": None,
        "truth_admix_indices": None,
        "truth_admix_sample_ids": None,
        "admix_haplotypes": None,
        "admix_local_ancestry_codes": None,
        "donor_fallback_indices": None,
        "donor_fallback_sample_ids": None,
        "donor_fallback_sample_pops": None,
        "la_genotypes_by_population": {},
    }

    if la_enabled:
        if local_ancestry_pattern is None:
            raise ValueError("Local ancestry mode requires a cohort ancestry input")
        entry.update(
            _prepare_local_ancestry_inputs(
                run_cfg=run_cfg,
                cohort=cohort,
                replicate=replicate,
                adapter=adapter,
                sample_ids=sample_ids,
                sample_populations=sample_populations,
                variant_ids=variant_ids,
                variant_indices_by_chrom=variant_indices_by_chrom,
                variant_pos_bp=variant_pos_bp,
                local_ancestry_pattern=local_ancestry_pattern,
                cohort_tmp=cohort_tmp,
                caches=caches,
                timed=timed,
            )
        )
    else:
        started = time.perf_counter()
        if hasattr(adapter, "export_genotypes_u8"):
            entry["genotype_matrix"] = adapter.export_genotypes_u8(
                sample_ids=sample_ids,
                sample_populations=sample_populations,
                variant_ids=variant_ids,
                out_dir=cohort_tmp,
                chromosomes=run_cfg.chromosomes,
            )
            entry["sample_frame"] = _sample_frame(sample_ids)
            export_stage = "simulate.export_genotypes_u8"
        else:
            dosage_df = adapter.export_dosages(
                sample_ids=sample_ids,
                sample_populations=sample_populations,
                variant_ids=variant_ids,
                out_dir=cohort_tmp,
                chromosomes=run_cfg.chromosomes,
            )
            entry["sample_frame"] = dosage_df[["FID", "IID"]].copy()
            entry["genotype_matrix"] = dosage_df[variant_ids].to_numpy(
                dtype=np.float32, copy=False
            )
            export_stage = "simulate.export_dosages"
        timed(
            export_stage,
            started,
            replicate=int(replicate),
            cohort=str(cohort.name),
            n_samples=int(len(sample_ids)),
            n_variants=int(len(variant_ids)),
        )

    entry["dosage_effect_allele"] = dict(adapter.last_export_allele_by_variant)
    entry["dosage_other_allele"] = dict(adapter.last_export_other_allele_by_variant)
    caches.store_genotypes(cache_key, entry)
    return entry


def _prepare_background_grm(
    *,
    run_cfg: RunConfig,
    cohort: CohortConfig,
    replicate: int,
    adapter: GenotypeAdapter,
    sample_ids: List[str],
    sample_populations: List[str],
    variant_ids: List[str],
    configured_variant_ids: List[str],
    cohort_tmp: Path,
    timed: TimingCallback,
) -> tuple[np.ndarray | None, int, int]:
    if run_cfg.background_genetic.mode != "compute_grm":
        return None, 0, 0

    grm_variant_ids, excluded_count = _exclude_causal_grm_variants(
        configured_variant_ids=configured_variant_ids,
        causal_variant_ids=variant_ids,
    )
    started = time.perf_counter()
    if hasattr(adapter, "export_genotypes_u8"):
        background_genotypes = adapter.export_genotypes_u8(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            variant_ids=grm_variant_ids,
            out_dir=cohort_tmp / "background_grm",
        )
    else:
        background_dosages = adapter.export_dosages(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            variant_ids=grm_variant_ids,
            out_dir=cohort_tmp / "background_grm",
        )
        check_id_alignment(sample_ids, background_dosages)
        background_genotypes = background_dosages[grm_variant_ids].to_numpy(
            dtype=np.float32,
            copy=False,
        )
    factor = _standardize_grm_genotypes(background_genotypes)
    variant_count = int(factor.shape[1])
    timed(
        "simulate.prepare_background_grm",
        started,
        replicate=int(replicate),
        cohort=str(cohort.name),
        n_samples=int(len(sample_ids)),
        n_configured_variants=int(len(configured_variant_ids)),
        n_excluded_causal=int(excluded_count),
        n_polymorphic_variants=int(variant_count),
    )
    return factor, variant_count, int(excluded_count)


def prepare_cohort_inputs(
    *,
    run_cfg: RunConfig,
    cohort: CohortConfig,
    replicate: int,
    adapter: GenotypeAdapter,
    run_dir: Path,
    default_population: str,
    variant_ids: List[str],
    variant_indices_by_chrom: dict[int, np.ndarray],
    variant_pos_bp: np.ndarray,
    configured_background_variant_ids: List[str],
    caches: CohortInputCaches,
    timed: TimingCallback,
) -> PreparedCohortInputs:
    sample_ids, sample_populations = _load_sample_metadata(
        cohort,
        default_population=default_population,
        caches=caches,
    )
    local_ancestry_pattern = _local_ancestry_pattern(run_cfg, cohort)
    la_enabled = bool(run_cfg.local_ancestry.enabled and local_ancestry_pattern)

    if caches.runtime_cache is not None and hasattr(adapter, "set_io_threads"):
        adapter.set_io_threads(
            int(caches.runtime_cache.get("hapstore_io_threads", 1))
        )

    cohort_tmp = run_dir / "tmp" / "dosages" / f"rep{replicate}" / cohort.name
    genotype_inputs = _prepare_genotype_inputs(
        run_cfg=run_cfg,
        cohort=cohort,
        replicate=replicate,
        adapter=adapter,
        sample_ids=sample_ids,
        sample_populations=sample_populations,
        variant_ids=variant_ids,
        variant_indices_by_chrom=variant_indices_by_chrom,
        variant_pos_bp=variant_pos_bp,
        la_enabled=la_enabled,
        local_ancestry_pattern=local_ancestry_pattern,
        cohort_tmp=cohort_tmp,
        caches=caches,
        timed=timed,
    )
    sample_frame = genotype_inputs["sample_frame"]
    check_id_alignment(sample_ids, sample_frame)

    grm_factor, grm_variant_count, grm_excluded_count = _prepare_background_grm(
        run_cfg=run_cfg,
        cohort=cohort,
        replicate=replicate,
        adapter=adapter,
        sample_ids=sample_ids,
        sample_populations=sample_populations,
        variant_ids=variant_ids,
        configured_variant_ids=configured_background_variant_ids,
        cohort_tmp=cohort_tmp,
        timed=timed,
    )

    return PreparedCohortInputs(
        sample_ids=sample_ids,
        sample_populations=sample_populations,
        sample_frame=sample_frame,
        genotype_matrix=genotype_inputs.get("genotype_matrix"),
        truth_admix_indices=genotype_inputs.get("truth_admix_indices"),
        truth_admix_sample_ids=genotype_inputs.get("truth_admix_sample_ids"),
        admix_haplotypes=genotype_inputs.get("admix_haplotypes"),
        admix_local_ancestry_codes=genotype_inputs.get(
            "admix_local_ancestry_codes"
        ),
        donor_fallback_indices=genotype_inputs.get("donor_fallback_indices"),
        donor_fallback_sample_ids=genotype_inputs.get(
            "donor_fallback_sample_ids"
        ),
        donor_fallback_sample_pops=genotype_inputs.get(
            "donor_fallback_sample_pops"
        ),
        la_genotypes_by_population=genotype_inputs.get(
            "la_genotypes_by_population", {}
        ),
        dosage_effect_allele=dict(genotype_inputs["dosage_effect_allele"]),
        dosage_other_allele=dict(genotype_inputs["dosage_other_allele"]),
        background_grm_factor=grm_factor,
        background_grm_variant_count=grm_variant_count,
        background_grm_excluded_causal_count=grm_excluded_count,
        la_enabled=la_enabled,
        cohort_tmp=cohort_tmp,
    )


__all__ = [
    "CohortInputCaches",
    "PreparedCohortInputs",
    "prepare_cohort_inputs",
]
