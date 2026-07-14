"""QC input resolution and HapStore-backed data access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hapstore import HapStore

from ..config import RunConfig
from ..donor_panels import load_donor_haplotypes_from_source
from .types import QcInputs


def resolve_donor_panel_source(config: RunConfig, chrom: int, pop: str) -> tuple[str, Path]:
    return config.resolve_donor_panel_source(chrom=int(chrom), population=str(pop))


def resolve_map_path(config: RunConfig, chrom: int, pop: str) -> Path:
    pattern = config.input.genetic_map_pattern
    candidates: list[Path] = []
    if "{population}" in pattern:
        candidates.append(Path(pattern.format(chrom=chrom, population=pop)))
        candidates.append(Path(pattern.format(chrom=chrom, population="")))
    else:
        candidates.append(Path(pattern.format(chrom=chrom)))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def subsample_ids(sample_ids: list[str], target_n: int, seed: int) -> list[str]:
    if target_n <= 0:
        raise ValueError("target_n must be > 0 for subsampling")
    unique_ids = sorted({str(value) for value in sample_ids})
    if len(unique_ids) <= target_n:
        return unique_ids
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(unique_ids), size=target_n, replace=False))
    return [unique_ids[int(index)] for index in indices.tolist()]


def load_sample_haplotypes_from_panel(
    *,
    config: RunConfig,
    source_path: Path,
    chrom: int,
    sample_ids: list[str],
    source_kind: str = "vcf",
) -> tuple[np.ndarray, np.ndarray]:
    panel, _, _ = load_donor_haplotypes_from_source(
        source_path=source_path,
        source_kind=source_kind,
        chrom=chrom,
        donor_ids=sample_ids,
        cache_dir=config.donor_cache_root,
        force_refresh=False,
        ingest_workers=max(1, int(config.simulation.donor_io_workers)),
    )
    return panel.variants.pos.astype(np.int64, copy=False), panel.donor_haps


def load_synthetic_population_haplotypes(
    *,
    config: RunConfig | None = None,
    inputs: QcInputs,
    chrom: int,
    pop: str,
    sample_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    del config

    def load_subset(store_path: Path, *, missing_context: str) -> tuple[np.ndarray, np.ndarray]:
        store = HapStore(store_path, mode="r")
        all_ids = [str(value) for value in store.sample_ids(chrom)]
        index_by_id = {sample_id: index for index, sample_id in enumerate(all_ids)}
        missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} synthetic samples for {pop} missing in {missing_context}; "
                f"first={missing[:5]}"
            )
        requested = np.asarray([index_by_id[sample_id] for sample_id in sample_ids], dtype=np.int64)
        mask = np.zeros(len(all_ids), dtype=bool)
        mask[requested] = True
        haplotypes = store.haplotypes(chrom, sample_mask=mask)

        selected = np.flatnonzero(mask).astype(np.int64, copy=False)
        positions = np.searchsorted(selected, requested)
        if positions.size != requested.size or not np.array_equal(selected[positions], requested):
            raise RuntimeError(f"Unable to reorder {missing_context} haplotypes to requested sample order")
        haplotype_rows = np.empty(positions.size * 2, dtype=np.int64)
        haplotype_rows[0::2] = 2 * positions
        haplotype_rows[1::2] = (2 * positions) + 1
        return store.variant_positions(chrom), haplotypes[haplotype_rows, :]

    if inputs.combined_hapstore is not None:
        return load_subset(inputs.combined_hapstore, missing_context="combined hapstore")
    if pop in inputs.population_hapstores:
        return load_subset(inputs.population_hapstores[pop], missing_context="population hapstore")
    raise ValueError(
        f"Synthetic population {pop!r} is not available in HapStore QC inputs; "
        "run simulation/merge with outputs.write_hapstore=true"
    )


def sample_groups_from_manifest(sample_manifest_path: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(sample_manifest_path, sep="\t", dtype=str)
    if "population" not in frame.columns or "sample_id" not in frame.columns:
        raise ValueError(f"sample manifest missing population/sample_id columns: {sample_manifest_path}")
    return {
        str(population): sorted(group["sample_id"].astype(str).unique().tolist())
        for population, group in frame.groupby("population")
    }


def resolve_synthetic_outputs(config: RunConfig, chrom: int, n_shards: int) -> QcInputs:
    merged_manifest = config.run_root / "manifests" / f"merged_samples_chr{chrom}.tsv"
    merged_hapstore = (
        config.run_root / "merged" / f"chr{chrom}" / f"combined.chr{chrom}.merged.hapstore.zarr"
    )
    if merged_hapstore.exists() and merged_manifest.exists():
        return QcInputs(
            mode="combined_merged_hapstore",
            sample_manifest=merged_manifest,
            combined_vcf=None,
            combined_hapstore=merged_hapstore,
            population_hapstores={},
        )

    population_dir = config.run_root / "merged" / f"chr{chrom}" / "populations"
    population_hapstores = {
        population.name: population_dir / f"{population.name}.chr{chrom}.merged.hapstore.zarr"
        for population in config.populations
        if not population.donor_only
    }
    if (
        merged_manifest.exists()
        and population_hapstores
        and all(path.exists() for path in population_hapstores.values())
    ):
        return QcInputs(
            mode="population_merged_hapstore",
            sample_manifest=merged_manifest,
            combined_vcf=None,
            combined_hapstore=None,
            population_hapstores=population_hapstores,
        )

    if n_shards != 1:
        raise FileNotFoundError(
            "Merged HapStore outputs not found for multi-shard QC; run merge first with "
            "outputs.write_hapstore=true."
        )

    shard_dir = config.run_root / "synthetic" / f"chr{chrom}" / "shard_0_of_1"
    single_vcf = shard_dir / f"combined.chr{chrom}.vcf.gz"
    single_hapstore = shard_dir / f"combined.chr{chrom}.hapstore.zarr"
    single_manifest = config.run_root / "manifests" / f"synthetic_samples_chr{chrom}_shard_0_of_1.tsv"
    if single_hapstore.exists() and single_manifest.exists():
        return QcInputs(
            mode="single_shard_combined_hapstore",
            sample_manifest=single_manifest,
            combined_vcf=single_vcf if single_vcf.exists() else None,
            combined_hapstore=single_hapstore,
            population_hapstores={},
        )

    raise FileNotFoundError(
        "Expected HapStore-backed synthetic outputs for QC were not found for "
        f"chr{chrom}: {single_hapstore} / {single_manifest}. "
        "Run simulation with outputs.write_hapstore=true."
    )
