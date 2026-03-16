"""Shared sampling + alignment for extended QC metrics.

This module centralizes deterministic sample/variant selection so multiple QC
metrics can reuse the same donor/synthetic cohorts and aligned variant axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import numpy as np

from .metrics.fst import sample_variant_indices
from ..utils import derived_seed


@dataclass(frozen=True)
class ExtPopData:
    pop: str
    donor_ids: list[str]
    synth_ids: list[str]
    donor_haps: np.ndarray  # (2N, V) uint8
    synth_haps: np.ndarray  # (2N, V) uint8


@dataclass(frozen=True)
class ExtAlignedData:
    chrom: int
    pops: list[str]
    global_pos: np.ndarray  # (V_common,) bp positions across all eligible pops
    sampled_pos: np.ndarray  # (V_sampled,) bp positions used by extended metrics
    variant_subset_id: str
    per_pop: dict[str, ExtPopData]


def _align_by_position(
    donor_pos: np.ndarray,
    donor_haps: np.ndarray,
    synth_pos: np.ndarray,
    synth_haps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donor_pos = np.asarray(donor_pos, dtype=np.int64)
    synth_pos = np.asarray(synth_pos, dtype=np.int64)
    if donor_haps.shape[1] != donor_pos.shape[0]:
        raise ValueError(
            "donor haplotype axis mismatch: "
            f"haps={donor_haps.shape}, n_positions={donor_pos.shape[0]}"
        )
    if synth_haps.shape[1] != synth_pos.shape[0]:
        raise ValueError(
            "synthetic haplotype axis mismatch: "
            f"haps={synth_haps.shape}, n_positions={synth_pos.shape[0]}"
        )
    if donor_pos.shape == synth_pos.shape and np.array_equal(donor_pos, synth_pos):
        return donor_pos, donor_haps, synth_haps

    common, idx_donor, idx_synth = np.intersect1d(
        donor_pos,
        synth_pos,
        assume_unique=False,
        return_indices=True,
    )
    if common.shape[0] < 2:
        raise ValueError(
            "Unable to align donor/synthetic variant axes by position; "
            f"common_positions={common.shape[0]}"
        )
    return (
        common.astype(np.int64, copy=False),
        donor_haps[:, idx_donor],
        synth_haps[:, idx_synth],
    )


def _variant_subset_id(
    *,
    chrom: int,
    sampled_pos: np.ndarray,
    seed: int,
    ext_max_diploids_per_population: int,
    ext_max_variants: int,
) -> str:
    pos = np.asarray(sampled_pos, dtype=np.int64)
    h = hashlib.sha1()
    h.update(f"chrom={int(chrom)};seed={int(seed)};n={int(ext_max_diploids_per_population)};v={int(ext_max_variants)};".encode())
    h.update(pos.tobytes(order="C"))
    return h.hexdigest()


def build_ext_aligned_data(
    *,
    chrom: int,
    seed: int,
    ext_max_diploids_per_population: int,
    ext_max_variants: int,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    inputs: Any,
    subsample_ids: Callable[[list[str], int, int], list[str]],
    resolve_panel_source_for_population: Callable[[int, str], tuple[str, Any]],
    load_sample_haplotypes_from_panel_source: Callable[[str, Any, int, list[str]], tuple[np.ndarray, np.ndarray]],
    load_synthetic_population_haplotypes: Callable[[Any, int, str, list[str]], tuple[np.ndarray, np.ndarray]],
) -> ExtAlignedData | None:
    """Prepare shared donor/synthetic sample IDs and aligned variant subset.

    Returns None when there are no eligible populations or no common positions.
    """
    pop_names = sorted(groups.keys())
    eligible: list[str] = []
    per_pop_raw: dict[str, dict[str, Any]] = {}

    for pop in pop_names:
        donors_all = donors_by_population.get(pop, [])
        synth_all = groups.get(pop, [])
        if not donors_all or not synth_all:
            continue

        donor_ids = subsample_ids(
            donors_all,
            min(len(donors_all), int(ext_max_diploids_per_population)),
            int(derived_seed(int(seed), "ext_donor_ids", pop, chrom)),
        )
        synth_ids = subsample_ids(
            synth_all,
            min(len(synth_all), int(ext_max_diploids_per_population)),
            int(derived_seed(int(seed), "ext_synth_ids", pop, chrom)),
        )
        if not donor_ids or not synth_ids:
            continue

        source_kind, source_path = resolve_panel_source_for_population(int(chrom), str(pop))
        donor_pos, donor_haps = load_sample_haplotypes_from_panel_source(
            str(source_kind),
            source_path,
            int(chrom),
            donor_ids,
        )
        synth_pos, synth_haps = load_synthetic_population_haplotypes(inputs, int(chrom), str(pop), synth_ids)
        aligned_pos, donor_haps_aligned, synth_haps_aligned = _align_by_position(
            donor_pos=donor_pos,
            donor_haps=donor_haps,
            synth_pos=synth_pos,
            synth_haps=synth_haps,
        )
        per_pop_raw[pop] = {
            "donor_ids": donor_ids,
            "synth_ids": synth_ids,
            "pos": np.asarray(aligned_pos, dtype=np.int64),
            "donor_haps": donor_haps_aligned,
            "synth_haps": synth_haps_aligned,
        }
        eligible.append(pop)

    pops = sorted(set(eligible))
    if not pops:
        return None

    global_pos = per_pop_raw[pops[0]]["pos"]
    for pop in pops[1:]:
        global_pos = np.intersect1d(global_pos, per_pop_raw[pop]["pos"], assume_unique=False)
    if global_pos.size < 2:
        return None

    variant_idx = sample_variant_indices(
        n_variants=int(global_pos.size),
        max_variants=int(ext_max_variants),
        seed=int(derived_seed(int(seed), "ext_variants", chrom)),
    )
    sampled_pos = global_pos if variant_idx is None else global_pos[variant_idx]

    per_pop: dict[str, ExtPopData] = {}
    for pop in pops:
        pos = per_pop_raw[pop]["pos"]
        idx = np.searchsorted(pos, sampled_pos)
        if idx.size != sampled_pos.size or not np.array_equal(pos[idx], sampled_pos):
            raise RuntimeError(f"position axis mismatch while building ext cache for pop={pop}")
        per_pop[pop] = ExtPopData(
            pop=str(pop),
            donor_ids=list(per_pop_raw[pop]["donor_ids"]),
            synth_ids=list(per_pop_raw[pop]["synth_ids"]),
            donor_haps=np.asarray(per_pop_raw[pop]["donor_haps"][:, idx], dtype=np.uint8),
            synth_haps=np.asarray(per_pop_raw[pop]["synth_haps"][:, idx], dtype=np.uint8),
        )

    subset_id = _variant_subset_id(
        chrom=int(chrom),
        sampled_pos=sampled_pos,
        seed=int(seed),
        ext_max_diploids_per_population=int(ext_max_diploids_per_population),
        ext_max_variants=int(ext_max_variants),
    )

    return ExtAlignedData(
        chrom=int(chrom),
        pops=pops,
        global_pos=np.asarray(global_pos, dtype=np.int64),
        sampled_pos=np.asarray(sampled_pos, dtype=np.int64),
        variant_subset_id=str(subset_id),
        per_pop=per_pop,
    )
