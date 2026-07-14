"""HapStore-native main QC matrix backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import numpy as np
from hapstore import HapStore

from ..config import RunConfig
from ..utils import derived_seed
from .metrics.fst import (
    hapstore_counts_for_samples,
    hudson_fst_from_counts as matrix_hudson_fst_from_counts,
    sample_population_indices,
    sample_variant_indices,
)
from .types import QcInputs
from .policy_support import pair_key


def _threads_per_matrix_worker(total_threads: int | None, workers: int) -> int | None:
    if total_threads is None:
        return None
    return max(1, int(total_threads) // max(1, int(workers)))


def validate_population_variant_axes_impl(stores: dict[str, HapStore], chrom: int) -> int:
    pops = sorted(stores.keys())
    if not pops:
        raise ValueError("population_hapstores cannot be empty")
    ref_pop = pops[0]
    ref_info = stores[ref_pop].variant_info(chrom)
    ref_pos = np.asarray(ref_info["pos"])
    for pop in pops[1:]:
        cur_info = stores[pop].variant_info(chrom)
        fields_match = all(
            np.array_equal(np.asarray(cur_info[field]), np.asarray(ref_info[field]))
            for field in ("pos", "ref", "alt", "vid")
        )
        if not fields_match:
            raise ValueError(
                f"variant axis mismatch between populations {ref_pop!r} and {pop!r} "
                f"in HapStore QC for chr{chrom}"
            )
    return int(ref_pos.shape[0])


def run_qc_matrix_hapstore_impl(
    *,
    config: RunConfig,
    chrom: int,
    inputs: QcInputs,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    within_checks: list[str],
    numba_threads: int | None,
    logger: logging.Logger,
) -> tuple[dict[str, float], dict[str, float]]:
    sampled = config.qc.main_pairwise_mode == "sampled"
    max_dip = int(config.qc.sampled_max_diploids_per_population) if sampled else None
    variant_block = int(config.qc.matrix_variant_block_size)
    pop_seed_base = int(derived_seed(int(config.qc.seed), "main_qc_pop", chrom))

    per_pop_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_pop_indices: dict[str, np.ndarray] = {}

    def _subselect_counts(
        ac_full: np.ndarray,
        an_full: np.ndarray,
        variant_idx: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if variant_idx is None:
            return ac_full, an_full
        return ac_full[variant_idx], an_full[variant_idx]

    if inputs.combined_hapstore is not None:
        hs = HapStore(inputs.combined_hapstore, mode="r")
        n_variants = hs.n_variants(chrom)
        variant_indices = sample_variant_indices(
            n_variants=n_variants,
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "main_qc_variants", chrom)),
        )
        all_sample_ids = hs.sample_ids(chrom)
        pop_indices = sample_population_indices(
            all_sample_ids=all_sample_ids,
            groups=groups,
            max_diploids_per_population=max_dip,
            seed=pop_seed_base,
        )
        per_pop_indices.update(pop_indices)
        max_workers = min(max(1, int(config.qc.matrix_workers)), max(1, len(pop_indices)))
        count_numba_threads = _threads_per_matrix_worker(numba_threads, max_workers)
        selected_variant_count = int(variant_indices.size) if variant_indices is not None else int(n_variants)
        logger.info(
            "main QC matrix count-build strategy mode=combined_hapstore chrom=%s strategy=%s total_variants=%s selected_variants=%s",
            chrom,
            "full_then_subselect",
            int(n_variants),
            selected_variant_count,
        )

        def _count_pop(pop: str) -> tuple[str, np.ndarray, np.ndarray]:
            ac_full, an_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=pop_indices[pop],
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=count_numba_threads,
            )
            ac, an = _subselect_counts(ac_full, an_full, variant_indices)
            return pop, ac, an

        if max_workers > 1 and len(pop_indices) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_count_pop, pop) for pop in sorted(pop_indices.keys())]
                for fut in as_completed(futures):
                    pop, ac, an = fut.result()
                    per_pop_counts[pop] = (ac, an)
        else:
            for pop in sorted(pop_indices.keys()):
                pop, ac, an = _count_pop(pop)
                per_pop_counts[pop] = (ac, an)
    elif inputs.population_hapstores:
        stores = {pop: HapStore(path, mode="r") for pop, path in inputs.population_hapstores.items()}
        n_variants = validate_population_variant_axes_impl(stores, chrom=chrom)
        variant_indices = sample_variant_indices(
            n_variants=n_variants,
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "main_qc_variants", chrom)),
        )
        selected_variant_count = int(variant_indices.size) if variant_indices is not None else int(n_variants)
        pops = sorted(stores.keys())
        max_workers = min(max(1, int(config.qc.matrix_workers)), max(1, len(pops)))
        count_numba_threads = _threads_per_matrix_worker(numba_threads, max_workers)
        logger.info(
            "main QC matrix count-build strategy mode=population_hapstores chrom=%s strategy=%s total_variants=%s selected_variants=%s",
            chrom,
            "full_then_subselect",
            int(n_variants),
            selected_variant_count,
        )

        def _count_pop(pop: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
            hs = stores[pop]
            all_ids = hs.sample_ids(chrom)
            pop_idx = sample_population_indices(
                all_sample_ids=all_ids,
                groups={pop: groups[pop]},
                max_diploids_per_population=max_dip,
                seed=int(derived_seed(pop_seed_base, pop)),
            )[pop]
            ac_full, an_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=pop_idx,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=count_numba_threads,
            )
            ac, an = _subselect_counts(ac_full, an_full, variant_indices)
            return pop, ac, an, pop_idx

        if max_workers > 1 and len(pops) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_count_pop, pop) for pop in pops]
                for fut in as_completed(futures):
                    pop, ac, an, pop_idx = fut.result()
                    per_pop_counts[pop] = (ac, an)
                    per_pop_indices[pop] = pop_idx
        else:
            for pop in pops:
                pop, ac, an, pop_idx = _count_pop(pop)
                per_pop_counts[pop] = (ac, an)
                per_pop_indices[pop] = pop_idx
    else:
        raise ValueError("matrix QC requested but no HapStore inputs are available")

    pairwise: dict[str, float] = {}
    for a, b in pairs:
        if a not in per_pop_counts or b not in per_pop_counts:
            raise ValueError(f"pairwise populations not available in HapStore counts: ({a}, {b})")
        ac_a, an_a = per_pop_counts[a]
        ac_b, an_b = per_pop_counts[b]
        pairwise[pair_key(a, b)] = matrix_hudson_fst_from_counts(
            ac_a,
            an_a,
            ac_b,
            an_b,
            numba_threads=numba_threads,
        )

    within: dict[str, float] = {}
    if inputs.combined_hapstore is not None:
        hs = HapStore(inputs.combined_hapstore, mode="r")
        variant_indices_within = sample_variant_indices(
            n_variants=hs.n_variants(chrom),
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "within_variants", chrom)),
        )
        for pop in within_checks:
            indices = per_pop_indices.get(pop)
            if indices is None or len(indices) < 4:
                within[pop] = float("nan")
                continue
            rng = np.random.default_rng(int(derived_seed(int(config.qc.seed), "within", chrom, pop)))
            shuffled = np.asarray(indices, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            split_a = shuffled[:half]
            split_b = shuffled[half:]
            ac_a_full, an_a_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_a,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_b_full, an_b_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_b,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_a, an_a = _subselect_counts(ac_a_full, an_a_full, variant_indices_within)
            ac_b, an_b = _subselect_counts(ac_b_full, an_b_full, variant_indices_within)
            within[pop] = matrix_hudson_fst_from_counts(
                ac_a,
                an_a,
                ac_b,
                an_b,
                numba_threads=numba_threads,
            )
    else:
        stores = {pop: HapStore(path, mode="r") for pop, path in inputs.population_hapstores.items()}
        variant_indices_within = sample_variant_indices(
            n_variants=validate_population_variant_axes_impl(stores, chrom=chrom),
            max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
            seed=int(derived_seed(int(config.qc.seed), "within_variants", chrom)),
        )
        for pop in within_checks:
            hs = stores.get(pop)
            indices = per_pop_indices.get(pop)
            if hs is None or indices is None or len(indices) < 4:
                within[pop] = float("nan")
                continue
            rng = np.random.default_rng(int(derived_seed(int(config.qc.seed), "within", chrom, pop)))
            shuffled = np.asarray(indices, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            split_a = shuffled[:half]
            split_b = shuffled[half:]
            ac_a_full, an_a_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_a,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_b_full, an_b_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=split_b,
                variant_indices=None,
                variant_block_size=variant_block,
                numba_threads=numba_threads,
            )
            ac_a, an_a = _subselect_counts(ac_a_full, an_a_full, variant_indices_within)
            ac_b, an_b = _subselect_counts(ac_b_full, an_b_full, variant_indices_within)
            within[pop] = matrix_hudson_fst_from_counts(
                ac_a,
                an_a,
                ac_b,
                an_b,
                numba_threads=numba_threads,
            )

    return pairwise, within
