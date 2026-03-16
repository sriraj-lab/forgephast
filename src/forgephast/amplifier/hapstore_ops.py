"""Wrappers around HapStore APIs for merge and ingest operations."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np

from hapstore import (
    HapStore,
    concat_hapstores as _pkg_concat_hapstores,
    ingest_arrays as _pkg_ingest_arrays,
    merge_chromosomes_sample_axis as _pkg_merge_chromosomes_sample_axis,
)

logger = logging.getLogger(__name__)
_VALID_HAPSTORE_MERGE_STRATEGIES = {"streaming", "legacy_materialize"}


def _is_stale_hapstore_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("already exists" in msg) and (
        ("chromosome" in msg)
        or ("chrom" in msg)
        or ("group" in msg)
        or ("zarr" in msg)
    )


def ingest_arrays(*args, **kwargs):
    """Compatibility ingest wrapper with stale-store overwrite retry.

    If a prior failed run left a stale chromosome/group in the destination Zarr
    store, the hapstore library may raise an "already exists" style error. In
    that case we remove the destination store and retry once.
    """
    output_zarr = kwargs.get("output_zarr", args[0] if args else None)
    output_path = Path(output_zarr) if output_zarr is not None else None

    try:
        return _pkg_ingest_arrays(*args, **kwargs)
    except Exception as exc:
        if output_path is None or not output_path.exists() or not _is_stale_hapstore_error(exc):
            raise
        logger.warning(
            "Detected stale HapStore at %s after ingest error; removing and retrying once: %s",
            output_path,
            exc,
        )
        shutil.rmtree(output_path, ignore_errors=True)
        return _pkg_ingest_arrays(*args, **kwargs)


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
    merge_strategy: str = "streaming",
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
            merge_strategy=merge_strategy,
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
                    merge_strategy=merge_strategy,
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
            merge_strategy=merge_strategy,
            n_workers=n_workers,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    merge_strategy: str = "streaming",
    n_workers: int = 1,
) -> HapStore:
    """Merge stores along sample axis.

    - streaming: use hapstore concat_hapstores (preferred) or merge_chromosomes_sample_axis.
    - legacy_materialize: use local in-memory concatenation fallback.
    """
    if merge_strategy not in _VALID_HAPSTORE_MERGE_STRATEGIES:
        raise ValueError(
            f"merge_strategy must be one of {_VALID_HAPSTORE_MERGE_STRATEGIES}, got {merge_strategy!r}"
        )
    if int(n_workers) <= 0:
        raise ValueError("n_workers must be > 0")

    out_path = Path(output_store)
    if output_mode == "w" and out_path.exists():
        shutil.rmtree(out_path, ignore_errors=True)

    if merge_strategy == "legacy_materialize":
        logger.warning(
            "Using legacy materializing HapStore merge strategy; this can be "
            "substantially more memory intensive on large chromosomes."
        )
        return _fallback_merge_chromosomes_sample_axis(
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
        )

    # Preferred path: use concat_hapstores for overwrite mode.
    if output_mode == "w":
        try:
            return _pkg_concat_hapstores(
                input_stores=input_stores,
                output_store=output_store,
                chrom=chrom,
                axis="samples",
                validate="spot",
                n_workers=int(n_workers),
                compression_level=compression_level,
                chunk_variant_size=chunk_variant_size,
                chunk_haplotype_size=chunk_haplotype_size,
                chunk_sample_size=chunk_sample_size,
                store_genotypes=store_genotypes,
                attrs=attrs,
                overwrite=True,
            )
        except AssertionError as exc:
            # Defensive retry: if parallel concat spot-validation fails on a given
            # platform/filesystem, retry serially before falling back.
            if "spot validation failed" in str(exc).lower() and int(n_workers) > 1:
                logger.warning(
                    "Parallel HapStore concat failed spot validation; retrying serial concat "
                    "with n_workers=1. error=%s",
                    exc,
                )
                return _pkg_concat_hapstores(
                    input_stores=input_stores,
                    output_store=output_store,
                    chrom=chrom,
                    axis="samples",
                    validate="spot",
                    n_workers=1,
                    compression_level=compression_level,
                    chunk_variant_size=chunk_variant_size,
                    chunk_haplotype_size=chunk_haplotype_size,
                    chunk_sample_size=chunk_sample_size,
                    store_genotypes=store_genotypes,
                    attrs=attrs,
                    overwrite=True,
                )
            raise

    # Non-overwrite mode: fall back to merge_chromosomes_sample_axis.
    if output_mode != "w":
        logger.warning(
            "concat_hapstores supports overwrite semantics only; output_mode=%s requested. "
            "Falling back to merge_chromosomes_sample_axis.",
            output_mode,
        )

    return _pkg_merge_chromosomes_sample_axis(
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
    )


def _fallback_merge_chromosomes_sample_axis(
    *,
    input_stores: Sequence[str | Path],
    output_store: str | Path,
    chrom: str | int,
    output_mode: str,
    chunk_variant_size: int | None,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int | None,
    store_genotypes: bool,
    attrs: dict | None,
) -> HapStore:
    if not input_stores:
        raise ValueError("input_stores must not be empty")

    stores = [HapStore(path, mode="r") for path in input_stores]

    first = stores[0]
    variant_info = first.variant_info(chrom)
    variant_pos = first.variant_positions(chrom)
    genetic_pos = first.genetic_positions(chrom)

    merged_haps: list[np.ndarray] = []
    merged_samples: list[str] = []
    for idx, hs in enumerate(stores[1:], start=1):
        cur_pos = hs.variant_positions(chrom)
        if cur_pos.shape != variant_pos.shape or not np.array_equal(cur_pos, variant_pos):
            raise ValueError(
                "fallback hapstore merge detected variant mismatch between "
                f"store[0]={stores[0].path} and store[{idx}]={hs.path}"
            )

    for hs in stores:
        merged_haps.append(hs.haplotypes(chrom))
        merged_samples.extend(hs.sample_ids(chrom))

    haps = np.concatenate(merged_haps, axis=0)
    return ingest_arrays(
        output_zarr=output_store,
        chrom=chrom,
        haplotypes=haps,
        variant_pos=variant_pos,
        sample_ids=merged_samples,
        ref=variant_info["ref"],
        alt=variant_info["alt"],
        vid=variant_info["vid"],
        genetic_positions=genetic_pos,
        store_genotypes=store_genotypes,
        chunk_variant_size=(500 if chunk_variant_size is None else int(chunk_variant_size)),
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=(3 if compression_level is None else int(compression_level)),
        attrs=attrs,
    )
