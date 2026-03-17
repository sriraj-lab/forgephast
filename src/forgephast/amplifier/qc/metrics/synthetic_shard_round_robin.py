"""Post-merge shard-vs-shard synthetic QC metric."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from hapstore import HapStore
import numpy as np
import pandas as pd

from ...telemetry import RateLimitedProgress, telemetry_span
from ...utils import derived_seed
from ..round_robin_support import pair_count, subset_shards_for_target_pairs, summarize_fst_values
from .fst import (
    hapstore_counts_for_samples,
    hudson_fst_from_counts as matrix_hudson_fst_from_counts,
    sample_variant_indices,
)

logger = logging.getLogger(__name__)


def _empty_payload(*, chrom: int, table_path: Path, pairwise_mode: str, status: str) -> dict[str, Any]:
    pd.DataFrame(
        columns=["population", "chrom", "shard_a", "shard_b", "n_a", "n_b", "fst"]
    ).to_csv(table_path, sep="\t", index=False)
    return {
        "enabled": True,
        "chrom": int(chrom),
        "status": str(status),
        "gate_pass": None,
        "table_path": str(table_path),
        "population_summary": {},
        "n_candidate_pairs": 0,
        "n_selected_pairs": 0,
        "n_records": 0,
        "pairwise_mode": str(pairwise_mode),
    }


def _group_samples_by_population_and_shard(sample_manifest: Path) -> dict[str, dict[int, list[str]]]:
    df = pd.read_csv(sample_manifest, sep="\t", dtype=str)
    required = {"population", "sample_id", "shard_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"sample manifest missing required columns {missing}: {sample_manifest}")
    shard_numeric = pd.to_numeric(df["shard_id"], errors="coerce")
    if shard_numeric.isna().any():
        raise ValueError(f"sample manifest has non-numeric shard_id values: {sample_manifest}")
    df = df.assign(shard_id=shard_numeric.astype(int))
    groups: dict[str, dict[int, list[str]]] = {}
    for (pop, shard_id), grp in df.groupby(["population", "shard_id"], sort=True):
        sample_ids = sorted(grp["sample_id"].astype(str).unique().tolist())
        if sample_ids:
            groups.setdefault(str(pop), {})[int(shard_id)] = sample_ids
    return groups


def _sample_indices_for_ids(hs: HapStore, chrom: int, sample_ids: list[str], *, context: str) -> np.ndarray:
    all_ids = [str(x) for x in hs.sample_ids(chrom)]
    idx_by_id = {sid: idx for idx, sid in enumerate(all_ids)}
    missing = [sid for sid in sample_ids if sid not in idx_by_id]
    if missing:
        raise ValueError(f"{len(missing)} samples missing in {context}; first={missing[:5]}")
    return np.asarray([idx_by_id[sid] for sid in sample_ids], dtype=np.int64)


def _summary_with_shards(
    *,
    fst_values: list[float],
    total_samples: int,
    shard_ids: list[int],
    shard_sizes: dict[int, int],
    candidate_pairs_total: int,
    selected_pairs: list[tuple[int, int]],
    max_pairs_per_population: int | None,
) -> dict[str, float | int]:
    summary = summarize_fst_values(fst_values, n_donors=total_samples, n_folds=len(shard_ids))
    summary["n_shards"] = int(len(shard_ids))
    summary["n_candidate_pairs"] = int(candidate_pairs_total)
    summary["n_selected_pairs"] = int(len(selected_pairs))
    summary["shard_min_size"] = int(min(shard_sizes.values())) if shard_sizes else 0
    summary["shard_max_size"] = int(max(shard_sizes.values())) if shard_sizes else 0
    if max_pairs_per_population is not None:
        summary["rr_target_pairs"] = int(max_pairs_per_population)
        summary["rr_shard_subset_size"] = int(len(sorted({sid for pair in selected_pairs for sid in pair})))
    return summary


def _compute_population_metric(
    *,
    config: Any,
    chrom: int,
    pop: str,
    shard_to_sample_ids: dict[int, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    combined_hapstore: Path | None,
    population_hapstore: Path | None,
) -> tuple[list[dict[str, float | int | str]], dict[str, Any], int, int]:
    sampled = str(config.qc.synthetic_shard_round_robin_pairwise_mode) == "sampled"
    max_pairs_per_population = config.qc.synthetic_shard_round_robin_max_pairs_per_population
    total_threads = max(1, int(numba_threads))
    min_threads_per_match = max(1, int(config.qc.synthetic_shard_round_robin_min_cpus_per_match))
    shard_ids = sorted(int(x) for x in shard_to_sample_ids.keys() if shard_to_sample_ids[x])
    shard_sizes = {int(shard_id): len(shard_to_sample_ids[shard_id]) for shard_id in shard_ids}
    total_samples = int(sum(shard_sizes.values()))
    candidate_pairs_total = pair_count(len(shard_ids))

    if len(shard_ids) <= 1:
        summary = _summary_with_shards(
            fst_values=[],
            total_samples=total_samples,
            shard_ids=shard_ids,
            shard_sizes=shard_sizes,
            candidate_pairs_total=candidate_pairs_total,
            selected_pairs=[],
            max_pairs_per_population=max_pairs_per_population,
        )
        summary["status"] = "insufficient_shards"
        return [], summary, int(candidate_pairs_total), 0

    if max_pairs_per_population is not None:
        shard_ids_used = subset_shards_for_target_pairs(
            shard_ids=shard_ids,
            target_pairs=int(max_pairs_per_population),
            seed=int(derived_seed(int(config.qc.seed), "synthetic_shard_rr_subset", pop, chrom)),
        )
        selected_pairs = list(combinations(shard_ids_used, 2))
    else:
        selected_pairs = list(combinations(shard_ids, 2))

    if not selected_pairs:
        summary = _summary_with_shards(
            fst_values=[],
            total_samples=total_samples,
            shard_ids=shard_ids,
            shard_sizes=shard_sizes,
            candidate_pairs_total=candidate_pairs_total,
            selected_pairs=[],
            max_pairs_per_population=max_pairs_per_population,
        )
        summary["status"] = "no_selected_pairs"
        return [], summary, int(candidate_pairs_total), 0

    needed_shards = sorted({int(sid) for pair in selected_pairs for sid in pair})
    rr_threads_per_match = min(max(1, int(min_threads_per_match)), total_threads)
    rr_max_parallel_matches = max(1, total_threads // rr_threads_per_match)
    ingest_workers = min(len(needed_shards), total_threads)
    ingest_threads_per_worker = max(1, total_threads // max(1, ingest_workers))
    counts_by_shard: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    hs = HapStore(combined_hapstore if combined_hapstore is not None else population_hapstore, mode="r")
    variant_indices = sample_variant_indices(
        n_variants=hs.n_variants(chrom),
        max_variants=(int(config.qc.sampled_max_variants) if sampled else None),
        seed=int(derived_seed(int(config.qc.seed), "synthetic_shard_rr_variants", pop, chrom)),
    )

    with telemetry_span(
        logger,
        "qc.synthetic_shard_round_robin.build_counts",
        chrom=int(chrom),
        pop=str(pop),
        selected_pairs=int(len(selected_pairs)),
        needed_shards=int(len(needed_shards)),
        sampled=bool(sampled),
        ingest_workers=int(ingest_workers),
        ingest_threads_per_worker=int(ingest_threads_per_worker),
    ):
        count_progress = RateLimitedProgress(
            logger=logger,
            name="qc.synthetic_shard_round_robin.build_counts",
            every_sec=30.0,
        )

        def _count_one(shard_id: int) -> tuple[int, tuple[np.ndarray, np.ndarray]]:
            sample_indices = _sample_indices_for_ids(
                hs,
                chrom,
                shard_to_sample_ids[shard_id],
                context=f"synthetic shard round-robin pop={pop} shard={shard_id}",
            )
            ac_full, an_full = hapstore_counts_for_samples(
                store=hs,
                chrom=chrom,
                sample_indices=sample_indices,
                variant_indices=None,
                variant_block_size=int(config.qc.matrix_variant_block_size),
                numba_threads=ingest_threads_per_worker,
            )
            if variant_indices is None:
                return shard_id, (ac_full, an_full)
            return shard_id, (ac_full[variant_indices], an_full[variant_indices])

        if ingest_workers > 1 and len(needed_shards) > 1:
            with ThreadPoolExecutor(max_workers=ingest_workers) as pool:
                futures = [pool.submit(_count_one, shard_id) for shard_id in needed_shards]
                done = 0
                for fut in as_completed(futures):
                    shard_id, counts = fut.result()
                    counts_by_shard[shard_id] = counts
                    done += 1
                    count_progress.log(
                        pop=str(pop),
                        chrom=int(chrom),
                        done=int(done),
                        total=int(len(needed_shards)),
                        last_shard=int(shard_id),
                    )
        else:
            for shard_id in needed_shards:
                shard_id_out, counts = _count_one(shard_id)
                counts_by_shard[shard_id_out] = counts
                count_progress.log(
                    pop=str(pop),
                    chrom=int(chrom),
                    done=int(len(counts_by_shard)),
                    total=int(len(needed_shards)),
                    last_shard=int(shard_id_out),
                )

    records: list[dict[str, float | int | str]] = []
    fst_values: list[float] = []
    pair_progress = RateLimitedProgress(
        logger=logger,
        name="qc.synthetic_shard_round_robin.compute_pairs",
        every_sec=30.0,
    )
    with telemetry_span(
        logger,
        "qc.synthetic_shard_round_robin.compute_pairs",
        chrom=int(chrom),
        pop=str(pop),
        selected_pairs=int(len(selected_pairs)),
        threads_per_match=int(rr_threads_per_match),
        max_parallel=int(rr_max_parallel_matches),
    ):
        t_pairs = perf_counter()

        def _compute_pair(pair: tuple[int, int]) -> dict[str, float | int | str]:
            shard_a, shard_b = pair
            ac_a, an_a = counts_by_shard[shard_a]
            ac_b, an_b = counts_by_shard[shard_b]
            fst = matrix_hudson_fst_from_counts(
                ac_a,
                an_a,
                ac_b,
                an_b,
                numba_threads=rr_threads_per_match,
            )
            return {
                "population": pop,
                "chrom": int(chrom),
                "shard_a": int(shard_a),
                "shard_b": int(shard_b),
                "n_a": int(shard_sizes[shard_a]),
                "n_b": int(shard_sizes[shard_b]),
                "fst": float(fst),
            }

        if rr_max_parallel_matches > 1 and len(selected_pairs) > 1:
            with ThreadPoolExecutor(max_workers=min(rr_max_parallel_matches, len(selected_pairs))) as pool:
                futures = [pool.submit(_compute_pair, pair) for pair in selected_pairs]
                done = 0
                for fut in as_completed(futures):
                    rec = fut.result()
                    records.append(rec)
                    fst_values.append(float(rec["fst"]))
                    done += 1
                    elapsed = perf_counter() - t_pairs
                    pair_progress.log(
                        pop=str(pop),
                        chrom=int(chrom),
                        done=int(done),
                        total=int(len(selected_pairs)),
                        rate_per_min=float((done / elapsed) * 60.0) if elapsed > 0 else float("nan"),
                        last_pair=f"S{rec['shard_a']}__S{rec['shard_b']}",
                        last_fst=float(rec["fst"]),
                    )
            records.sort(key=lambda x: (int(x["shard_a"]), int(x["shard_b"])))
        else:
            done = 0
            for pair in selected_pairs:
                rec = _compute_pair(pair)
                records.append(rec)
                fst_values.append(float(rec["fst"]))
                done += 1
                elapsed = perf_counter() - t_pairs
                pair_progress.log(
                    pop=str(pop),
                    chrom=int(chrom),
                    done=int(done),
                    total=int(len(selected_pairs)),
                    rate_per_min=float((done / elapsed) * 60.0) if elapsed > 0 else float("nan"),
                    last_pair=f"S{rec['shard_a']}__S{rec['shard_b']}",
                    last_fst=float(rec["fst"]),
                )

    summary = _summary_with_shards(
        fst_values=fst_values,
        total_samples=total_samples,
        shard_ids=shard_ids,
        shard_sizes=shard_sizes,
        candidate_pairs_total=candidate_pairs_total,
        selected_pairs=selected_pairs,
        max_pairs_per_population=max_pairs_per_population,
    )
    summary["status"] = "ok"
    return records, summary, int(candidate_pairs_total), int(len(selected_pairs))


def compute_synthetic_shard_round_robin_metrics(
    *,
    config: Any,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
) -> dict[str, Any]:
    del groups
    table_path = diagnostics_dir / f"synthetic_shard_round_robin_chr{chrom}.tsv"
    pairwise_mode = str(config.qc.synthetic_shard_round_robin_pairwise_mode)
    if not bool(config.qc.synthetic_shard_round_robin_enabled):
        return _empty_payload(chrom=chrom, table_path=table_path, pairwise_mode=pairwise_mode, status="disabled")

    shard_groups = _group_samples_by_population_and_shard(inputs.sample_manifest)
    if not shard_groups:
        return _empty_payload(chrom=chrom, table_path=table_path, pairwise_mode=pairwise_mode, status="no_populations")

    records: list[dict[str, float | int | str]] = []
    summary_by_population: dict[str, dict[str, Any]] = {}
    total_candidate_pairs = 0
    total_selected_pairs = 0

    for pop in sorted(shard_groups.keys()):
        population_hapstore = inputs.population_hapstores.get(pop)
        if inputs.combined_hapstore is None and population_hapstore is None:
            raise FileNotFoundError(f"Missing merged HapStore input for population {pop!r}")
        pop_records, pop_summary, pop_candidate_pairs, pop_selected_pairs = _compute_population_metric(
            config=config,
            chrom=int(chrom),
            pop=pop,
            shard_to_sample_ids=shard_groups[pop],
            diagnostics_dir=diagnostics_dir,
            numba_threads=int(numba_threads),
            combined_hapstore=inputs.combined_hapstore,
            population_hapstore=population_hapstore,
        )
        records.extend(pop_records)
        summary_by_population[pop] = pop_summary
        total_candidate_pairs += int(pop_candidate_pairs)
        total_selected_pairs += int(pop_selected_pairs)

    table_df = pd.DataFrame.from_records(records)
    if table_df.empty:
        table_df = pd.DataFrame(columns=["population", "chrom", "shard_a", "shard_b", "n_a", "n_b", "fst"])
    table_df = table_df.sort_values(["population", "shard_a", "shard_b"]).reset_index(drop=True)
    table_df.to_csv(table_path, sep="\t", index=False)

    status = "ok" if records else "insufficient_shards"
    return {
        "enabled": True,
        "chrom": int(chrom),
        "status": status,
        "gate_pass": None,
        "table_path": str(table_path),
        "population_summary": summary_by_population,
        "n_candidate_pairs": int(total_candidate_pairs),
        "n_selected_pairs": int(total_selected_pairs),
        "n_records": int(len(records)),
        "pairwise_mode": pairwise_mode,
    }
