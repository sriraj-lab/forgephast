"""Donor round-robin orchestration support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .types import DonorWithinPopRoundRobinReport


def run_donor_within_pop_round_robin_impl(
    *,
    config: Any,
    chrom: int,
    fold_fraction: float,
    seed: int,
    workers: int,
    populations: list[str] | None,
    round_robin_folds_from_fraction_fn: Callable[[float], int],
    load_manifest_fn: Callable[[Any], Any],
    resolve_requested_populations_fn: Callable[[list[str] | None, list[str]], list[str]],
    ensure_dir_fn: Callable[[Path], Path],
    round_robin_for_population_fn: Callable[..., tuple[list[dict[str, float | int | str]], dict[str, float | int]]],
    write_json_fn: Callable[[Path, dict[str, Any]], None],
) -> DonorWithinPopRoundRobinReport:
    if workers <= 0:
        raise ValueError("workers must be > 0")
    n_folds = round_robin_folds_from_fraction_fn(float(fold_fraction))

    manifest = load_manifest_fn(config)
    donors_by_population = manifest.donors_by_population
    pop_names = resolve_requested_populations_fn(populations, sorted(donors_by_population.keys()))
    diagnostics_dir = ensure_dir_fn(config.run_root / "diagnostics")

    records: list[dict[str, float | int | str]] = []
    summaries: dict[str, dict[str, float | int]] = {}
    for pop_index, pop in enumerate(pop_names):
        donors = sorted(donors_by_population[pop])
        pop_seed = int(seed + (pop_index + 1) * 1009)
        pop_records, pop_summary = round_robin_for_population_fn(
            pop=str(pop),
            chrom=int(chrom),
            sample_ids=donors,
            n_folds=int(n_folds),
            seed=pop_seed,
            numba_threads=int(workers),
        )
        records.extend(pop_records)
        summaries[str(pop)] = pop_summary

    table_path = diagnostics_dir / f"donor_within_pop_round_robin_chr{chrom}.tsv"
    diagnostics_path = diagnostics_dir / f"donor_within_pop_round_robin_chr{chrom}.json"
    table_df = pd.DataFrame.from_records(records)
    if table_df.empty:
        table_df = pd.DataFrame(columns=["population", "fold_a", "fold_b", "n_a", "n_b", "fst"])
    table_df = table_df.sort_values(["population", "fold_a", "fold_b"]).reset_index(drop=True)
    table_df.to_csv(table_path, sep="\t", index=False)

    payload = {
        "chrom": int(chrom),
        "n_folds": int(n_folds),
        "fold_fraction": float(fold_fraction),
        "seed": int(seed),
        "workers": int(workers),
        "n_records": int(len(records)),
        "table_path": str(table_path),
        "summary_by_population": summaries,
    }
    write_json_fn(diagnostics_path, payload)

    return DonorWithinPopRoundRobinReport(
        chrom=int(chrom),
        n_folds=int(n_folds),
        fold_fraction=float(fold_fraction),
        seed=int(seed),
        workers=int(workers),
        diagnostics_path=diagnostics_path,
        table_path=table_path,
        summary_by_population=summaries,
        n_records=len(records),
    )
