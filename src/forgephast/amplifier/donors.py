"""Donor manifest construction from metadata selectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import RunConfig
from .utils import apply_selector, ensure_dir, read_table


@dataclass
class Manifest:
    manifest_path: Path
    donor_counts_path: Path
    donor_files: dict[str, Path]
    donors_by_population: dict[str, list[str]]
    donor_weights_by_population: dict[str, dict[str, float]]



def build_manifest_from_config(config: RunConfig) -> Manifest:
    run_root = ensure_dir(config.run_root)
    out_dir = ensure_dir(run_root / "manifests")

    metadata = read_table(config.input.metadata_path, sep=config.input.metadata_sep)
    if config.input.sample_id_column not in metadata.columns:
        raise ValueError(
            f"sample_id_column {config.input.sample_id_column!r} not found in metadata"
        )
    if config.input.population_column not in metadata.columns:
        raise ValueError(
            f"population_column {config.input.population_column!r} not found in metadata"
        )

    sample_col = config.input.sample_id_column
    pop_col = config.input.population_column
    metadata[sample_col] = metadata[sample_col].astype(str).str.strip()
    metadata[pop_col] = metadata[pop_col].astype(str).str.strip()

    allowed: set[str] | None = None
    if config.input.allowed_samples_path:
        allowed_series = pd.read_csv(
            config.input.allowed_samples_path,
            header=None,
            dtype=str,
        ).iloc[:, 0]
        allowed = set(allowed_series.astype(str).str.strip())

    donor_rows: list[dict[str, object]] = []
    donor_files: dict[str, Path] = {}
    donors_by_population: dict[str, list[str]] = {}
    donor_weights_by_population: dict[str, dict[str, float]] = {}

    for pop in config.populations:
        donors: list[str]
        if pop.engine == "msprime_tracts":
            donors = []
        else:
            mask = apply_selector(metadata[pop_col], pop.selector)
            selected_meta = metadata.loc[mask, [sample_col, pop_col]].copy()
            selected_meta[sample_col] = selected_meta[sample_col].astype(str)
            selected_meta[pop_col] = selected_meta[pop_col].astype(str)
            selected = selected_meta[sample_col]
            if allowed is not None:
                selected_meta = selected_meta[selected_meta[sample_col].isin(allowed)].copy()
                selected = selected_meta[sample_col]
            donors = sorted(set(selected.tolist()))
            if not donors:
                raise ValueError(
                    f"selector for population {pop.name!r} returned zero donors"
                )
            selected_meta = selected_meta[selected_meta[sample_col].isin(donors)].drop_duplicates(
                subset=[sample_col]
            )
            source_value_by_donor = {
                str(row[sample_col]): str(row[pop_col])
                for _, row in selected_meta.iterrows()
            }
            weights_by_donor: dict[str, float] = {}
            if pop.selector_weights is None:
                uniform_weight = 1.0 / float(len(donors))
                weights_by_donor = {sid: uniform_weight for sid in donors}
            else:
                group_to_donors: dict[str, list[str]] = {}
                for _, row in selected_meta.iterrows():
                    group_to_donors.setdefault(str(row[pop_col]), []).append(str(row[sample_col]))
                missing_groups = sorted(set(group_to_donors).difference(pop.selector_weights))
                extra_groups = sorted(set(pop.selector_weights).difference(group_to_donors))
                if missing_groups or extra_groups:
                    raise ValueError(
                        f"population {pop.name!r}: selector_weights mismatch; "
                        f"missing={missing_groups} extra={extra_groups}"
                    )
                for group, donor_group in group_to_donors.items():
                    group_weight = float(pop.selector_weights[group])
                    per_donor_weight = group_weight / float(len(donor_group))
                    for sid in donor_group:
                        weights_by_donor[str(sid)] = per_donor_weight
                total_weight = sum(weights_by_donor.values())
                if total_weight <= 0.0:
                    raise ValueError(f"population {pop.name!r}: selector_weights produced zero donor weight")
                weights_by_donor = {
                    str(sid): float(weight) / float(total_weight)
                    for sid, weight in weights_by_donor.items()
                }
        if pop.engine == "msprime_tracts":
            weights_by_donor = {}
            source_value_by_donor = {}

        donor_file = out_dir / f"donors_{pop.name}.txt"
        with open(donor_file, "w", encoding="utf-8") as handle:
            for sid in donors:
                handle.write(f"{sid}\n")

        donor_files[pop.name] = donor_file
        donors_by_population[pop.name] = donors
        donor_weights_by_population[pop.name] = weights_by_donor
        donor_rows.extend(
            {
                "sample_id": sid,
                "population": pop.name,
                "source_value": source_value_by_donor.get(sid, ""),
                "sampling_weight": float(weights_by_donor.get(sid, 0.0)),
            }
            for sid in donors
        )

    for pop in config.populations:
        if pop.engine != "msprime_tracts":
            continue
        assert pop.msprime is not None
        for source in pop.msprime.sources:
            source_name = str(source.population)
            donors = donors_by_population.get(source_name, [])
            if len(donors) <= 0:
                raise ValueError(
                    f"population {pop.name!r}: msprime source population {source_name!r} has zero donors"
                )

    if donor_rows:
        manifest_df = (
            pd.DataFrame(donor_rows)
            .drop_duplicates()
            .sort_values(["population", "sample_id"])
        )
    else:
        manifest_df = pd.DataFrame(columns=["sample_id", "population", "source_value", "sampling_weight"])
    manifest_path = out_dir / "donor_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    counts_df = pd.DataFrame(
        [
            {"population": pop.name, "n_donors": len(donors_by_population.get(pop.name, []))}
            for pop in config.populations
        ]
    ).sort_values("population")
    donor_counts_path = out_dir / "donor_counts.tsv"
    counts_df.to_csv(donor_counts_path, sep="\t", index=False)

    return Manifest(
        manifest_path=manifest_path,
        donor_counts_path=donor_counts_path,
        donor_files=donor_files,
        donors_by_population=donors_by_population,
        donor_weights_by_population=donor_weights_by_population,
    )



def load_manifest(config: RunConfig) -> Manifest:
    out_dir = config.run_root / "manifests"
    manifest_path = out_dir / "donor_manifest.tsv"
    donor_counts_path = out_dir / "donor_counts.tsv"
    if not manifest_path.exists() or not donor_counts_path.exists():
        return build_manifest_from_config(config)

    manifest_df = pd.read_csv(manifest_path, sep="\t", dtype=str)
    counts_df = pd.read_csv(donor_counts_path, sep="\t")
    donors_by_population: dict[str, list[str]] = {}
    donor_files: dict[str, Path] = {}
    donor_weights_by_population: dict[str, dict[str, float]] = {}
    for pop, group in manifest_df.groupby("population"):
        donors = sorted(group["sample_id"].astype(str).unique().tolist())
        donors_by_population[str(pop)] = donors
        donor_files[str(pop)] = out_dir / f"donors_{pop}.txt"
        if "sampling_weight" in group.columns:
            weights = {
                str(row["sample_id"]): float(row["sampling_weight"])
                for _, row in group.iterrows()
            }
            total_weight = sum(weights.values())
            if total_weight > 0.0:
                weights = {
                    sid: float(weight) / float(total_weight)
                    for sid, weight in weights.items()
                }
        else:
            uniform_weight = 1.0 / float(len(donors)) if donors else 0.0
            weights = {sid: uniform_weight for sid in donors}
        donor_weights_by_population[str(pop)] = weights
    for pop in counts_df["population"].astype(str).tolist():
        if pop not in donors_by_population:
            donors_by_population[pop] = []
            donor_weights_by_population[pop] = {}
        donor_files.setdefault(pop, out_dir / f"donors_{pop}.txt")

    return Manifest(
        manifest_path=manifest_path,
        donor_counts_path=donor_counts_path,
        donor_files=donor_files,
        donors_by_population=donors_by_population,
        donor_weights_by_population=donor_weights_by_population,
    )
