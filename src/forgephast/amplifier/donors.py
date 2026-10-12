"""Resolve configured donor populations into reusable run-local artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .._internal.cache import canonical_hash
from .config import RunConfig
from .utils import apply_selector, ensure_dir, read_table


@dataclass
class Manifest:
    """Resolved donor pools and their persistent manifest artifacts.

    Attributes:
        manifest_path: Tabular donor membership and sampling-weight artifact.
        donor_counts_path: Per-configured-population donor counts, including
            zero-count ``msprime_tracts`` output populations.
        donor_files: Per-population, one-sample-ID-per-line donor lists.
        donors_by_population: Sorted donor IDs for each configured population.
        donor_weights_by_population: Normalized sample-selection weights for
            each donor pool. ``msprime_tracts`` output populations have empty
            donor pools; their source pools are represented separately.
    """

    manifest_path: Path
    donor_counts_path: Path
    donor_files: dict[str, Path]
    donors_by_population: dict[str, list[str]]
    donor_weights_by_population: dict[str, dict[str, float]]


def _input_fingerprint(path_value: str | None) -> dict[str, object] | None:
    if path_value is None:
        return None
    path = Path(path_value)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _manifest_identity(config: RunConfig) -> str:
    """Hash the donor-defining configuration and input-file fingerprints."""
    return canonical_hash(
        {
            "input": asdict(config.input),
            "populations": [asdict(population) for population in config.populations],
            "metadata": _input_fingerprint(config.input.metadata_path),
            "allowed_samples": _input_fingerprint(config.input.allowed_samples_path),
        }
    )


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(tmp_path, sep="\t", index=False)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write_text(value: str, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(value, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)

def build_manifest_from_config(config: RunConfig) -> Manifest:
    """Resolve donor pools from metadata and persist their run-local manifest.

    Mosaic populations select and weight donor IDs from the metadata table.
    ``selector_weights`` allocates mass across metadata population groups and
    assigns equal weight within each group. ``msprime_tracts`` output
    populations instead validate the named source pools from which ancestry
    tracts will later copy donor haplotypes.

    The returned artifacts are keyed by a donor-defining identity that includes
    the relevant configuration and input-file fingerprints. See
    :func:`load_manifest` for cache reuse and validation behavior.
    """
    run_root = ensure_dir(config.run_root)
    out_dir = ensure_dir(run_root / "manifests")
    identity_path = out_dir / "donor_manifest.forgephast.json"
    identity_path.unlink(missing_ok=True)

    metadata = read_table(config.input.metadata_path, sep=config.input.metadata_sep)
    if config.input.sample_id_column not in metadata.columns:
        raise ValueError(
            f"sample_id_column {config.input.sample_id_column!r} not found in metadata"
        )
    if config.input.population_column not in metadata.columns:
        raise ValueError(
            f"population_column {config.input.population_column!r} not found in metadata"
        )

    # Normalize before selection so persisted IDs and selector inputs agree.
    sample_col = config.input.sample_id_column
    pop_col = config.input.population_column
    metadata[sample_col] = metadata[sample_col].astype("string").fillna("").str.strip()
    metadata[pop_col] = metadata[pop_col].astype("string").fillna("").str.strip()
    if metadata[sample_col].eq("").any() or metadata[pop_col].eq("").any():
        raise ValueError("metadata contains blank sample or population identifiers")
    if metadata[sample_col].duplicated(keep=False).any():
        duplicate_ids = metadata.loc[
            metadata[sample_col].duplicated(keep=False), sample_col
        ].unique().tolist()
        raise ValueError(f"metadata contains duplicate sample IDs: {duplicate_ids[:10]}")

    allowed: set[str] | None = None
    if config.input.allowed_samples_path:
        allowed_series = pd.read_csv(
            config.input.allowed_samples_path,
            header=None,
            dtype=str,
        ).iloc[:, 0]
        allowed = set(allowed_series.astype("string").fillna("").str.strip())
        if "" in allowed:
            raise ValueError("allowed_samples_path contains blank sample IDs")

    donor_rows: list[dict[str, object]] = []
    donor_files: dict[str, Path] = {}
    donors_by_population: dict[str, list[str]] = {}
    donor_weights_by_population: dict[str, dict[str, float]] = {}

    # Mosaic populations own donor pools; msprime outputs reference those pools.
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
            # Convert group-level selector weights into normalized per-donor sampling weights
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
        _atomic_write_text("".join(f"{sid}\n" for sid in donors), donor_file)

        donor_files[pop.name] = donor_file
        donors_by_population[pop.name] = donors
        donor_weights_by_population[pop.name] = weights_by_donor
        donor_rows.extend(
            {
                "sample_id": sid,
                "population": pop.name,
                "source_value": source_value_by_donor.get(sid, ""),
                "sampling_weight": float(weights_by_donor.get(sid, 0.0)),
                "donor_only": bool(pop.donor_only),
            }
            for sid in donors
        )

    # Every msprime ancestry source must resolve to a nonempty donor pool.
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

    # Write each artifact atomically, then publish the manifest identity receipt.
    if donor_rows:
        manifest_df = (
            pd.DataFrame(donor_rows)
            .drop_duplicates()
            .sort_values(["population", "sample_id"])
        )
    else:
        manifest_df = pd.DataFrame(
            columns=["sample_id", "population", "source_value", "sampling_weight", "donor_only"]
        )
    manifest_path = out_dir / "donor_manifest.tsv"
    _atomic_to_csv(manifest_df, manifest_path)

    counts_df = pd.DataFrame(
        [
            {
                "population": pop.name,
                "n_donors": len(donors_by_population.get(pop.name, [])),
                "donor_only": bool(pop.donor_only),
            }
            for pop in config.populations
        ]
    ).sort_values("population")
    donor_counts_path = out_dir / "donor_counts.tsv"
    _atomic_to_csv(counts_df, donor_counts_path)
    _atomic_write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_identity": _manifest_identity(config),
                "n_manifest_rows": int(len(manifest_df)),
                "populations": sorted(pop.name for pop in config.populations),
            },
            sort_keys=True,
        )
        + "\n",
        identity_path,
    )

    return Manifest(
        manifest_path=manifest_path,
        donor_counts_path=donor_counts_path,
        donor_files=donor_files,
        donors_by_population=donors_by_population,
        donor_weights_by_population=donor_weights_by_population,
    )

def load_manifest(config: RunConfig) -> Manifest:
    """Load a matching donor manifest or rebuild it when its identity is stale.

    A reusable manifest requires the expected schema, all companion artifacts,
    and an identity matching the donor-defining configuration and input-file
    fingerprints. Loaded donor weights are normalized again after table parse
    to preserve their sampling-mass contract.
    """
    out_dir = config.run_root / "manifests"
    manifest_path = out_dir / "donor_manifest.tsv"
    donor_counts_path = out_dir / "donor_counts.tsv"
    identity_path = out_dir / "donor_manifest.forgephast.json"
    try:
        identity_payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        identity_payload = {}
    if (
        not manifest_path.exists()
        or not donor_counts_path.exists()
        or identity_payload.get("schema_version") != 1
        or identity_payload.get("manifest_identity") != _manifest_identity(config)
    ):
        return build_manifest_from_config(config)

    manifest_df = pd.read_csv(manifest_path, sep="\t", dtype=str)
    counts_df = pd.read_csv(donor_counts_path, sep="\t")
    required_manifest_columns = {"sample_id", "population", "sampling_weight", "donor_only"}
    if not required_manifest_columns.issubset(manifest_df.columns):
        raise ValueError(
            f"donor manifest is missing required columns: {sorted(required_manifest_columns - set(manifest_df.columns))}"
        )
    if not {"population", "n_donors"}.issubset(counts_df.columns):
        raise ValueError("donor counts file is missing population or n_donors")
    for column in ("sample_id", "population"):
        manifest_df[column] = manifest_df[column].astype("string").fillna("").str.strip()
    if manifest_df[["sample_id", "population"]].eq("").any().any():
        raise ValueError("donor manifest contains blank sample or population identifiers")
    if manifest_df.duplicated(subset=["sample_id", "population"], keep=False).any():
        raise ValueError("donor manifest contains duplicate sample/population rows")
    configured_populations = {population.name for population in config.populations}
    counted_populations = set(counts_df["population"].astype(str))
    if counted_populations != configured_populations:
        raise ValueError(
            "donor counts populations do not match config: "
            f"configured={sorted(configured_populations)}, observed={sorted(counted_populations)}"
        )
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
