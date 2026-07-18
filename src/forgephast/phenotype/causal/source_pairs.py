"""Build causal/source-pair candidate tables directly from HapStore genotypes."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from rajlab_utils.popgen import normalize_chrom

from ..hapstore_contract import (
    HAPSTORE_MISSING_SENTINEL,
    validate_hapstore_missing_sentinel,
)

logger = logging.getLogger(__name__)

StoreOpener = Callable[[Path], object]


@dataclass
class SourcePairHapStoreConfig:
    """HapStore layout for source-pair MAF extraction."""

    layout: str = "per_chr"
    root_dir: str = ""
    file_pattern: str = "chr{chrom}.hapstore.zarr"
    store_path: str | None = None
    missing_sentinel: int = 255

    def validate(self) -> None:
        if self.layout not in {"per_chr", "single_store", "per_pop_per_chr"}:
            raise ValueError(
                "hapstore.layout must be one of {'per_chr', 'single_store', 'per_pop_per_chr'}"
            )
        if self.layout == "single_store":
            if not self.store_path:
                raise ValueError("hapstore.store_path is required when layout=single_store")
        else:
            if not self.root_dir:
                raise ValueError("hapstore.root_dir is required")
            if "{chrom}" not in self.file_pattern:
                raise ValueError("hapstore.file_pattern must include {chrom}")
            if self.layout == "per_pop_per_chr" and "{population}" not in self.file_pattern:
                raise ValueError(
                    "hapstore.file_pattern must include {population} for layout=per_pop_per_chr"
                )
        if int(self.missing_sentinel) != HAPSTORE_MISSING_SENTINEL:
            raise ValueError(
                "hapstore.missing_sentinel must be 255 to match the HapStore format contract"
            )


@dataclass
class SourcePairSampleConfig:
    """Sample metadata used to select pooled and group-specific MAF cohorts."""

    file: str
    id_column: str = "IID"
    group_column: str = "population"
    store_population_column: str | None = None
    delimiter: str = "\t"

    def validate(self, *, hapstore_layout: str) -> None:
        if not self.file:
            raise ValueError("samples.file cannot be empty")
        if not self.id_column:
            raise ValueError("samples.id_column cannot be empty")
        if not self.group_column:
            raise ValueError("samples.group_column cannot be empty")
        if hapstore_layout == "per_pop_per_chr" and not (
            self.store_population_column or self.group_column
        ):
            raise ValueError(
                "samples.store_population_column or samples.group_column is required "
                "for hapstore.layout=per_pop_per_chr"
            )

    def resolved_store_population_column(self) -> str:
        return str(self.store_population_column or self.group_column)


@dataclass
class SourcePairGroupConfig:
    """One MAF output group selected from sample metadata."""

    name: str
    column: str | None = None
    value: str | None = None
    values: list[str] | None = None
    filters: dict[str, str | list[str]] = field(default_factory=dict)
    maf_column: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | str) -> "SourcePairGroupConfig":
        if isinstance(payload, str):
            return cls(name=str(payload))
        data = dict(payload)
        filters_payload = data.get("filters", {})
        if filters_payload is None:
            filters: dict[str, str | list[str]] = {}
        elif isinstance(filters_payload, Mapping):
            filters = {str(key): _string_or_string_list(value) for key, value in filters_payload.items()}
        else:
            raise ValueError("groups[].filters must be a mapping")
        values_payload = data.get("values")
        values = None if values_payload is None else [str(value) for value in _as_list(values_payload)]
        return cls(
            name=str(data["name"]),
            column=None if data.get("column") is None else str(data.get("column")),
            value=None if data.get("value") is None else str(data.get("value")),
            values=values,
            filters=filters,
            maf_column=None if data.get("maf_column") is None else str(data.get("maf_column")),
        )

    def resolved_column(self, default_group_column: str) -> str:
        return str(self.column or default_group_column)

    def resolved_values(self) -> list[str]:
        if self.values is not None:
            if not self.values:
                raise ValueError(f"group {self.name}: values cannot be empty")
            return [str(value) for value in self.values]
        if self.value is not None:
            return [str(self.value)]
        return [str(self.name)]

    def resolved_maf_column(self) -> str:
        return str(self.maf_column or f"maf_{slugify_label(self.name)}")

    def validate(self, *, default_group_column: str) -> None:
        if not self.name:
            raise ValueError("groups[].name cannot be empty")
        if not self.resolved_column(default_group_column):
            raise ValueError(f"group {self.name}: column cannot be empty")
        if not self.resolved_maf_column():
            raise ValueError(f"group {self.name}: maf_column cannot be empty")
        self.resolved_values()


@dataclass
class SourcePairOutputConfig:
    """Output paths for source-pair table and optional QC summary."""

    source_pairs_file: str
    qc_file: str | None = None
    log_file: str | None = None

    def validate(self) -> None:
        if not self.source_pairs_file:
            raise ValueError("output.source_pairs_file cannot be empty")


@dataclass
class SourcePairBuildConfig:
    """Top-level source-pair build configuration."""

    hapstore: SourcePairHapStoreConfig
    samples: SourcePairSampleConfig
    output: SourcePairOutputConfig
    chromosomes: list[int]
    groups: list[SourcePairGroupConfig] = field(default_factory=list)
    pooled_maf_min: float = 0.0
    variant_block_size: int = 1024
    variant_id_column: str = "variant_id"
    chrom_column: str = "chrom"
    position_column: str = "pos"
    source_variant_id_column: str = "source_variant_id"
    pooled_maf_column: str = "maf_pooled"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SourcePairBuildConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("source-pair config YAML must contain a mapping")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourcePairBuildConfig":
        data = dict(payload)
        allowed_top_level = {
            "hapstore",
            "samples",
            "output",
            "chromosomes",
            "groups",
            "source_pairs",
            "source_pairs_file",
            "qc_file",
            "log_file",
            "pooled_maf_min",
            "variant_block_size",
            "variant_id_column",
            "chrom_column",
            "position_column",
            "source_variant_id_column",
            "pooled_maf_column",
        }
        unknown_top_level = sorted(set(data).difference(allowed_top_level))
        if unknown_top_level:
            raise ValueError(
                f"Unknown source-pair config key(s): {unknown_top_level}"
            )
        source_pair_options = dict(data.get("source_pairs", {}) or {})
        allowed_source_pair_options = {
            "pooled_maf_min",
            "variant_block_size",
            "variant_id_column",
            "chrom_column",
            "position_column",
            "source_variant_id_column",
            "pooled_maf_column",
        }
        unknown_source_pair_options = sorted(
            set(source_pair_options).difference(allowed_source_pair_options)
        )
        if unknown_source_pair_options:
            raise ValueError(
                "Unknown source_pairs config key(s): "
                f"{unknown_source_pair_options}"
            )
        output_payload = data.get("output", {})
        if isinstance(output_payload, Mapping):
            output_data = dict(output_payload)
        else:
            raise ValueError("output must be a mapping")
        if "source_pairs_file" not in output_data and "source_pairs_file" in data:
            output_data["source_pairs_file"] = data["source_pairs_file"]
        if "qc_file" not in output_data and "qc_file" in data:
            output_data["qc_file"] = data["qc_file"]
        if "log_file" not in output_data and "log_file" in data:
            output_data["log_file"] = data["log_file"]

        groups_payload = data.get("groups")
        groups = (
            []
            if groups_payload is None
            else [SourcePairGroupConfig.from_payload(item) for item in _as_list(groups_payload)]
        )

        config = cls(
            hapstore=SourcePairHapStoreConfig(**dict(data.get("hapstore", {}) or {})),
            samples=SourcePairSampleConfig(**dict(data["samples"])),
            output=SourcePairOutputConfig(**output_data),
            chromosomes=[int(chrom) for chrom in _as_list(data["chromosomes"])],
            groups=groups,
            pooled_maf_min=float(source_pair_options.get("pooled_maf_min", data.get("pooled_maf_min", 0.0))),
            variant_block_size=int(
                source_pair_options.get("variant_block_size", data.get("variant_block_size", 1024))
            ),
            variant_id_column=str(
                source_pair_options.get("variant_id_column", data.get("variant_id_column", "variant_id"))
            ),
            chrom_column=str(source_pair_options.get("chrom_column", data.get("chrom_column", "chrom"))),
            position_column=str(source_pair_options.get("position_column", data.get("position_column", "pos"))),
            source_variant_id_column=str(
                source_pair_options.get(
                    "source_variant_id_column",
                    data.get("source_variant_id_column", "source_variant_id"),
                )
            ),
            pooled_maf_column=str(
                source_pair_options.get("pooled_maf_column", data.get("pooled_maf_column", "maf_pooled"))
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.hapstore.validate()
        self.samples.validate(hapstore_layout=self.hapstore.layout)
        self.output.validate()
        if not self.chromosomes:
            raise ValueError("chromosomes cannot be empty")
        if any(int(chrom) <= 0 for chrom in self.chromosomes):
            raise ValueError("chromosomes must be positive integers")
        if not 0.0 <= float(self.pooled_maf_min) <= 0.5:
            raise ValueError("pooled_maf_min must be in [0.0, 0.5]")
        if int(self.variant_block_size) <= 0:
            raise ValueError("variant_block_size must be > 0")
        for column_name in (
            self.variant_id_column,
            self.chrom_column,
            self.position_column,
            self.source_variant_id_column,
            self.pooled_maf_column,
        ):
            if not column_name:
                raise ValueError("source-pair output column names cannot be empty")
        for group in self.groups:
            group.validate(default_group_column=self.samples.group_column)
        maf_columns = [group.resolved_maf_column() for group in self.groups]
        if len(set(maf_columns)) != len(maf_columns):
            raise ValueError("group maf_column values must be unique")


@dataclass
class SourcePairBuildResult:
    """Paths and counts produced by one source-pair build."""

    source_pairs_file: Path
    qc_file: Path | None
    n_variants_total: int
    n_variants_written: int
    chromosomes: list[int]


def slugify_label(label: str) -> str:
    """Return a stable lower-case MAF-column-safe label."""
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(label).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "group"


def build_source_pairs_from_hapstore(
    config: SourcePairBuildConfig,
    *,
    opener: StoreOpener | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a source-pair table and QC summary from HapStore genotype blocks."""
    config.validate()
    sample_df = load_sample_metadata(config.samples)
    groups = resolve_groups(config, sample_df)
    frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []

    for chrom in config.chromosomes:
        frame, qc = build_source_pairs_frame_for_chrom(
            config=config,
            sample_df=sample_df,
            groups=groups,
            chrom=int(chrom),
            opener=opener,
        )
        frames.append(frame)
        qc_rows.append(qc)

    if frames:
        source_pairs = pd.concat(frames, axis=0, ignore_index=True)
    else:
        source_pairs = pd.DataFrame()
    total_rows = {
        "scope": "total",
        "chrom": "",
        "n_variants_total": int(sum(int(row["n_variants_total"]) for row in qc_rows)),
        "n_variants_written": int(sum(int(row["n_variants_written"]) for row in qc_rows)),
        "pooled_maf_min": float(config.pooled_maf_min),
        "variant_block_size": int(config.variant_block_size),
        "n_samples_pooled": int(sample_df.shape[0]),
        "maf_columns_json": json.dumps(
            [config.pooled_maf_column, *[group.resolved_maf_column() for group in groups]]
        ),
        "groups_json": json.dumps(
            {group.name: group.resolved_maf_column() for group in groups},
            sort_keys=True,
        ),
    }
    qc_frame = pd.DataFrame.from_records([*qc_rows, total_rows])
    return source_pairs, qc_frame


def build_source_pairs_frame_for_chrom(
    *,
    config: SourcePairBuildConfig,
    sample_df: pd.DataFrame,
    groups: Sequence[SourcePairGroupConfig],
    chrom: int,
    opener: StoreOpener | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one chromosome's source-pair rows from one or more HapStore files."""
    chrom_key = normalize_chrom(int(chrom), prefix=True)
    store_refs = _store_refs_for_chrom(config, sample_df, chrom=int(chrom))
    reference: dict[str, Any] | None = None
    pooled_alt: np.ndarray | None = None
    pooled_an: np.ndarray | None = None
    group_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    seen_pooled_ids: set[str] = set()
    group_sample_counts = {group.name: 0 for group in groups}
    expected_pooled_ids = set(sample_df[config.samples.id_column].astype(str).tolist())

    for store_population, path in store_refs:
        store = _open_store(path, opener=opener)
        try:
            info = _read_variant_reference(store, chrom_key=chrom_key, chrom=int(chrom), config=config)
            if reference is None:
                reference = info
                n_variants = int(info["positions"].shape[0])
                pooled_alt = np.zeros(n_variants, dtype=np.float64)
                pooled_an = np.zeros(n_variants, dtype=np.float64)
                group_counts = {
                    group.name: (
                        np.zeros(n_variants, dtype=np.float64),
                        np.zeros(n_variants, dtype=np.float64),
                    )
                    for group in groups
                }
            else:
                _validate_matching_variants(reference, info, path=path)
                n_variants = int(reference["positions"].shape[0])

            store_sample_ids = [str(iid) for iid in store.sample_ids(chrom_key)]
            masks = build_sample_masks_for_store(
                store_sample_ids=store_sample_ids,
                sample_df=sample_df,
                groups=groups,
                sample_cfg=config.samples,
            )
            seen_pooled_ids.update(iid for iid, keep in zip(store_sample_ids, masks["pooled"], strict=True) if keep)
            for group in groups:
                group_sample_counts[group.name] += int(np.count_nonzero(masks[group.name]))

            logger.info(
                "SOURCE_PAIRS_CHROM_STORE_START chrom=%s store_population=%s path=%s n_variants=%d n_samples=%d pooled_samples=%d",
                chrom_key,
                store_population,
                path,
                n_variants,
                int(len(store_sample_ids)),
                int(np.count_nonzero(masks["pooled"])),
            )
            assert pooled_alt is not None
            assert pooled_an is not None
            _accumulate_store_counts(
                store=store,
                chrom_key=chrom_key,
                n_variants=n_variants,
                variant_block_size=int(config.variant_block_size),
                missing_sentinel=int(config.hapstore.missing_sentinel),
                masks=masks,
                pooled_alt=pooled_alt,
                pooled_an=pooled_an,
                group_counts=group_counts,
            )
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()

    if reference is None or pooled_alt is None or pooled_an is None:
        raise ValueError(f"No HapStore inputs found for chromosome {chrom}")

    missing = sorted(expected_pooled_ids - seen_pooled_ids)
    if missing:
        raise ValueError(
            f"HapStore inputs for chromosome {chrom} are missing {len(missing)} selected sample(s); "
            f"first_missing={missing[0]}"
        )
    empty_groups = [name for name, count in group_sample_counts.items() if int(count) == 0]
    if empty_groups:
        raise ValueError(f"No HapStore samples selected for MAF group(s): {empty_groups}")

    frame = _counts_to_frame(
        config=config,
        groups=groups,
        chrom=int(chrom),
        reference=reference,
        pooled_alt=pooled_alt,
        pooled_an=pooled_an,
        group_counts=group_counts,
    )
    n_before = int(frame.shape[0])
    frame = frame.loc[
        pd.to_numeric(frame[config.pooled_maf_column], errors="raise") >= float(config.pooled_maf_min)
    ].reset_index(drop=True)
    qc = {
        "scope": "chrom",
        "chrom": str(normalize_chrom(int(chrom), prefix=False)),
        "n_variants_total": n_before,
        "n_variants_written": int(frame.shape[0]),
        "pooled_maf_min": float(config.pooled_maf_min),
        "variant_block_size": int(config.variant_block_size),
        "n_samples_pooled": int(len(expected_pooled_ids)),
        "maf_columns_json": json.dumps(
            [config.pooled_maf_column, *[group.resolved_maf_column() for group in groups]]
        ),
        "groups_json": json.dumps(
            {group.name: group.resolved_maf_column() for group in groups},
            sort_keys=True,
        ),
    }
    logger.info(
        "SOURCE_PAIRS_CHROM_DONE chrom=%s n_variants_total=%d n_variants_written=%d",
        chrom_key,
        n_before,
        int(frame.shape[0]),
    )
    return frame, qc


def write_source_pairs_from_hapstore(
    config: SourcePairBuildConfig,
    *,
    opener: StoreOpener | None = None,
) -> SourcePairBuildResult:
    """Build and write source-pair table plus optional QC summary."""
    source_pairs, qc = build_source_pairs_from_hapstore(config, opener=opener)
    out_path = Path(config.output.source_pairs_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_pairs.to_csv(out_path, sep="\t", index=False)

    qc_path = None
    if config.output.qc_file:
        qc_path = Path(config.output.qc_file)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc.to_csv(qc_path, sep="\t", index=False)

    result = SourcePairBuildResult(
        source_pairs_file=out_path,
        qc_file=qc_path,
        n_variants_total=int(qc.loc[qc["scope"] == "total", "n_variants_total"].iloc[0]),
        n_variants_written=int(qc.loc[qc["scope"] == "total", "n_variants_written"].iloc[0]),
        chromosomes=[int(chrom) for chrom in config.chromosomes],
    )
    logger.info(
        "SOURCE_PAIRS_COMPLETED source_pairs=%s qc=%s n_variants_total=%d n_variants_written=%d",
        result.source_pairs_file,
        result.qc_file or "",
        result.n_variants_total,
        result.n_variants_written,
    )
    return result


def load_sample_metadata(sample_cfg: SourcePairSampleConfig) -> pd.DataFrame:
    """Load and validate source-pair sample metadata."""
    df = pd.read_csv(sample_cfg.file, sep=sample_cfg.delimiter)
    required = {sample_cfg.id_column, sample_cfg.group_column}
    if sample_cfg.store_population_column:
        required.add(sample_cfg.store_population_column)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"sample metadata missing required columns: {missing}")
    out = df.copy()
    out[sample_cfg.id_column] = out[sample_cfg.id_column].astype(str)
    if out[sample_cfg.id_column].isna().any() or (out[sample_cfg.id_column].str.len() == 0).any():
        raise ValueError(f"sample metadata column {sample_cfg.id_column!r} contains blank IDs")
    if out[sample_cfg.id_column].duplicated().any():
        duplicates = out.loc[out[sample_cfg.id_column].duplicated(), sample_cfg.id_column].head(5).tolist()
        raise ValueError(f"sample metadata contains duplicate IDs; first duplicates={duplicates}")
    if out.empty:
        raise ValueError("sample metadata contains zero rows")
    return out


def resolve_groups(
    config: SourcePairBuildConfig,
    sample_df: pd.DataFrame,
) -> list[SourcePairGroupConfig]:
    """Resolve explicit or sample-column-derived MAF groups."""
    groups = list(config.groups)
    if not groups:
        if config.samples.group_column not in sample_df.columns:
            raise ValueError(
                "groups are not configured and samples.group_column is missing from sample metadata"
            )
        groups = [
            SourcePairGroupConfig(name=str(value), column=config.samples.group_column, value=str(value))
            for value in dict.fromkeys(sample_df[config.samples.group_column].astype(str).tolist())
        ]
    for group in groups:
        group.validate(default_group_column=config.samples.group_column)
        required = {group.resolved_column(config.samples.group_column), *group.filters.keys()}
        missing = sorted(required - set(sample_df.columns))
        if missing:
            raise ValueError(f"group {group.name} references missing sample metadata columns: {missing}")
        mask = group_mask(sample_df, group=group, sample_cfg=config.samples)
        if not bool(mask.any()):
            raise ValueError(f"group {group.name} selects zero rows from sample metadata")
    return groups


def group_mask(
    sample_df: pd.DataFrame,
    *,
    group: SourcePairGroupConfig,
    sample_cfg: SourcePairSampleConfig,
) -> pd.Series:
    """Return a boolean sample metadata mask for one MAF group."""
    values = {str(value) for value in group.resolved_values()}
    column = group.resolved_column(sample_cfg.group_column)
    mask = sample_df[column].astype(str).isin(values)
    for filter_column, filter_values in group.filters.items():
        allowed = {str(value) for value in _as_list(filter_values)}
        mask &= sample_df[str(filter_column)].astype(str).isin(allowed)
    return mask


def build_sample_masks_for_store(
    *,
    store_sample_ids: Sequence[str],
    sample_df: pd.DataFrame,
    groups: Sequence[SourcePairGroupConfig],
    sample_cfg: SourcePairSampleConfig,
) -> dict[str, np.ndarray]:
    """Build pooled and group masks aligned to one store's sample order."""
    pooled_ids = set(sample_df[sample_cfg.id_column].astype(str).tolist())
    masks: dict[str, np.ndarray] = {
        "pooled": np.asarray([str(iid) in pooled_ids for iid in store_sample_ids], dtype=bool)
    }
    for group in groups:
        selected_ids = set(
            sample_df.loc[group_mask(sample_df, group=group, sample_cfg=sample_cfg), sample_cfg.id_column]
            .astype(str)
            .tolist()
        )
        masks[group.name] = np.asarray([str(iid) in selected_ids for iid in store_sample_ids], dtype=bool)
    return masks


def _accumulate_store_counts(
    *,
    store: object,
    chrom_key: str,
    n_variants: int,
    variant_block_size: int,
    missing_sentinel: int,
    masks: Mapping[str, np.ndarray],
    pooled_alt: np.ndarray,
    pooled_an: np.ndarray,
    group_counts: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for start in range(0, int(n_variants), int(variant_block_size)):
        end = min(int(n_variants), start + int(variant_block_size))
        genotype_block = np.asarray(
            store.genotypes(chrom_key, variant_slice=(start, end)),
            dtype=np.uint8,
        )
        valid = genotype_block != np.uint8(missing_sentinel)
        pooled_block = genotype_block[masks["pooled"], :]
        pooled_valid = valid[masks["pooled"], :]
        pooled_alt[start:end] += np.where(pooled_valid, pooled_block, 0).sum(axis=0, dtype=np.float64)
        pooled_an[start:end] += 2.0 * pooled_valid.sum(axis=0, dtype=np.float64)

        for group_name, (alt_counts, allele_numbers) in group_counts.items():
            group_block = genotype_block[masks[group_name], :]
            group_valid = valid[masks[group_name], :]
            alt_counts[start:end] += np.where(group_valid, group_block, 0).sum(axis=0, dtype=np.float64)
            allele_numbers[start:end] += 2.0 * group_valid.sum(axis=0, dtype=np.float64)


def _counts_to_frame(
    *,
    config: SourcePairBuildConfig,
    groups: Sequence[SourcePairGroupConfig],
    chrom: int,
    reference: Mapping[str, Any],
    pooled_alt: np.ndarray,
    pooled_an: np.ndarray,
    group_counts: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    positions = np.asarray(reference["positions"], dtype=np.int64)
    source_variant_ids = [str(value) for value in reference["source_variant_ids"]]
    variant_ids = [f"{normalize_chrom(chrom, prefix=True)}:{int(pos)}" for pos in positions.tolist()]
    pooled_freq = np.divide(pooled_alt, pooled_an, out=np.zeros_like(pooled_alt), where=pooled_an > 0.0)
    pooled_maf = np.minimum(pooled_freq, 1.0 - pooled_freq)
    frame = pd.DataFrame(
        {
            config.variant_id_column: variant_ids,
            config.chrom_column: normalize_chrom(chrom, prefix=False),
            config.position_column: positions,
            config.source_variant_id_column: source_variant_ids,
            "ref": [str(value) for value in reference["ref"]],
            "alt": [str(value) for value in reference["alt"]],
            config.pooled_maf_column: pooled_maf,
        }
    )
    for group in groups:
        alt_counts, allele_numbers = group_counts[group.name]
        freq = np.divide(alt_counts, allele_numbers, out=np.zeros_like(alt_counts), where=allele_numbers > 0.0)
        frame[group.resolved_maf_column()] = np.minimum(freq, 1.0 - freq)
    return frame


def _read_variant_reference(
    store: object,
    *,
    chrom_key: str,
    chrom: int,
    config: SourcePairBuildConfig,
) -> dict[str, Any]:
    has_chromosome = getattr(store, "has_chromosome", None)
    if callable(has_chromosome) and not bool(has_chromosome(chrom_key)):
        raise ValueError(f"HapStore does not contain chromosome {chrom_key}")
    validate_hapstore_missing_sentinel(
        store,
        chrom_key,
        context=f"source-pair HapStore chrom={chrom_key}",
    )
    info = store.variant_info(chrom_key)
    positions = np.asarray(info["pos"], dtype=np.int64)
    if positions.size == 0:
        raise ValueError(f"HapStore chromosome {chrom_key} contains zero variants")
    if positions.size > 1 and not bool(np.all(positions[1:] > positions[:-1])):
        raise ValueError(
            f"HapStore chromosome {chrom_key} positions must be strictly increasing; "
            "duplicate positions are incompatible with chr:pos causal IDs"
        )
    ref = [str(value) for value in info["ref"]]
    alt = [str(value) for value in info["alt"]]
    if len(ref) != int(positions.shape[0]) or len(alt) != int(positions.shape[0]):
        raise ValueError(f"HapStore chromosome {chrom_key} REF/ALT arrays are not length-aligned")
    vid = info.get("vid")
    if vid is None or len(vid) != int(positions.shape[0]):
        source_variant_ids = [f"{normalize_chrom(chrom, prefix=True)}:{int(pos)}" for pos in positions.tolist()]
    else:
        source_variant_ids = [str(value) for value in vid]
    return {
        "positions": positions,
        "source_variant_ids": source_variant_ids,
        "ref": ref,
        "alt": alt,
    }


def _validate_matching_variants(
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    if not np.array_equal(np.asarray(reference["positions"]), np.asarray(observed["positions"])):
        raise ValueError(f"HapStore variants do not match reference positions: {path}")
    if list(reference["source_variant_ids"]) != list(observed["source_variant_ids"]):
        raise ValueError(f"HapStore variants do not match reference source IDs: {path}")
    if list(reference["ref"]) != list(observed["ref"]):
        raise ValueError(f"HapStore variants do not match reference REF alleles: {path}")
    if list(reference["alt"]) != list(observed["alt"]):
        raise ValueError(f"HapStore variants do not match reference ALT alleles: {path}")


def _store_refs_for_chrom(
    config: SourcePairBuildConfig,
    sample_df: pd.DataFrame,
    *,
    chrom: int,
) -> list[tuple[str, Path]]:
    hap = config.hapstore
    if hap.layout == "single_store":
        assert hap.store_path is not None
        return [("__single_store__", Path(hap.store_path))]
    if hap.layout == "per_chr":
        return [
            (
                "__per_chr__",
                Path(hap.root_dir) / hap.file_pattern.format(chrom=int(chrom)),
            )
        ]

    column = config.samples.resolved_store_population_column()
    if column not in sample_df.columns:
        raise ValueError(f"sample metadata missing store population column: {column}")
    populations = list(dict.fromkeys(sample_df[column].astype(str).tolist()))
    return [
        (
            population,
            Path(hap.root_dir) / hap.file_pattern.format(population=population, chrom=int(chrom)),
        )
        for population in populations
    ]


def _open_store(path: Path, *, opener: StoreOpener | None) -> object:
    if opener is not None:
        return opener(path)
    if not path.exists():
        raise FileNotFoundError(f"HapStore path does not exist: {path}")
    try:
        from hapstore import HapStore  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("HapStore support requires the hapstore package") from exc
    return HapStore(path, mode="r")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string_or_string_list(value: Any) -> str | list[str]:
    values = _as_list(value)
    if len(values) == 1 and not isinstance(value, (list, tuple)):
        return str(values[0])
    return [str(item) for item in values]


__all__ = [
    "SourcePairBuildConfig",
    "SourcePairBuildResult",
    "SourcePairGroupConfig",
    "SourcePairHapStoreConfig",
    "SourcePairOutputConfig",
    "SourcePairSampleConfig",
    "build_sample_masks_for_store",
    "build_source_pairs_frame_for_chrom",
    "build_source_pairs_from_hapstore",
    "group_mask",
    "load_sample_metadata",
    "resolve_groups",
    "slugify_label",
    "write_source_pairs_from_hapstore",
]
