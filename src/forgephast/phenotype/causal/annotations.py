"""Annotation-backed weighting utilities for causal variant sampling."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..config import AnnotationSamplingConfig, CausalSelectionConfig


@dataclass(frozen=True)
class AnnotationIntervalSet:
    """One named annotation set with merged intervals and one multiplier."""

    name: str
    weight_multiplier: float
    intervals_by_chrom: Dict[int, tuple[np.ndarray, np.ndarray]]


def bundled_default_gene_annotation_path(genome_build: str) -> Path:
    """Return the path to the bundled default gene annotation asset."""
    if genome_build != "GRCh38":
        raise ValueError(f"Unsupported built-in gene annotation genome build: {genome_build}")
    path = resources.files("forgephast.phenotype.assets").joinpath(
        "grch38_default_gene_annotation.tsv"
    )
    return Path(path)


def normalize_chromosome(value: object) -> int:
    """Normalize chromosome labels like chr1/1 to integer autosomes."""
    text = str(value).strip()
    if not text:
        raise ValueError("Empty chromosome value")
    if text.lower().startswith("chr"):
        text = text[3:]
    try:
        chrom = int(text)
    except ValueError as exc:  # pragma: no cover - branch kept for clearer error text
        raise ValueError(f"Unsupported non-autosomal chromosome label: {value!r}") from exc
    if chrom <= 0:
        raise ValueError(f"Chromosome must be positive; got {value!r}")
    return chrom


def parse_chr_pos_variant_id(variant_id: object) -> tuple[int, int]:
    """Parse chrN:POS variant IDs into normalized coordinates."""
    value = str(variant_id).strip()
    if not value.startswith("chr") or ":" not in value:
        raise ValueError(f"Expected chrN:POS variant_id, got {variant_id!r}")
    chrom_text, pos_text = value[3:].split(":", 1)
    chrom = normalize_chromosome(chrom_text)
    pos = int(pos_text)
    if pos <= 0:
        raise ValueError(f"Variant position must be positive; got {variant_id!r}")
    return chrom, pos


def resolve_variant_coordinates(
    df: pd.DataFrame,
    config: CausalSelectionConfig,
) -> pd.DataFrame:
    """Attach normalized 1-based and 0-based coordinates to the pool dataframe."""
    resolved = df.copy()
    variant_col = config.variant_id_column
    ann_cfg = config.annotation_sampling
    chrom_col = ann_cfg.variant_chrom_column
    pos_col = ann_cfg.variant_pos_column

    if chrom_col in resolved.columns and pos_col in resolved.columns:
        chrom = resolved[chrom_col].map(normalize_chromosome)
        pos = pd.to_numeric(resolved[pos_col], errors="raise").astype(int)
    else:
        parsed = resolved[variant_col].map(parse_chr_pos_variant_id)
        chrom = parsed.map(lambda item: item[0]).astype(int)
        pos = parsed.map(lambda item: item[1]).astype(int)

    if (pos <= 0).any():
        raise ValueError("Variant positions must be positive integers")

    resolved["chrom"] = chrom.astype(int)
    resolved["pos"] = pos.astype(int)
    resolved["pos0"] = resolved["pos"].astype(np.int64) - 1
    return resolved


def _merge_intervals(intervals: pd.DataFrame) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    """Merge overlapping half-open intervals within each chromosome."""
    merged: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if intervals.empty:
        return merged
    for chrom, group in intervals.groupby("chrom", sort=True):
        starts = group["start0"].to_numpy(dtype=np.int64, copy=True)
        ends = group["end0"].to_numpy(dtype=np.int64, copy=True)
        order = np.argsort(starts, kind="mergesort")
        starts = starts[order]
        ends = ends[order]

        out_starts: List[int] = []
        out_ends: List[int] = []
        current_start = int(starts[0])
        current_end = int(ends[0])
        for start, end in zip(starts[1:], ends[1:]):
            start_i = int(start)
            end_i = int(end)
            if start_i <= current_end:
                current_end = max(current_end, end_i)
                continue
            out_starts.append(current_start)
            out_ends.append(current_end)
            current_start = start_i
            current_end = end_i
        out_starts.append(current_start)
        out_ends.append(current_end)
        merged[int(chrom)] = (
            np.asarray(out_starts, dtype=np.int64),
            np.asarray(out_ends, dtype=np.int64),
        )
    return merged


def _load_bed_intervals(path: Path) -> pd.DataFrame:
    """Load a standard BED file into normalized interval columns."""
    rows: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split("\t")
            if len(parts) < 3:
                raise ValueError(f"BED line must have at least 3 columns: {text!r}")
            chrom = normalize_chromosome(parts[0])
            start0 = int(parts[1])
            end0 = int(parts[2])
            if start0 < 0 or end0 <= start0:
                raise ValueError(f"Invalid BED interval in {path}: {text!r}")
            rows.append({"chrom": chrom, "start0": start0, "end0": end0})
    return pd.DataFrame(rows, columns=["chrom", "start0", "end0"])


def _load_default_gene_window_intervals(config: AnnotationSamplingConfig) -> pd.DataFrame:
    """Build the default gene-window interval table from the bundled asset."""
    gene_cfg = config.default_gene_window
    asset_path = bundled_default_gene_annotation_path(gene_cfg.genome_build)
    genes = pd.read_csv(asset_path, sep="\t")
    required = {"chrom", "gene_start", "gene_end", "strand", "tss", "gene_name"}
    missing = [col for col in required if col not in genes.columns]
    if missing:
        raise ValueError(
            f"Bundled default gene annotation missing required columns: {missing}"
        )

    genes = genes.copy()
    genes = genes[
        genes["chrom"].astype(str).str.fullmatch(r"(?:chr)?[0-9]+")
    ].copy()
    genes["chrom"] = genes["chrom"].map(normalize_chromosome).astype(int)
    genes["gene_start"] = pd.to_numeric(genes["gene_start"], errors="raise").astype(np.int64)
    genes["gene_end"] = pd.to_numeric(genes["gene_end"], errors="raise").astype(np.int64)
    genes["tss"] = pd.to_numeric(genes["tss"], errors="raise").astype(np.int64)

    start0 = np.where(
        genes["strand"].astype(str) == "+",
        np.maximum(0, genes["tss"].to_numpy(dtype=np.int64) - int(gene_cfg.upstream_bp)),
        genes["gene_start"].to_numpy(dtype=np.int64) - 1,
    )
    end0 = np.where(
        genes["strand"].astype(str) == "+",
        genes["gene_end"].to_numpy(dtype=np.int64),
        genes["tss"].to_numpy(dtype=np.int64) + int(gene_cfg.upstream_bp),
    )
    intervals = pd.DataFrame(
        {
            "chrom": genes["chrom"].astype(int),
            "start0": start0.astype(np.int64),
            "end0": end0.astype(np.int64),
        }
    )
    if (intervals["end0"] <= intervals["start0"]).any():
        raise ValueError("Default gene-window interval construction produced non-positive spans")
    return intervals


def _build_interval_sets(config: AnnotationSamplingConfig) -> list[AnnotationIntervalSet]:
    """Build all enabled annotation interval sets."""
    interval_sets: list[AnnotationIntervalSet] = []
    if config.default_gene_window.enabled:
        interval_sets.append(
            AnnotationIntervalSet(
                name="default_gene_window",
                weight_multiplier=float(config.default_gene_window.weight_multiplier),
                intervals_by_chrom=_merge_intervals(_load_default_gene_window_intervals(config)),
            )
        )
    for item in config.custom_annotations:
        interval_sets.append(
            AnnotationIntervalSet(
                name=str(item.name),
                weight_multiplier=float(item.weight_multiplier),
                intervals_by_chrom=_merge_intervals(_load_bed_intervals(Path(item.bed_file))),
            )
        )
    return interval_sets


def _interval_hits(pos0: np.ndarray, intervals: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Return a boolean hit vector for positions against merged intervals."""
    if intervals is None:
        return np.zeros(pos0.shape[0], dtype=bool)
    starts, ends = intervals
    if starts.size == 0:
        return np.zeros(pos0.shape[0], dtype=bool)
    idx = np.searchsorted(starts, pos0, side="right") - 1
    hits = idx >= 0
    if not np.any(hits):
        return hits
    valid_idx = idx[hits]
    hits[hits] = pos0[hits] < ends[valid_idx]
    return hits


def annotate_sampling_weights(
    df: pd.DataFrame,
    config: CausalSelectionConfig,
) -> pd.DataFrame:
    """Attach sampling weights and annotation metadata to the causal pool."""
    annotated = resolve_variant_coordinates(df, config)
    annotated["sampling_weight"] = 1.0
    annotated["annotation_hit"] = False
    labels: list[list[str]] = [[] for _ in range(annotated.shape[0])]

    ann_cfg = config.annotation_sampling
    if not ann_cfg.enabled:
        annotated["annotation_labels"] = ""
        return annotated

    interval_sets = _build_interval_sets(ann_cfg)
    if not interval_sets:
        annotated["annotation_labels"] = ""
        return annotated

    weights = annotated["sampling_weight"].to_numpy(dtype=np.float64, copy=True)
    hit_flags = annotated["annotation_hit"].to_numpy(dtype=bool, copy=True)
    chrom_vec = annotated["chrom"].to_numpy(dtype=np.int64, copy=False)
    pos0_vec = annotated["pos0"].to_numpy(dtype=np.int64, copy=False)

    for interval_set in interval_sets:
        current_hits = np.zeros(annotated.shape[0], dtype=bool)
        for chrom in np.unique(chrom_vec):
            chrom_mask = chrom_vec == int(chrom)
            if not np.any(chrom_mask):
                continue
            chrom_hits = _interval_hits(
                pos0_vec[chrom_mask],
                interval_set.intervals_by_chrom.get(int(chrom)),
            )
            current_hits[chrom_mask] = chrom_hits
        if not np.any(current_hits):
            continue
        weights[current_hits] = np.maximum(
            weights[current_hits],
            float(interval_set.weight_multiplier),
        )
        hit_flags |= current_hits
        for idx in np.flatnonzero(current_hits):
            labels[int(idx)].append(interval_set.name)

    annotated["sampling_weight"] = weights.astype(float)
    annotated["annotation_hit"] = hit_flags.astype(bool)
    annotated["annotation_labels"] = [",".join(item) for item in labels]
    return annotated


def load_annotation_intervals_for_config(
    config: AnnotationSamplingConfig,
) -> Iterable[AnnotationIntervalSet]:
    """Expose interval-set building for direct unit tests."""
    return tuple(_build_interval_sets(config))
