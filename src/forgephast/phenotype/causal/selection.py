"""Causal pool loading and MAF-bucket selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import CausalSelectionConfig, MafBucketConfig

logger = logging.getLogger(__name__)


@dataclass
class BucketedPool:
    """Container for bucketed causal pool and fixed set."""

    full_pool: pd.DataFrame
    fixed_set: pd.DataFrame


def _bucket_mask(df: pd.DataFrame, maf_col: str, bucket: MafBucketConfig) -> pd.Series:
    """Boolean mask for selecting rows in one MAF bucket."""
    return (df[maf_col] >= bucket.maf_min) & (df[maf_col] < bucket.maf_max)


def load_causal_pool(config: CausalSelectionConfig) -> pd.DataFrame:
    """Load and validate causal pool table."""
    df = pd.read_csv(config.source_pairs_file, sep="\t")
    required = {config.variant_id_column}
    for maf_col in config.maf_columns_by_population.values():
        required.add(maf_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"causal pool missing required columns: {missing}")

    variant_col = config.variant_id_column
    if df[variant_col].duplicated().any():
        logger.warning("Duplicate variant IDs in pool; keeping first occurrence")
        df = df.drop_duplicates(subset=[variant_col], keep="first").copy()

    # Normalize anchor MAF column for bucketing.
    anchor_pop = config.anchor_population
    if anchor_pop is None:
        anchor_pop = next(iter(config.maf_columns_by_population.keys()))
    anchor_maf_col = config.maf_columns_by_population[anchor_pop]
    df = df.copy()
    df["_anchor_maf"] = pd.to_numeric(df[anchor_maf_col], errors="coerce")
    if df["_anchor_maf"].isna().any():
        raise ValueError(f"NaN MAF values found in anchor MAF column {anchor_maf_col}")

    # Assign bucket labels.
    df["maf_bucket"] = ""
    for bucket in config.maf_buckets:
        mask = _bucket_mask(df, "_anchor_maf", bucket)
        df.loc[mask, "maf_bucket"] = bucket.name

    uncovered = df[df["maf_bucket"] == ""]
    if not uncovered.empty:
        logger.info("Dropping %d variants outside configured MAF buckets", len(uncovered))
        df = df[df["maf_bucket"] != ""].copy()

    if df.empty:
        raise ValueError("No variants remain in causal pool after MAF bucket assignment")

    return df


def select_fixed_set(
    df: pd.DataFrame,
    config: CausalSelectionConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Select fixed variants per bucket once per run."""
    variant_col = config.variant_id_column
    fixed_frames: List[pd.DataFrame] = []

    for bucket in config.maf_buckets:
        bucket_df = df[df["maf_bucket"] == bucket.name]
        available = len(bucket_df)
        required = bucket.fixed_count + bucket.subset_count
        if required > available:
            raise ValueError(
                f"Bucket {bucket.name}: need fixed+subset={required} but only {available} variants available"
            )
        if bucket.fixed_count == 0:
            continue

        picks = rng.choice(bucket_df.index.to_numpy(), size=bucket.fixed_count, replace=False)
        fixed_bucket = bucket_df.loc[picks].copy()
        fixed_bucket["membership_type"] = "fixed"
        fixed_frames.append(fixed_bucket)

    if not fixed_frames:
        return pd.DataFrame(columns=list(df.columns) + ["membership_type"])

    fixed_set = pd.concat(fixed_frames, ignore_index=True)
    if fixed_set[variant_col].duplicated().any():
        raise RuntimeError("Unexpected duplicate variants in fixed set")

    return fixed_set


def build_bucketed_pool(config: CausalSelectionConfig) -> BucketedPool:
    """Build full bucketed pool and run-level fixed set."""
    rng = np.random.default_rng(config.random_seed)
    full = load_causal_pool(config)
    fixed = select_fixed_set(full, config, rng)
    logger.info("Causal pool variants: %d, fixed set variants: %d", len(full), len(fixed))
    return BucketedPool(full_pool=full, fixed_set=fixed)
