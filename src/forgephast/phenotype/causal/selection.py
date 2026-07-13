"""Causal pool loading and MAF-bucket selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd

from .annotations import annotate_sampling_weights
from ..config import CausalSelectionConfig, MafBucketConfig, TightnessFilterConfig

logger = logging.getLogger(__name__)


@dataclass
class BucketedPool:
    """Container for bucketed causal pool and fixed set."""

    full_pool: pd.DataFrame
    fixed_set: pd.DataFrame
    tightness_summary: Dict[str, object] = field(default_factory=dict)


def _bucket_mask(df: pd.DataFrame, maf_col: str, bucket: MafBucketConfig) -> pd.Series:
    """Boolean mask for selecting rows in one MAF bucket."""
    return (df[maf_col] >= bucket.maf_min) & (df[maf_col] < bucket.maf_max)


def _apply_tightness_filter(
    df: pd.DataFrame,
    *,
    maf_columns_by_population: Dict[str, str],
    filter_config: TightnessFilterConfig,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Drop variants whose max-over-pairs MAF fold change exceeds the configured cap.

    Returns a new dataframe (possibly filtered) and a summary dict with tier,
    cap, pre/post counts, and per-pair fold-change quantiles of the survivors.
    """
    pops = list(maf_columns_by_population.keys())
    tier_label = filter_config.label()
    cap = filter_config.resolved_cap()
    pre_count = int(len(df))

    empty_summary: Dict[str, object] = {
        "tier": tier_label,
        "fold_change_cap": float(cap),
        "maf_floor": float(filter_config.maf_floor),
        "maf_groups": list(pops),
        "maf_columns_by_group": dict(maf_columns_by_population),
        "pre_filter_count": pre_count,
        "post_filter_count": pre_count,
        "dropped": 0,
        "fold_change_max_kept": float("nan"),
        "per_pair_quantiles": {},
    }

    if pre_count == 0:
        logger.info(
            "tightness filter tier=%s cap=%s kept=0 dropped=0 (empty input pool)",
            tier_label,
            f"{cap:g}",
        )
        return df, empty_summary

    if len(pops) < 2:
        logger.warning(
            "tightness filter tier=%s requested but only one population MAF column "
            "is configured; skipping filter",
            tier_label,
        )
        return df, empty_summary

    maf_arr = np.empty((pre_count, len(pops)), dtype=np.float64)
    for col_idx, pop in enumerate(pops):
        maf_col = maf_columns_by_population[pop]
        maf_series = pd.to_numeric(df[maf_col], errors="raise")
        if maf_series.isna().any():
            raise ValueError(
                f"NaN MAF values found in column {maf_col!r} during tightness filtering"
            )
        maf_arr[:, col_idx] = maf_series.to_numpy(dtype=np.float64)

    floor = float(filter_config.maf_floor)
    maf_clipped = np.clip(maf_arr, floor, None)

    pair_ratios: List[np.ndarray] = []
    pair_quantiles: Dict[str, Dict[str, float]] = {}
    for i, j in combinations(range(len(pops)), 2):
        maf_i = maf_clipped[:, i]
        maf_j = maf_clipped[:, j]
        ratio = np.maximum(maf_i, maf_j) / np.minimum(maf_i, maf_j)
        pair_ratios.append(ratio)

    if not pair_ratios:
        return df, empty_summary

    ratio_stack = np.stack(pair_ratios, axis=1)
    fold_change_max = ratio_stack.max(axis=1)

    if np.isfinite(cap):
        keep_mask = fold_change_max <= cap
    else:
        keep_mask = np.ones(pre_count, dtype=bool)

    kept_df = df.loc[keep_mask].copy()
    kept_df["_fold_change_max"] = fold_change_max[keep_mask]

    # Per-pair quantiles computed over surviving rows (matches what the kept pool looks like).
    if int(keep_mask.sum()) > 0:
        kept_ratio_stack = ratio_stack[keep_mask]
        for pair_idx, (i, j) in enumerate(combinations(range(len(pops)), 2)):
            pair_label = f"{pops[i]}|{pops[j]}"
            pair_col = kept_ratio_stack[:, pair_idx]
            pair_quantiles[pair_label] = {
                "q50": float(np.quantile(pair_col, 0.50)),
                "q75": float(np.quantile(pair_col, 0.75)),
                "q90": float(np.quantile(pair_col, 0.90)),
                "q99": float(np.quantile(pair_col, 0.99)),
                "max": float(pair_col.max()),
            }
        fold_change_max_kept = float(fold_change_max[keep_mask].max())
    else:
        fold_change_max_kept = float("nan")

    post_count = int(keep_mask.sum())
    dropped = pre_count - post_count

    summary: Dict[str, object] = {
        "tier": tier_label,
        "fold_change_cap": float(cap),
        "maf_floor": floor,
        "maf_groups": list(pops),
        "maf_columns_by_group": dict(maf_columns_by_population),
        "pre_filter_count": pre_count,
        "post_filter_count": post_count,
        "dropped": int(dropped),
        "fold_change_max_kept": fold_change_max_kept,
        "per_pair_quantiles": pair_quantiles,
    }

    logger.info(
        "tightness filter tier=%s cap=%s kept=%d dropped=%d",
        tier_label,
        f"{cap:g}",
        post_count,
        dropped,
    )

    return kept_df, summary


def sample_rows_without_replacement(
    df: pd.DataFrame,
    *,
    count: int,
    rng: np.random.Generator,
    weight_column: str = "sampling_weight",
) -> pd.DataFrame:
    """Sample rows without replacement, optionally using per-row weights."""
    if count == 0:
        return df.iloc[0:0].copy()
    if count > len(df):
        raise ValueError(f"Cannot sample count={count} from only {len(df)} rows")

    weights = pd.to_numeric(df.get(weight_column, 1.0), errors="raise").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError(f"Sampling weights in column {weight_column!r} must be finite and > 0")

    indices = df.index.to_numpy()
    if np.allclose(weights, weights[0]):
        picks = rng.choice(indices, size=count, replace=False)
    else:
        probs = weights / weights.sum()
        picks = rng.choice(indices, size=count, replace=False, p=probs)
    return df.loc[picks].copy()


def load_causal_pool(
    config: CausalSelectionConfig,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Load, tightness-filter, and bucket the causal pool.

    Returns the annotated/bucketed dataframe and the tightness filter summary
    dict produced by :func:`_apply_tightness_filter`. The summary is threaded
    up to the simulate loop so it can be persisted in component_summary.tsv.
    """
    df = pd.read_csv(config.source_pairs_file, sep="\t")
    required = {config.variant_id_column}
    for maf_col in config.maf_columns_by_population.values():
        required.add(maf_col)
    tightness_maf_columns = config.tightness_maf_columns()
    for maf_col in tightness_maf_columns.values():
        required.add(maf_col)
    if config.maf_bucket_column is not None:
        required.add(str(config.maf_bucket_column))
    if config.effect_scale_maf_column is not None:
        required.add(str(config.effect_scale_maf_column))

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"causal pool missing required columns: {missing}")

    variant_col = config.variant_id_column
    if df[variant_col].duplicated().any():
        duplicates = df.loc[df[variant_col].duplicated(keep=False), variant_col].astype(str).head(5).tolist()
        raise ValueError(
            f"causal pool contains duplicate variant IDs; first duplicates={duplicates}"
        )

    # Tightness filter runs on the broadest pool, before bucket assignment,
    # so tier changes can be compared on equal footing for reporting.
    df, tightness_summary = _apply_tightness_filter(
        df,
        maf_columns_by_population=tightness_maf_columns,
        filter_config=config.tightness_filter,
    )

    if df.empty:
        raise ValueError(
            "No variants remain in causal pool after tightness filtering; "
            f"tightness tier={config.tightness_filter.label()} "
            f"cap={config.tightness_filter.resolved_cap():g} "
            f"(loosen the tier to keep more variants)"
        )

    # Normalize anchor MAF column for bucketing.
    bucket_maf_col = config.maf_bucket_column
    if bucket_maf_col is None:
        anchor_pop = config.anchor_population
        if anchor_pop is None:
            anchor_pop = next(iter(config.maf_columns_by_population.keys()))
        bucket_maf_col = config.maf_columns_by_population[anchor_pop]
    df = df.copy()
    df["_anchor_maf"] = pd.to_numeric(df[bucket_maf_col], errors="coerce")
    if df["_anchor_maf"].isna().any():
        raise ValueError(f"NaN MAF values found in bucket MAF column {bucket_maf_col}")

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

    return annotate_sampling_weights(df, config), tightness_summary


def select_fixed_set(
    df: pd.DataFrame,
    config: CausalSelectionConfig,
    rng: np.random.Generator,
    tightness_summary: Dict[str, object] | None = None,
) -> pd.DataFrame:
    """Select fixed variants per bucket once per run."""
    variant_col = config.variant_id_column
    fixed_frames: List[pd.DataFrame] = []

    dropped_by_tightness = 0
    if tightness_summary is not None:
        try:
            dropped_by_tightness = int(tightness_summary.get("dropped", 0))
        except Exception:
            dropped_by_tightness = 0

    for bucket in config.maf_buckets:
        bucket_df = df[df["maf_bucket"] == bucket.name]
        available = len(bucket_df)
        required = bucket.fixed_count + bucket.subset_count
        if required > available:
            hint = ""
            if dropped_by_tightness > 0:
                hint = (
                    f" (tightness tier={config.tightness_filter.label()} dropped "
                    f"{dropped_by_tightness} variants from the pool; loosen the "
                    "tier to keep more)"
                )
            raise ValueError(
                f"Bucket {bucket.name}: need fixed+subset={required} but only "
                f"{available} variants available{hint}"
            )
        if bucket.fixed_count == 0:
            continue

        fixed_bucket = sample_rows_without_replacement(
            bucket_df,
            count=bucket.fixed_count,
            rng=rng,
        )
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
    full, tightness_summary = load_causal_pool(config)
    fixed = select_fixed_set(full, config, rng, tightness_summary=tightness_summary)
    logger.info(
        "Causal pool variants: %d, fixed set variants: %d (tightness_tier=%s cap=%s)",
        len(full),
        len(fixed),
        tightness_summary.get("tier", "extreme"),
        f"{float(tightness_summary.get('fold_change_cap', float('inf'))):g}",
    )
    return BucketedPool(
        full_pool=full,
        fixed_set=fixed,
        tightness_summary=tightness_summary,
    )
