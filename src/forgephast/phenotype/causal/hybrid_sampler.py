"""Hybrid fixed+subset causal membership sampling."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..config import CausalSelectionConfig


def sample_membership_for_replicate(
    replicate_idx: int,
    full_pool: pd.DataFrame,
    fixed_set: pd.DataFrame,
    config: CausalSelectionConfig,
) -> pd.DataFrame:
    """Sample per-replicate subset variants by MAF bucket and combine with fixed set."""
    variant_col = config.variant_id_column

    rng = np.random.default_rng(config.random_seed + replicate_idx)

    fixed_ids = set(fixed_set[variant_col].tolist()) if not fixed_set.empty else set()
    subset_candidates = full_pool[~full_pool[variant_col].isin(fixed_ids)].copy()

    sampled_frames: List[pd.DataFrame] = []
    for bucket in config.maf_buckets:
        bucket_df = subset_candidates[subset_candidates["maf_bucket"] == bucket.name]
        if bucket.subset_count == 0:
            continue
        if len(bucket_df) < bucket.subset_count:
            raise ValueError(
                f"replicate {replicate_idx}: bucket {bucket.name} has only {len(bucket_df)} "
                f"candidates but subset_count={bucket.subset_count}"
            )

        picks = rng.choice(bucket_df.index.to_numpy(), size=bucket.subset_count, replace=False)
        sampled = bucket_df.loc[picks].copy()
        sampled["membership_type"] = "subset"
        sampled_frames.append(sampled)

    frames = []
    if not fixed_set.empty:
        frames.append(fixed_set.copy())
    if sampled_frames:
        frames.append(pd.concat(sampled_frames, ignore_index=True))

    if not frames:
        raise ValueError("No selected causal variants (fixed + subset both empty)")

    membership = pd.concat(frames, ignore_index=True)
    dup = membership[membership.duplicated(subset=[variant_col], keep=False)]
    if not dup.empty:
        duplicate_ids = dup[variant_col].astype(str).unique().tolist()
        raise RuntimeError(
            "Duplicate variant IDs detected while combining fixed/subset membership: "
            f"{duplicate_ids[:5]} (n={len(duplicate_ids)})"
        )
    membership["replicate"] = replicate_idx

    return membership
