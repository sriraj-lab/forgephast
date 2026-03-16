"""Causal pool and hybrid sampler tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forgephast.phenotype.causal.hybrid_sampler import sample_membership_for_replicate
from forgephast.phenotype.causal.selection import build_bucketed_pool
from forgephast.phenotype.config import CausalSelectionConfig, MafBucketConfig


def _write_pool(path: Path) -> None:
    rows = []
    for idx in range(100):
        maf = 0.01 + (0.01 * (idx % 10) / 10)
        rows.append(
            {
                "causal_id": f"chr1:{idx+1}",
                "causal_maf_eur": maf,
                "causal_maf_afr": maf + 0.001,
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_fixed_set_stable_subset_varies(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    _write_pool(pool_path)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        anchor_population="EUR",
        maf_buckets=[
            MafBucketConfig("b1", 0.01, 0.015, fixed_count=5, subset_count=4),
            MafBucketConfig("b2", 0.015, 0.02, fixed_count=6, subset_count=3),
        ],
        random_seed=42,
    )

    bucketed = build_bucketed_pool(cfg)

    rep1 = sample_membership_for_replicate(1, bucketed.full_pool, bucketed.fixed_set, cfg)
    rep2 = sample_membership_for_replicate(2, bucketed.full_pool, bucketed.fixed_set, cfg)

    fixed1 = set(rep1[rep1["membership_type"] == "fixed"]["causal_id"])
    fixed2 = set(rep2[rep2["membership_type"] == "fixed"]["causal_id"])
    assert fixed1 == fixed2

    subset1 = set(rep1[rep1["membership_type"] == "subset"]["causal_id"])
    subset2 = set(rep2[rep2["membership_type"] == "subset"]["causal_id"])
    assert subset1 != subset2

    counts = rep1.groupby(["maf_bucket", "membership_type"]).size().to_dict()
    assert counts[("b1", "fixed")] == 5
    assert counts[("b1", "subset")] == 4
    assert counts[("b2", "fixed")] == 6
    assert counts[("b2", "subset")] == 3


def test_variant_id_column_can_be_custom_name(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    _write_pool(pool_path)
    pool_df = pd.read_csv(pool_path, sep="\t").rename(columns={"causal_id": "variant_id"})
    pool_df.to_csv(pool_path, sep="\t", index=False)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="variant_id",
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        anchor_population="EUR",
        maf_buckets=[
            MafBucketConfig("b1", 0.01, 0.02, fixed_count=2, subset_count=2),
        ],
        random_seed=42,
    )

    bucketed = build_bucketed_pool(cfg)
    rep = sample_membership_for_replicate(1, bucketed.full_pool, bucketed.fixed_set, cfg)
    assert "variant_id" in rep.columns
    assert not rep["variant_id"].duplicated().any()


def test_duplicate_membership_rows_raise_runtime_error(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    _write_pool(pool_path)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "causal_maf_eur"},
        anchor_population="EUR",
        maf_buckets=[MafBucketConfig("b1", 0.01, 0.02, fixed_count=1, subset_count=0)],
        random_seed=1,
    )
    bucketed = build_bucketed_pool(cfg)

    dup_fixed = pd.concat([bucketed.fixed_set, bucketed.fixed_set], ignore_index=True)
    with pytest.raises(RuntimeError, match="Duplicate variant IDs detected"):
        sample_membership_for_replicate(
            1,
            full_pool=bucketed.full_pool,
            fixed_set=dup_fixed,
            config=cfg,
        )
