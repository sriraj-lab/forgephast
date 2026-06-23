"""Causal pool and hybrid sampler tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forgephast.phenotype.causal.hybrid_sampler import sample_membership_for_replicate
from forgephast.phenotype.causal.selection import (
    _apply_tightness_filter,
    build_bucketed_pool,
)
from forgephast.phenotype.config import (
    CausalSelectionConfig,
    MafBucketConfig,
    TightnessFilterConfig,
)


def _write_pool(path: Path) -> None:
    rows = []
    for idx in range(100):
        maf = 0.01 + (0.01 * (idx % 10) / 10)
        rows.append(
            {
                "causal_id": f"chr1:{idx+1}",
                "causal_maf_eur": maf,
                "causal_maf_afr": maf + 0.001,
                "maf_pooled": maf + 0.0005,
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


# ── Tightness filter tests ──────────────────────────────────────────────


def _fold_change_pool() -> pd.DataFrame:
    """Four variants spanning fold-change ratios 1, 2, 5, 30."""
    return pd.DataFrame(
        [
            {"causal_id": "chr1:1", "causal_maf_eur": 0.10, "causal_maf_afr": 0.10},  # ratio 1
            {"causal_id": "chr1:2", "causal_maf_eur": 0.20, "causal_maf_afr": 0.10},  # ratio 2
            {"causal_id": "chr1:3", "causal_maf_eur": 0.30, "causal_maf_afr": 0.06},  # ratio 5
            {"causal_id": "chr1:4", "causal_maf_eur": 0.30, "causal_maf_afr": 0.01},  # ratio 30
        ]
    )


def test_tightness_filter_tight_tier_drops_fold_change_above_2x() -> None:
    df = _fold_change_pool()
    filter_cfg = TightnessFilterConfig(tier="tight")
    filter_cfg.validate()
    kept, summary = _apply_tightness_filter(
        df,
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        filter_config=filter_cfg,
    )
    assert set(kept["causal_id"]) == {"chr1:1", "chr1:2"}
    assert summary["tier"] == "tight"
    assert summary["fold_change_cap"] == 2.0
    assert summary["pre_filter_count"] == 4
    assert summary["post_filter_count"] == 2
    assert summary["dropped"] == 2
    assert summary["fold_change_max_kept"] == pytest.approx(2.0, abs=1e-12)
    assert "EUR|AFR" in summary["per_pair_quantiles"]


def test_tightness_filter_moderate_tier_drops_fold_change_above_8x() -> None:
    df = _fold_change_pool()
    filter_cfg = TightnessFilterConfig(tier="moderate")
    kept, summary = _apply_tightness_filter(
        df,
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        filter_config=filter_cfg,
    )
    assert set(kept["causal_id"]) == {"chr1:1", "chr1:2", "chr1:3"}
    assert summary["fold_change_cap"] == 8.0
    assert summary["post_filter_count"] == 3
    assert summary["dropped"] == 1


def test_tightness_filter_extreme_tier_uses_300x_cap_and_reports_quantiles() -> None:
    df = _fold_change_pool()
    filter_cfg = TightnessFilterConfig(tier="extreme")
    kept, summary = _apply_tightness_filter(
        df,
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        filter_config=filter_cfg,
    )
    assert len(kept) == 4
    assert summary["tier"] == "extreme"
    assert summary["fold_change_cap"] == 300.0
    assert summary["dropped"] == 0
    # This toy pool tops out below the 300x cap, so extreme keeps everything.
    assert "EUR|AFR" in summary["per_pair_quantiles"]
    assert summary["per_pair_quantiles"]["EUR|AFR"]["max"] == pytest.approx(30.0, abs=1e-12)


def test_tightness_filter_zero_maf_uses_maf_floor() -> None:
    df = pd.DataFrame(
        [
            {"causal_id": "chr1:1", "causal_maf_eur": 0.0, "causal_maf_afr": 0.10},
            {"causal_id": "chr1:2", "causal_maf_eur": 0.10, "causal_maf_afr": 0.10},
        ]
    )
    filter_cfg = TightnessFilterConfig(tier="tight", maf_floor=1e-4)
    kept, summary = _apply_tightness_filter(
        df,
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        filter_config=filter_cfg,
    )
    # With floor 1e-4, the first variant's fold change is 0.10 / 1e-4 = 1000,
    # so it must be dropped by the tight tier (cap 2x).
    assert set(kept["causal_id"]) == {"chr1:2"}
    assert summary["dropped"] == 1


def test_tightness_filter_numeric_cap_overrides_tier() -> None:
    df = _fold_change_pool()
    filter_cfg = TightnessFilterConfig(tier="tight", fold_change_cap=10.0)
    filter_cfg.validate()
    assert filter_cfg.resolved_cap() == 10.0
    kept, summary = _apply_tightness_filter(
        df,
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        filter_config=filter_cfg,
    )
    # cap=10 keeps ratios 1, 2, 5 and drops 30
    assert set(kept["causal_id"]) == {"chr1:1", "chr1:2", "chr1:3"}
    assert summary["fold_change_cap"] == 10.0
    assert summary["tier"] == "custom:cap=10"


def test_tightness_filter_invalid_tier_raises() -> None:
    filter_cfg = TightnessFilterConfig(tier="super_tight")
    with pytest.raises(ValueError, match="tightness_filter.tier"):
        filter_cfg.validate()


def test_tightness_filter_cap_below_one_raises() -> None:
    filter_cfg = TightnessFilterConfig(tier="tight", fold_change_cap=0.5)
    with pytest.raises(ValueError, match="fold_change_cap"):
        filter_cfg.validate()


def test_tightness_filter_maf_floor_out_of_range_raises() -> None:
    filter_cfg = TightnessFilterConfig(maf_floor=0.0)
    with pytest.raises(ValueError, match="maf_floor"):
        filter_cfg.validate()


def test_tightness_filter_single_population_is_noop() -> None:
    df = _fold_change_pool()
    filter_cfg = TightnessFilterConfig(tier="tight")
    kept, summary = _apply_tightness_filter(
        df,
        maf_columns_by_population={"EUR": "causal_maf_eur"},
        filter_config=filter_cfg,
    )
    assert len(kept) == 4
    assert summary["dropped"] == 0


def test_build_bucketed_pool_populates_tightness_summary(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    _fold_change_pool().to_csv(pool_path, sep="\t", index=False)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        anchor_population="EUR",
        maf_buckets=[
            MafBucketConfig("b1", 0.01, 0.5, fixed_count=1, subset_count=0),
        ],
        tightness_filter=TightnessFilterConfig(tier="moderate"),
        random_seed=1,
    )
    bucketed = build_bucketed_pool(cfg)
    assert bucketed.tightness_summary["tier"] == "moderate"
    assert bucketed.tightness_summary["fold_change_cap"] == 8.0
    assert bucketed.tightness_summary["dropped"] == 1
    # moderate keeps variants with fold change in {1, 2, 5}
    assert len(bucketed.full_pool) == 3


def test_build_bucketed_pool_uses_subgroup_tightness_map_when_supplied(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    pd.DataFrame(
        [
            {
                "causal_id": "chr1:1",
                "maf_eur": 0.10,
                "maf_afr": 0.10,
                "maf_afr_gulf": 0.10,
                "maf_afr_east": 0.10,
                "maf_afr_west": 0.10,
                "maf_pooled": 0.10,
            },
            {
                "causal_id": "chr1:2",
                "maf_eur": 0.10,
                "maf_afr": 0.10,
                "maf_afr_gulf": 0.10,
                "maf_afr_east": 0.10,
                "maf_afr_west": 0.02,
                "maf_pooled": 0.10,
            },
        ]
    ).to_csv(pool_path, sep="\t", index=False)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur", "AFR": "maf_afr"},
        tightness_maf_columns_by_group={
            "EUR": "maf_eur",
            "AFR_GULF": "maf_afr_gulf",
            "AFR_EAST": "maf_afr_east",
            "AFR_WEST": "maf_afr_west",
        },
        anchor_population="EUR",
        maf_bucket_column="maf_pooled",
        effect_scale_maf_column="maf_pooled",
        maf_buckets=[
            MafBucketConfig("b1", 0.01, 0.5, fixed_count=1, subset_count=0),
        ],
        tightness_filter=TightnessFilterConfig(tier="tight"),
        random_seed=1,
    )
    bucketed = build_bucketed_pool(cfg)
    assert set(bucketed.full_pool["causal_id"]) == {"chr1:1"}
    assert bucketed.tightness_summary["dropped"] == 1
    assert "AFR_WEST|EUR" not in bucketed.tightness_summary["per_pair_quantiles"]
    assert "EUR|AFR_WEST" in bucketed.tightness_summary["per_pair_quantiles"]


def test_build_bucketed_pool_requires_subgroup_tightness_columns(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    pd.DataFrame(
        [
            {
                "causal_id": "chr1:1",
                "maf_eur": 0.10,
                "maf_afr": 0.10,
                "maf_pooled": 0.10,
            },
        ]
    ).to_csv(pool_path, sep="\t", index=False)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur", "AFR": "maf_afr"},
        tightness_maf_columns_by_group={"EUR": "maf_eur", "AFR_GULF": "maf_afr_gulf"},
        anchor_population="EUR",
        maf_bucket_column="maf_pooled",
        effect_scale_maf_column="maf_pooled",
        maf_buckets=[
            MafBucketConfig("b1", 0.01, 0.5, fixed_count=1, subset_count=0),
        ],
        tightness_filter=TightnessFilterConfig(tier="tight"),
        random_seed=1,
    )
    with pytest.raises(ValueError, match="maf_afr_gulf"):
        build_bucketed_pool(cfg)


def test_bucket_assignment_can_use_pooled_maf_column(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    pd.DataFrame(
        [
            {
                "causal_id": "chr1:1",
                "causal_maf_eur": 0.001,
                "causal_maf_afr": 0.002,
                "maf_pooled": 0.12,
            },
            {
                "causal_id": "chr1:2",
                "causal_maf_eur": 0.18,
                "causal_maf_afr": 0.16,
                "maf_pooled": 0.004,
            },
        ]
    ).to_csv(pool_path, sep="\t", index=False)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        anchor_population="EUR",
        maf_bucket_column="maf_pooled",
        effect_scale_maf_column="maf_pooled",
        maf_buckets=[
            MafBucketConfig("rare", 0.0, 0.01, fixed_count=0, subset_count=0),
            MafBucketConfig("common", 0.01, 0.5, fixed_count=0, subset_count=0),
        ],
        random_seed=1,
    )

    bucketed = build_bucketed_pool(cfg)
    observed = dict(zip(bucketed.full_pool["causal_id"], bucketed.full_pool["maf_bucket"]))
    assert observed["chr1:1"] == "common"
    assert observed["chr1:2"] == "rare"


def test_tightness_filter_empties_bucket_raises_with_hint(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    # Every row has EUR:AFR ratio 10 (>2x), so tight tier drops the whole pool.
    pd.DataFrame(
        [
            {"causal_id": f"chr1:{i}", "causal_maf_eur": 0.20, "causal_maf_afr": 0.02}
            for i in range(5)
        ]
    ).to_csv(pool_path, sep="\t", index=False)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "causal_maf_eur", "AFR": "causal_maf_afr"},
        anchor_population="EUR",
        maf_buckets=[
            MafBucketConfig("b1", 0.01, 0.5, fixed_count=1, subset_count=0),
        ],
        tightness_filter=TightnessFilterConfig(tier="tight"),
        random_seed=1,
    )
    with pytest.raises(ValueError, match="tightness"):
        build_bucketed_pool(cfg)
