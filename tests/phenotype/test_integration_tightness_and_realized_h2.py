"""Integration smoke test: tightness filter → bucketed pool → write_outputs columns.

This test does NOT spin up a full ``run_simulation`` call (which would require a
hapstore fixture, causal pool, and a cohort config). Instead it exercises the
end-to-end path that matters for this feature:

1. Build a synthetic EUR/AFR MAF pool spanning fold-change tiers (1, 2, 5, 30x).
2. Run ``build_bucketed_pool`` at each tier and verify the tightness summary
   threads into ``BucketedPool.tightness_summary`` with the expected counts.
3. Simulate a records list (mirroring what ``run_simulation`` emits) with
   per-population ``realized_fixed_by_population`` and the bucketed pool's
   tightness summary attached.
4. Call ``write_simulation_outputs`` and verify the new columns land on
   ``component_summary.tsv`` and ``run_manifest.tsv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forgephast.phenotype.causal.selection import build_bucketed_pool
from forgephast.phenotype.config import (
    CausalSelectionConfig,
    MafBucketConfig,
    TightnessFilterConfig,
)
from forgephast.phenotype.io.write_outputs import write_simulation_outputs


def _write_fold_change_pool(pool_path: Path) -> None:
    pd.DataFrame(
        [
            {"causal_id": "chr1:1", "causal_maf_eur": 0.10, "causal_maf_afr": 0.10},  # 1x
            {"causal_id": "chr1:2", "causal_maf_eur": 0.20, "causal_maf_afr": 0.10},  # 2x
            {"causal_id": "chr1:3", "causal_maf_eur": 0.30, "causal_maf_afr": 0.06},  # 5x
            {"causal_id": "chr1:4", "causal_maf_eur": 0.30, "causal_maf_afr": 0.01},  # 30x
        ]
    ).to_csv(pool_path, sep="\t", index=False)


def _build_config(pool_path: Path, tier: str) -> CausalSelectionConfig:
    return CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={
            "EUR": "causal_maf_eur",
            "AFR": "causal_maf_afr",
        },
        anchor_population="EUR",
        maf_buckets=[
            MafBucketConfig("b1", 0.01, 0.5, fixed_count=1, subset_count=0),
        ],
        tightness_filter=TightnessFilterConfig(tier=tier),
        random_seed=1,
    )


def test_tightness_tiers_yield_expected_pool_sizes(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.tsv"
    _write_fold_change_pool(pool_path)

    tight = build_bucketed_pool(_build_config(pool_path, "tight"))
    moderate = build_bucketed_pool(_build_config(pool_path, "moderate"))
    extreme = build_bucketed_pool(_build_config(pool_path, "extreme"))

    assert len(tight.full_pool) == 2
    assert len(moderate.full_pool) == 3
    assert len(extreme.full_pool) == 4

    assert tight.tightness_summary["tier"] == "tight"
    assert tight.tightness_summary["dropped"] == 2
    assert moderate.tightness_summary["dropped"] == 1
    assert extreme.tightness_summary["dropped"] == 0


def test_write_outputs_flows_tightness_and_realized_h2_columns(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.tsv"
    _write_fold_change_pool(pool_path)

    bucketed = build_bucketed_pool(_build_config(pool_path, "moderate"))
    tightness_summary = dict(bucketed.tightness_summary)

    # Realized per-population h² — plausible unequal values for EUR/AFR, sitting
    # in the (0.05, 0.80) window the plan calls for in the end-to-end smoke.
    realized_by_pop = {"EUR": 0.18, "AFR": 0.24}

    record = {
        "replicate": 1,
        "architecture": "arch1",
        "cohort": "cohort_multi",
        "plugin": "polygenic_no_pleiotropy",
        "pgen_prefix": "",
        "sample_ids_file": "/tmp/samples.txt",
        "phenotype_df": pd.DataFrame(
            {"FID": ["s1"], "IID": ["s1"], "PHENO": [0.1]}
        ),
        "covariate_df": pd.DataFrame({"FID": ["s1"], "IID": ["s1"]}),
        "pcs_df": pd.DataFrame({"FID": ["s1"], "IID": ["s1"]}),
        "coefficients_df": pd.DataFrame({"covariate": ["age"], "beta": [0.2]}),
        "membership_df": pd.DataFrame(
            {
                "variant_id": ["v1"],
                "replicate": [1],
                "architecture": ["arch1"],
                "cohort": ["cohort_multi"],
                "maf_bucket": ["b1"],
                "annotation_hit": [False],
                "annotation_labels": [""],
                "sampling_weight": [1.0],
            }
        ),
        "effects_df": pd.DataFrame({"variant_id": ["v1"], "beta": [0.3]}),
        "component_targets": {
            "base_h2_fixed": 0.30,
            "resolved_h2_fixed": float("nan"),
            "h2_jitter_delta": float("nan"),
            "h2_jitter_clipped": False,
            "target_mode": "free_per_population",
            "resolved_h2_fixed_by_population_json": '{"AFR": 0.30, "EUR": 0.30}',
            "resolved_h2_bg_by_population_json": '{"AFR": 0.0, "EUR": 0.0}',
            "resolved_covariate_by_population_json": '{"AFR": 0.2, "EUR": 0.2}',
            "resolved_residual_by_population_json": '{"AFR": 0.5, "EUR": 0.5}',
            "h2_jitter_delta_by_population_json": '{"AFR": 0.0, "EUR": 0.0}',
            "h2_jitter_clipped_by_population_json": '{"AFR": false, "EUR": false}',
            "genetic_fixed": float("nan"),
            "genetic_bg": 0.0,
            "genetic_total": float("nan"),
            "covariate": 0.2,
            "residual": float("nan"),
        },
        "component_realized": {
            "genetic_fixed": 0.21,
            "genetic_bg": 0.0,
            "genetic_total": 0.21,
            "covariate": 0.2,
            "residual": 0.5,
            "phenotype": 0.91,
        },
        "realized_fixed_by_population": realized_by_pop,
        "tightness_summary": tightness_summary,
        "effect_geometry": {
            "population_corr_mode": "manual",
            "base_jitter_sd": 0.1,
            "resolved_jitter_sd_by_population": {"EUR": 0.1, "AFR": 0.1},
            "population_fst_file": "",
            "fst_corr_transform": "",
            "fst_corr_lambda": 0.0,
            "population_corr_matrix_json": "[[1.0,0.8],[0.8,1.0]]",
        },
    }

    output_root = tmp_path / "outputs"
    manifest_df = write_simulation_outputs("run_integration", output_root, [record])

    component_df = pd.read_csv(
        output_root / "run_integration" / "truth" / "component_summary.tsv",
        sep="\t",
    )

    # target_mode threads through as free_per_population
    assert component_df.loc[0, "target_mode"] == "free_per_population"
    assert manifest_df.loc[0, "target_mode"] == "free_per_population"

    # realized per-pop h² lands as sorted JSON
    realized_json = json.loads(component_df.loc[0, "realized_h2_fixed_by_population"])
    assert set(realized_json.keys()) == {"EUR", "AFR"}
    assert realized_json["EUR"] != realized_json["AFR"]  # not forced equal
    assert 0.05 < realized_json["EUR"] < 0.80
    assert 0.05 < realized_json["AFR"] < 0.80

    # tightness summary columns reflect the moderate tier (drop 1 of 4)
    assert component_df.loc[0, "shift_tightness_tier"] == "moderate"
    assert float(component_df.loc[0, "shift_tightness_fold_change_cap"]) == 8.0
    assert int(component_df.loc[0, "shift_tightness_pre_filter_count"]) == 4
    assert int(component_df.loc[0, "shift_tightness_post_filter_count"]) == 3

    quantiles = json.loads(
        component_df.loc[0, "shift_tightness_fold_change_quantiles_json"]
    )
    assert "EUR|AFR" in quantiles
    # max fold change among survivors = 5.0 (the ratio-5 variant)
    assert quantiles["EUR|AFR"]["max"] == 5.0
