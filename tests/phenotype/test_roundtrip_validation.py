"""Round-trip validation report tests."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from forgephast.local_ancestry.source import LocalAncestryTract, write_local_ancestry_source
from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.validation.roundtrip import (
    _linear_metrics,
    _population_weights,
    build_roundtrip_report,
)


def _base_payload(tmp_path: Path) -> dict:
    sample_ids = tmp_path / "samples.txt"
    sample_ids.write_text("s1\ns2\n", encoding="utf-8")
    pop_map = tmp_path / "pop.tsv"
    pop_map.write_text("IID\tpopulation\ns1\tEUR\ns2\tAFR\n", encoding="utf-8")

    cov = tmp_path / "cov.tsv"
    cov.write_text("FID\tIID\tage\ns1\ts1\t30\ns2\ts2\t40\n", encoding="utf-8")
    cov_fx = tmp_path / "cov_fx.tsv"
    cov_fx.write_text("covariate\tbeta\nage\t0.1\n", encoding="utf-8")

    return {
        "name": "rt_unit",
        "genotype_input": {
            "format": "hapstore",
            "hapstore": {
                "root_dir": "/tmp/fake",
                "layout": "per_pop_per_chr",
                "file_pattern": "{population}/chr{chrom}.hapstore.zarr",
            },
        },
        "cohorts": [
            {
                "name": "c1",
                "sample_ids_file": str(sample_ids),
                "population_map_file": str(pop_map),
            }
        ],
        "architectures": [
            {
                "name": "a1",
                "plugin": "polygenic_no_pleiotropy",
                "params": {"sigma_beta": 1.0, "h2": 0.2},
            }
        ],
        "causal_selection": {
            "source_pairs_file": "/tmp/pairs.tsv",
            "variant_id_column": "causal_id",
            "maf_columns_by_population": {"EUR": "maf_eur", "AFR": "maf_afr"},
            "anchor_population": "EUR",
            "maf_buckets": [
                {
                    "name": "b1",
                    "maf_min": 0.01,
                    "maf_max": 0.02,
                    "fixed_count": 5,
                    "subset_count": 2,
                }
            ],
        },
        "effect_model": {
            "population_order": ["EUR", "AFR"],
            "population_corr_matrix": [[1.0, 0.9], [0.9, 1.0]],
            "jitter_sd_by_population": {"EUR": 0.1, "AFR": 0.1},
        },
        "covariates": {
            "effects_mode": "fixed",
            "covariate_file": str(cov),
            "id_column": "IID",
            "fixed_effects_file": str(cov_fx),
        },
        "phenotype": {"continuous_only": True, "n_replicates": 1},
        "variance_budget": {
            "h2_fixed": 0.2,
            "h2_bg": 0.0,
            "covariate": 0.2,
            "residual": 0.6,
            "strict_sum_to_one": True,
        },
        "gwas": {"global_pcs": {"mode": "none"}},
        "roundtrip_validation": {
            "enabled": True,
            "top_k": 2,
            "require_positive_slope": True,
        },
        "paths": {"output_root": str(tmp_path / "out")},
    }


def test_build_roundtrip_report_harmonizes_allele_and_computes_metrics(tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_base_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    truth = pd.DataFrame(
        {
            "variant_id": ["chr1:10", "chr1:20", "chr1:10", "chr1:20"],
            "population": ["EUR", "EUR", "AFR", "AFR"],
            "beta": [0.5, -0.4, 0.5, -0.4],
            "dosage_effect_allele": ["A", "G", "A", "G"],
            "dosage_other_allele": ["C", "A", "C", "A"],
            "architecture": ["a1", "a1", "a1", "a1"],
            "cohort": ["c1", "c1", "c1", "c1"],
            "replicate": [1, 1, 1, 1],
        }
    )
    truth.to_csv(truth_dir / "effect_sizes.tsv", sep="\t", index=False)

    membership = pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
        }
    )
    membership.to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)

    assoc = pd.DataFrame(
        {
            "ID": ["chr1:10", "chr1:20", "chr1:30"],
            "A1": ["A", "A", "T"],  # chr1:20 mismatches dosage allele G and should be sign-flipped
            "BETA": [0.45, 0.35, 0.01],
            "P": [1e-9, 2e-8, 0.8],
        }
    )
    assoc_path = run_dir / "assoc.tsv"
    run_dir.mkdir(parents=True, exist_ok=True)
    assoc.to_csv(assoc_path, sep="\t", index=False)

    gwas_manifest = pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "engine": ["plink2"],
            "status": ["success"],
            "primary_output": [str(assoc_path)],
        }
    )
    gwas_manifest_path = run_dir / "gwas_manifest.tsv"
    gwas_manifest.to_csv(gwas_manifest_path, sep="\t", index=False)

    report_path = build_roundtrip_report(
        run_cfg=cfg,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )
    assert report_path.exists()
    detail_path = report_path.with_name("roundtrip_validation_details.tsv")
    assert detail_path.exists()
    report = pd.read_csv(report_path, sep="\t")
    assert report.shape[0] == 1
    assert float(report.loc[0, "beta_slope"]) > 0.0
    assert int(report.loc[0, "top_k_causal_count"]) == 2
    details = pd.read_csv(detail_path, sep="\t")
    assert "expected_liability_beta" in details.columns


def test_build_roundtrip_report_validates_binary_trait_on_log_odds_scale(tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_base_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "variant_id": ["chr1:10", "chr1:20", "chr1:10", "chr1:20"],
            "population": ["EUR", "EUR", "AFR", "AFR"],
            "beta": [0.4, -0.2, 0.4, -0.2],
            "dosage_effect_allele": ["A", "G", "A", "G"],
            "dosage_other_allele": ["C", "A", "C", "A"],
            "architecture": ["a1", "a1", "a1", "a1"],
            "cohort": ["c1", "c1", "c1", "c1"],
            "replicate": [1, 1, 1, 1],
        }
    ).to_csv(truth_dir / "effect_sizes.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
        }
    ).to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "trait_type": ["binary"],
            "realized_var_liability": [4.0],
            "binary_liability_logit_scale": [1.5],
        }
    ).to_csv(truth_dir / "component_summary.tsv", sep="\t", index=False)

    # Expected binary truth is liability_beta * gamma / sd(liability):
    # chr1:10 = 0.4 * 1.5 / 2 = 0.3
    # chr1:20 = -0.2 * 1.5 / 2 = -0.15, then allele-harmonized from A back to G.
    assoc_path = run_dir / "step2_PHENO.regenie"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ID": ["chr1:10", "chr1:20", "chr1:30"],
            "A1": ["A", "A", "T"],
            "BETA": [0.30, 0.15, 0.01],
            "P": [1e-9, 2e-8, 0.9],
        }
    ).to_csv(assoc_path, sep="\t", index=False)

    gwas_manifest = pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "engine": ["regenie"],
            "trait_type": ["binary"],
            "status": ["success"],
            "primary_output": [str(assoc_path)],
        }
    )
    gwas_manifest_path = run_dir / "gwas_manifest.tsv"
    gwas_manifest.to_csv(gwas_manifest_path, sep="\t", index=False)

    report_path = build_roundtrip_report(
        run_cfg=cfg,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )

    report = pd.read_csv(report_path, sep="\t")
    assert report.loc[0, "trait_type"] == "binary"
    assert report.loc[0, "validation_status"] == "validated_binary_log_odds"
    assert float(report.loc[0, "binary_validation_scale"]) == 0.75
    assert float(report.loc[0, "binary_liability_logit_scale"]) == 1.5
    assert float(report.loc[0, "binary_liability_sd"]) == 2.0
    assert abs(float(report.loc[0, "beta_slope"]) - 1.0) < 1e-12
    assert int(report.loc[0, "top_k_causal_count"]) == 2
    details = pd.read_csv(report_path.with_name("roundtrip_validation_details.tsv"), sep="\t")
    detail_by_id = details.set_index("variant_id")
    assert "expected_log_odds_beta" in details.columns
    assert "beta_true_liability" in details.columns
    assert float(detail_by_id.loc["chr1:10", "beta_true_liability"]) == 0.4
    assert float(detail_by_id.loc["chr1:10", "expected_log_odds_beta"]) == 0.3
    assert float(detail_by_id.loc["chr1:20", "beta_true_liability"]) == -0.2
    assert float(detail_by_id.loc["chr1:20", "expected_log_odds_beta"]) == -0.15


def test_population_weights_expand_admix_from_local_ancestry_tracts(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    sample_ids = tmp_path / "samples_admix.txt"
    sample_ids.write_text("eur1\nadm1\n", encoding="utf-8")
    pop_map = tmp_path / "pop_admix.tsv"
    pop_map.write_text("IID\tpopulation\neur1\tEUR\nadm1\tADMIX\n", encoding="utf-8")
    tracts = tmp_path / "chr1_tracts.tsv.gz"
    with gzip.open(tracts, "wt", encoding="utf-8") as handle:
        handle.write("sample_id\thap_idx\tleft_bp\tright_bp\tancestry\n")
        handle.write("adm1\t0\t0\t40\tEUR\n")
        handle.write("adm1\t0\t40\t100\tAFR\n")
        handle.write("adm1\t1\t0\t100\tAFR\n")

    payload["cohorts"][0]["sample_ids_file"] = str(sample_ids)
    payload["cohorts"][0]["population_map_file"] = str(pop_map)
    payload["cohorts"][0]["local_ancestry_tracts_pattern"] = str(tracts)
    payload["chromosomes"] = [1]
    payload["local_ancestry"] = {
        "enabled": True,
        "ancestry_order": ["EUR", "AFR"],
        "label_map": {},
    }

    cfg = RunConfig.from_dict(payload)
    weights = _population_weights(cfg.cohorts[0], ["eur1", "adm1"], cfg)

    assert set(weights) == {"EUR", "AFR"}
    assert abs(weights["EUR"] - 0.6) < 1e-9
    assert abs(weights["AFR"] - 0.4) < 1e-9


def test_population_weights_expand_admix_from_local_ancestry_source(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    sample_ids = tmp_path / "samples_admix_source.txt"
    sample_ids.write_text("eur1\nadm1\n", encoding="utf-8")
    pop_map = tmp_path / "pop_admix_source.tsv"
    pop_map.write_text("IID\tpopulation\neur1\tEUR\nadm1\tADMIX\n", encoding="utf-8")
    source_path = tmp_path / "chr1.local_ancestry.zarr"
    write_local_ancestry_source(
        source_path,
        sample_ids=["adm1"],
        tracts_by_sample_hap={
            ("adm1", 0): [
                LocalAncestryTract(0, 40, "EUR"),
                LocalAncestryTract(40, 100, "AFR"),
            ],
            ("adm1", 1): [LocalAncestryTract(0, 100, "AFR")],
        },
        ancestry_order=["EUR", "AFR"],
        chrom=1,
    )
    payload["cohorts"][0]["sample_ids_file"] = str(sample_ids)
    payload["cohorts"][0]["population_map_file"] = str(pop_map)
    payload["cohorts"][0]["local_ancestry_source_pattern"] = str(source_path)
    payload["chromosomes"] = [1]
    payload["local_ancestry"] = {
        "enabled": True,
        "mode": "source_zarr",
        "ancestry_order": ["EUR", "AFR"],
        "label_map": {},
    }

    cfg = RunConfig.from_dict(payload)
    weights = _population_weights(cfg.cohorts[0], ["eur1", "adm1"], cfg)

    assert set(weights) == {"EUR", "AFR"}
    assert abs(weights["EUR"] - 0.6) < 1e-9
    assert abs(weights["AFR"] - 0.4) < 1e-9


def test_linear_metrics_returns_nan_for_degenerate_null_fit() -> None:
    merged = pd.DataFrame(
        {
            "beta_true": [0.0, 0.0, 0.0],
            "beta_hat_harmonized": [0.0, 0.0, 0.0],
        }
    )

    slope, intercept, corr, sign = _linear_metrics(merged)

    assert pd.isna(slope)
    assert pd.isna(intercept)
    assert pd.isna(corr)
    assert pd.isna(sign)


def test_build_roundtrip_report_ids_only_reuses_base_truth(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["name"] = "rt_unit__ids_only"
    cfg = RunConfig.from_dict(payload)
    run_dir = Path(cfg.paths.output_root) / cfg.name
    base_truth_dir = run_dir.parent / "rt_unit" / "truth"
    base_truth_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "variant_id": ["chr1:10", "chr1:20", "chr1:10", "chr1:20"],
            "population": ["EUR", "EUR", "AFR", "AFR"],
            "beta": [0.5, -0.4, 0.5, -0.4],
            "dosage_effect_allele": ["A", "G", "A", "G"],
            "dosage_other_allele": ["C", "A", "C", "A"],
            "architecture": ["a1", "a1", "a1", "a1"],
            "cohort": ["c1", "c1", "c1", "c1"],
            "replicate": [1, 1, 1, 1],
        }
    ).to_csv(base_truth_dir / "effect_sizes.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
        }
    ).to_csv(base_truth_dir / "causal_membership.tsv", sep="\t", index=False)

    run_dir.mkdir(parents=True, exist_ok=True)
    assoc_path = run_dir / "assoc.tsv"
    pd.DataFrame(
        {
            "ID": ["chr1:10", "chr1:20"],
            "A1": ["A", "A"],
            "BETA": [0.45, 0.35],
            "P": [1e-9, 2e-8],
        }
    ).to_csv(assoc_path, sep="\t", index=False)
    gwas_manifest_path = run_dir / "gwas_manifest.tsv"
    pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "engine": ["plink2"],
            "status": ["success"],
            "primary_output": [str(assoc_path)],
        }
    ).to_csv(gwas_manifest_path, sep="\t", index=False)

    report_path = build_roundtrip_report(
        run_cfg=cfg,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )

    assert report_path.exists()


def test_build_roundtrip_report_regenie_resolves_assoc_output_and_parses_trait_suffix_columns(
    tmp_path: Path,
) -> None:
    cfg = RunConfig.from_dict(_base_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "variant_id": ["chr1:10", "chr1:20", "chr1:10", "chr1:20"],
            "population": ["EUR", "EUR", "AFR", "AFR"],
            "beta": [0.5, -0.4, 0.5, -0.4],
            "dosage_effect_allele": ["A", "G", "A", "G"],
            "dosage_other_allele": ["C", "A", "C", "A"],
            "architecture": ["a1", "a1", "a1", "a1"],
            "cohort": ["c1", "c1", "c1", "c1"],
            "replicate": [1, 1, 1, 1],
        }
    ).to_csv(truth_dir / "effect_sizes.tsv", sep="\t", index=False)

    pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
        }
    ).to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)

    assoc_path = run_dir / "step2.regenie.gz"
    ydict_path = run_dir / "step2.regenie.Ydict"
    run_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(assoc_path, "wt", encoding="utf-8") as handle:
        handle.write(
            "CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA.Y1 SE.Y1 CHISQ.Y1 LOG10P.Y1 EXTRA\n"
        )
        handle.write("1 10 rs1 C A 0.1 2 ADD 0.45 0.1 20 9.0 NA\n")
        handle.write("1 20 rs2 G A 0.1 2 ADD 0.35 0.1 20 8.0 NA\n")
    ydict_path.write_text("Y1 PHENO\n", encoding="utf-8")

    gwas_manifest_path = run_dir / "gwas_manifest.tsv"
    pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "engine": ["regenie"],
            "status": ["success"],
            "primary_output": [str(ydict_path)],
        }
    ).to_csv(gwas_manifest_path, sep="\t", index=False)

    report_path = build_roundtrip_report(
        run_cfg=cfg,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )

    report = pd.read_csv(report_path, sep="\t")
    assert report.shape[0] == 1
    assert int(report.loc[0, "n_assoc_variants"]) == 2
    assert int(report.loc[0, "n_overlap"]) == 2
    assert int(report.loc[0, "top_k_causal_count"]) == 2


def test_roundtrip_fails_when_nonzero_truth_has_no_overlap(tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_base_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "variant_id": ["chr1:10", "chr1:20", "chr1:10", "chr1:20"],
            "population": ["EUR", "EUR", "AFR", "AFR"],
            "beta": [0.5, -0.4, 0.5, -0.4],
            "dosage_effect_allele": ["A", "G", "A", "G"],
            "dosage_other_allele": ["C", "A", "C", "A"],
            "architecture": ["a1"] * 4,
            "cohort": ["c1"] * 4,
            "replicate": [1] * 4,
        }
    ).to_csv(truth_dir / "effect_sizes.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
        }
    ).to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)
    assoc_path = run_dir / "assoc.tsv"
    pd.DataFrame(
        {"ID": ["chr1:30"], "A1": ["T"], "BETA": [0.1], "P": [0.5]}
    ).to_csv(assoc_path, sep="\t", index=False)
    manifest_path = run_dir / "gwas_manifest.tsv"
    pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "engine": ["plink2"],
            "status": ["success"],
            "primary_output": [str(assoc_path)],
        }
    ).to_csv(manifest_path, sep="\t", index=False)

    with pytest.raises(RuntimeError, match="No harmonized causal overlap"):
        build_roundtrip_report(
            run_cfg=cfg,
            run_dir=run_dir,
            gwas_manifest_path=manifest_path,
        )
