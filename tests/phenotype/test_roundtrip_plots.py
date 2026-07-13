"""Round-trip plot generation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.validation.plots import QQ_AXIS_CAP, _write_qq
from forgephast.phenotype.validation.roundtrip import RoundTripComparison
from forgephast.phenotype.validation import build_roundtrip_plots


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
        "name": "rt_plots_unit",
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
        "roundtrip_validation": {"enabled": True, "top_k": 2},
        "roundtrip_plots": {
            "enabled": True,
            "formats": ["png"],
            "sample_group_file": str(pop_map),
            "id_column": "IID",
            "group_column": "population",
        },
        "paths": {"output_root": str(tmp_path / "out")},
    }


def test_build_roundtrip_plots_writes_default_and_group_outputs(tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_base_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "variant_id": ["chr1:10", "chr1:20", "chr1:10", "chr1:20"],
            "population": ["EUR", "EUR", "AFR", "AFR"],
                "beta": [0.5, -0.4, 0.2, -0.1],
                "dosage_effect_allele": ["A", "G", "A", "G"],
                "dosage_other_allele": ["G", "A", "G", "A"],
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
            "annotation_hit": [True, False],
            "annotation_labels": ["genes", ""],
            "sampling_weight": [5.0, 1.0],
        }
    ).to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)

    assoc_path = run_dir / "assoc.tsv"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "#CHROM": [1, 1, 1],
            "POS": [10, 20, 30],
            "ID": ["rs1", "rs2", "rs3"],
            "A1": ["A", "A", "T"],
            "BETA": [0.45, 0.10, 0.01],
            "P": [1e-9, 2e-8, 0.8],
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

    manifest_path = build_roundtrip_plots(
        run_cfg=cfg,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )

    assert manifest_path.exists()
    manifest = pd.read_csv(manifest_path, sep="\t")
    assert set(manifest["plot_family"]) >= {
        "beta_scatter",
        "manhattan",
        "qq",
        "topk_recovery",
        "enrichment_summary",
        "metrics_by_group",
        "beta_scatter_by_group",
    }
    written = manifest[manifest["status"] == "written"].copy()
    assert not written.empty
    assert written["file_path"].map(Path).map(Path.exists).all()


def test_write_qq_caps_axes_at_default_limit(monkeypatch, tmp_path: Path) -> None:
    assoc = pd.DataFrame(
        {
            "variant_id": [f"chr1:{i}" for i in range(1, 6)],
            "chrom": ["1"] * 5,
            "pos": [1, 2, 3, 4, 5],
            "beta_hat_harmonized": [0.1, 0.2, 0.3, 0.4, 0.5],
            "beta_hat": [0.1, 0.2, 0.3, 0.4, 0.5],
            "beta_true": [0.1, 0.2, 0.3, 0.4, 0.5],
            "p_value": [1e-1, 1e-4, 1e-8, 1e-20, 1e-50],
            "is_causal": [False, False, True, True, True],
            "annotation_hit": [False, False, True, True, True],
        }
    )
    comparison = RoundTripComparison(
        run_name="rt_plots_unit",
        replicate=1,
        architecture="a1",
        cohort="c1",
        engine="plink2",
        assoc_path=tmp_path / "assoc.tsv",
        assoc_df=assoc,
        truth_df=assoc[["variant_id", "beta_true"]].copy(),
        merged_df=assoc.copy(),
        top_df=assoc.nsmallest(2, "p_value").copy(),
        metrics={
            "n_overlap": 5,
            "beta_slope": 1.0,
            "beta_intercept": 0.0,
            "beta_corr": 1.0,
            "sign_concordance": 1.0,
            "top_k": 2,
        },
    )

    captured = {}

    def fake_save(fig, output_base, formats, dpi):  # noqa: ANN001
        captured["xlim"] = fig.axes[0].get_xlim()
        captured["ylim"] = fig.axes[0].get_ylim()
        return [output_base.with_suffix(".png")]

    monkeypatch.setattr("forgephast.phenotype.validation.plots._save_figure", fake_save)
    _write_qq(
        comparison,
        output_base=tmp_path / "qq",
        formats=["png"],
        dpi=150,
    )

    assert captured["xlim"] == (0.0, QQ_AXIS_CAP)
    assert captured["ylim"] == (0.0, QQ_AXIS_CAP)
