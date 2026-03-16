"""HapStore->PGEN manifest export tests with mocked adapter backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.data.pgen_export import export_manifest_pgens


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = []

    def validate_files(self) -> None:  # noqa: D401
        """No-op."""

    def export_to_pgen(
        self,
        sample_ids,
        sample_populations,
        out_prefix,
        chromosomes=None,
        variant_ids=None,
        export_cfg=None,
    ) -> str:
        self.calls.append(
            {
                "sample_ids": list(sample_ids),
                "sample_populations": list(sample_populations),
                "out_prefix": str(out_prefix),
                "chromosomes": list(chromosomes or []),
                "variant_ids": None if variant_ids is None else list(variant_ids),
            }
        )
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pgen", ".pvar", ".psam"):
            Path(f"{out_prefix}{ext}").write_text("", encoding="utf-8")
        return str(out_prefix)


def _payload(tmp_path: Path) -> dict:
    sample_ids = tmp_path / "samples.txt"
    sample_ids.write_text("s1\ns2\n", encoding="utf-8")
    pop_map = tmp_path / "pop.tsv"
    pop_map.write_text("IID\tpopulation\ns1\tEUR\ns2\tAFR\n", encoding="utf-8")
    cov = tmp_path / "cov.tsv"
    cov.write_text("FID\tIID\tage\ns1\ts1\t30\ns2\ts2\t40\n", encoding="utf-8")
    cov_fx = tmp_path / "cov_fx.tsv"
    cov_fx.write_text("covariate\tbeta\nage\t0.1\n", encoding="utf-8")
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(
        "causal_id\ttag_id\tmaf_eur\tmaf_afr\n"
        "chr1:10\tchr1:12\t0.015\t0.020\n"
        "chr1:20\tchr1:22\t0.016\t0.021\n",
        encoding="utf-8",
    )

    return {
        "name": "export_unit",
        "genotype_input": {
            "format": "hapstore",
            "hapstore": {
                "root_dir": str(tmp_path / "hapstores"),
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
        "architectures": [{"name": "a1", "params": {"sigma_beta": 1.0, "h2": 0.2}}],
        "causal_selection": {
            "source_pairs_file": str(pairs),
            "variant_id_column": "causal_id",
            "maf_columns_by_population": {"EUR": "maf_eur", "AFR": "maf_afr"},
            "maf_buckets": [
                {"name": "b1", "maf_min": 0.01, "maf_max": 0.02, "fixed_count": 1, "subset_count": 1}
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
        "pgen_export": {"enabled": True, "variant_scope": "causal_only"},
        "gwas": {"global_pcs": {"mode": "none"}},
        "paths": {"output_root": str(tmp_path / "out")},
        "chromosomes": [1],
    }


def test_export_manifest_pgens_updates_manifest(monkeypatch, tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    membership = pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
        }
    )
    membership.to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)

    out_dir = run_dir / "replicate_1" / "architecture_a1" / "cohort_c1"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "pgen_prefix": [""],
            "output_dir": [str(out_dir)],
        }
    )
    manifest_path = run_dir / "run_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    fake = _FakeAdapter()
    monkeypatch.setattr("forgephast.phenotype.data.pgen_export.build_genotype_adapter", lambda **_: fake)

    out_manifest = export_manifest_pgens(cfg, manifest_path)
    assert out_manifest.exists()
    out = pd.read_csv(out_manifest, sep="\t")
    assert out.loc[0, "pgen_prefix"] != ""
    assert out.loc[0, "exported_pgen_prefix"] == out.loc[0, "pgen_prefix"]
    assert len(fake.calls) == 1
    assert fake.calls[0]["variant_ids"] == ["chr1:10", "chr1:20"]


def test_export_manifest_pgens_reuses_identical_exports_in_memory(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RunConfig.from_dict(_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    membership = pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20", "chr1:10", "chr1:20"],
            "replicate": [1, 1, 2, 2],
            "architecture": ["a1", "a1", "a1", "a1"],
            "cohort": ["c1", "c1", "c1", "c1"],
        }
    )
    membership.to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)

    out_dir1 = run_dir / "replicate_1" / "architecture_a1" / "cohort_c1"
    out_dir2 = run_dir / "replicate_2" / "architecture_a1" / "cohort_c1"
    out_dir1.mkdir(parents=True, exist_ok=True)
    out_dir2.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "run_name": [cfg.name, cfg.name],
            "replicate": [1, 2],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
            "pgen_prefix": ["", ""],
            "output_dir": [str(out_dir1), str(out_dir2)],
        }
    )
    manifest_path = run_dir / "run_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    fake = _FakeAdapter()
    monkeypatch.setattr("forgephast.phenotype.data.pgen_export.build_genotype_adapter", lambda **_: fake)

    runtime_cache: dict[str, object] = {}
    out_manifest = export_manifest_pgens(cfg, manifest_path, runtime_cache=runtime_cache)
    out = pd.read_csv(out_manifest, sep="\t")

    assert len(fake.calls) == 1
    assert out["pgen_prefix"].nunique() == 1
    assert out["exported_pgen_prefix"].nunique() == 1

    export_manifest_pgens(cfg, manifest_path, runtime_cache=runtime_cache)
    assert len(fake.calls) == 1
    assert runtime_cache["cache_hits"]["pgen_export"] == 2
