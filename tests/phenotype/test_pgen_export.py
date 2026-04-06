"""HapStore->PGEN manifest export tests with mocked adapter backend."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.data.pgen_export import (
    _normalize_fid_for_exported_pgen,
    ensure_manifest_non_monomorphic_pgens,
    export_manifest_pgens,
)


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
        Path(f"{out_prefix}.pgen").write_text("", encoding="utf-8")
        Path(f"{out_prefix}.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n", encoding="utf-8")
        Path(f"{out_prefix}.psam").write_text("#FID\tIID\ns1\ts1\ns2\ts2\n", encoding="utf-8")
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
    def _fake_filter(prefix: str, *, plink2_path: str, threads: int, memory_mb=None, keep_file=None) -> str:  # noqa: ANN001
        out_prefix = Path(f"{prefix}.maf_gt0")
        Path(f"{out_prefix}.pgen").write_text("", encoding="utf-8")
        Path(f"{out_prefix}.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n", encoding="utf-8")
        Path(f"{out_prefix}.psam").write_text("#FID\tIID\ns1\ts1\ns2\ts2\n", encoding="utf-8")
        return str(out_prefix)
    monkeypatch.setattr("forgephast.phenotype.data.pgen_export._filter_zero_maf_pgen", _fake_filter)

    out_manifest = export_manifest_pgens(cfg, manifest_path)
    assert out_manifest.exists()
    out = pd.read_csv(out_manifest, sep="\t")
    assert out.loc[0, "pgen_prefix"].endswith("rep1_c1")
    assert out.loc[0, "exported_pgen_prefix"] == out.loc[0, "pgen_prefix"]
    assert out.loc[0, "merged_pgen_prefix"] == out.loc[0, "pgen_prefix"]
    chrom_manifest = json.loads(out.loc[0, "chrom_pgen_manifest"])
    assert chrom_manifest == {"1": str(run_dir / "exports" / "pgen" / "chrom" / "rep1_c1.chr1.maf_gt0")}
    assert Path(f"{out.loc[0, 'pgen_prefix']}.pgen").exists()
    assert len(fake.calls) == 1
    assert fake.calls[0]["variant_ids"] == ["chr1:10", "chr1:20"]
    assert fake.calls[0]["chromosomes"] == [1]


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
    def _fake_filter(prefix: str, *, plink2_path: str, threads: int, memory_mb=None, keep_file=None) -> str:  # noqa: ANN001
        out_prefix = Path(f"{prefix}.maf_gt0")
        Path(f"{out_prefix}.pgen").write_text("", encoding="utf-8")
        Path(f"{out_prefix}.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n", encoding="utf-8")
        Path(f"{out_prefix}.psam").write_text("#FID\tIID\ns1\ts1\ns2\ts2\n", encoding="utf-8")
        return str(out_prefix)
    monkeypatch.setattr("forgephast.phenotype.data.pgen_export._filter_zero_maf_pgen", _fake_filter)

    runtime_cache: dict[str, object] = {}
    out_manifest = export_manifest_pgens(cfg, manifest_path, runtime_cache=runtime_cache)
    out = pd.read_csv(out_manifest, sep="\t")

    assert len(fake.calls) == 1
    assert out["pgen_prefix"].nunique() == 1
    assert out["exported_pgen_prefix"].nunique() == 1
    assert out["pgen_prefix"].iloc[0].endswith("rep1_c1")
    assert out["chrom_pgen_manifest"].nunique() == 1

    export_manifest_pgens(cfg, manifest_path, runtime_cache=runtime_cache)
    assert len(fake.calls) == 1
    assert runtime_cache["cache_hits"]["pgen_export"] == 2


def test_ensure_manifest_non_monomorphic_pgens_is_keep_specific(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RunConfig.from_dict(_payload(tmp_path))
    pgen_prefix = tmp_path / "shared" / "rep1_c1"
    pgen_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pgen", ".pvar", ".psam"):
        Path(f"{pgen_prefix}{ext}").write_text("", encoding="utf-8")
    keep1 = tmp_path / "keep1.tsv"
    keep2 = tmp_path / "keep2.tsv"
    keep1.write_text("FID\tIID\ns1\ts1\n", encoding="utf-8")
    keep2.write_text("FID\tIID\ns2\ts2\n", encoding="utf-8")

    manifest = pd.DataFrame(
        {
            "run_name": [cfg.name, cfg.name],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c2"],
            "pgen_prefix": [str(pgen_prefix), str(pgen_prefix)],
            "keep_file": [str(keep1), str(keep2)],
            "global_pcs_file": ["foo.tsv", "bar.tsv"],
            "merged_covariates_file": ["a.tsv", "b.tsv"],
        }
    )

    calls: list[tuple[str, str | None]] = []

    def _fake_filter(prefix: str, *, plink2_path: str, threads: int, memory_mb=None, keep_file=None) -> str:  # noqa: ANN001
        calls.append((str(prefix), None if keep_file is None else str(keep_file)))
        suffix = "k1" if str(keep_file) == str(keep1) else "k2"
        out_prefix = Path(f"{prefix}.maf_gt0.{suffix}")
        for ext in (".pgen", ".pvar", ".psam"):
            Path(f"{out_prefix}{ext}").write_text("", encoding="utf-8")
        return str(out_prefix)

    monkeypatch.setattr("forgephast.phenotype.data.pgen_export._filter_zero_maf_pgen", _fake_filter)

    out = ensure_manifest_non_monomorphic_pgens(cfg, manifest)

    assert calls == [(str(pgen_prefix), str(keep1)), (str(pgen_prefix), str(keep2))]
    assert out.loc[0, "pgen_prefix"].endswith(".maf_gt0.k1")
    assert out.loc[1, "pgen_prefix"].endswith(".maf_gt0.k2")
    assert out.loc[0, "global_pcs_file"] == ""
    assert out.loc[1, "global_pcs_file"] == ""
    assert out.loc[0, "merged_covariates_file"] == ""
    assert out.loc[1, "merged_covariates_file"] == ""


def test_export_manifest_pgens_legacy_path_still_filters_merged_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _payload(tmp_path)
    payload["pgen_export"]["prep_mode"] = "legacy_whole_store"
    cfg = RunConfig.from_dict(payload)
    run_dir = Path(cfg.paths.output_root) / cfg.name
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "replicate": [1, 1],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
        }
    ).to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)

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

    def _fake_filter(prefix: str, *, plink2_path: str, threads: int, memory_mb=None, keep_file=None) -> str:  # noqa: ANN001
        out_prefix = Path(f"{prefix}.maf_gt0")
        Path(f"{out_prefix}.pgen").write_text("", encoding="utf-8")
        Path(f"{out_prefix}.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n", encoding="utf-8")
        Path(f"{out_prefix}.psam").write_text("#FID\tIID\ns1\ts1\ns2\ts2\n", encoding="utf-8")
        return str(out_prefix)

    monkeypatch.setattr("forgephast.phenotype.data.pgen_export._filter_zero_maf_pgen", _fake_filter)

    out_manifest = export_manifest_pgens(cfg, manifest_path)
    out = pd.read_csv(out_manifest, sep="\t")

    assert out.loc[0, "pgen_prefix"].endswith(".maf_gt0")
    assert out.loc[0, "exported_pgen_prefix"] != out.loc[0, "pgen_prefix"]
    assert out.loc[0, "chrom_pgen_manifest"] == "{}"


def test_normalize_fid_for_exported_pgen_sets_fid_to_iid(tmp_path: Path) -> None:
    table = tmp_path / "phenotype.tsv"
    table.write_text(
        "FID\tIID\tPHENO\n0\ts1\t1.2\n0\ts2\t-0.4\n",
        encoding="utf-8",
    )
    _normalize_fid_for_exported_pgen(str(table))
    out = pd.read_csv(table, sep="\t")
    assert out["FID"].astype(str).tolist() == ["s1", "s2"]
    assert out["IID"].astype(str).tolist() == ["s1", "s2"]
