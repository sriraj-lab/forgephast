"""Global PCA refresh in-memory dedupe tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.data.global_pcs import (
    _global_pcs_cache_key_path,
    refresh_manifest_global_pcs,
)


def _payload(tmp_path: Path) -> dict:
    sample_ids = tmp_path / "samples.txt"
    sample_ids.write_text("s1\ns2\n", encoding="utf-8")
    cov = tmp_path / "cov.tsv"
    cov.write_text("IID\tage\ns1\t30\ns2\t40\n", encoding="utf-8")
    cov_fx = tmp_path / "cov_fx.tsv"
    cov_fx.write_text("covariate\tbeta\nage\t0.1\n", encoding="utf-8")
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text(
        "causal_id\tmaf_eur\tmaf_afr\nchr1:10\t0.02\t0.02\n",
        encoding="utf-8",
    )
    precomputed = tmp_path / "precomputed_pcs.tsv"
    precomputed.write_text(
        "IID\tPC1\tPC2\ns1\t0.1\t0.2\ns2\t-0.1\t0.3\n",
        encoding="utf-8",
    )
    return {
        "name": "pcs_cache",
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
                "population_map_file": "/tmp/pop_map.tsv",
            },
        ],
        "architectures": [{"name": "a1", "params": {"sigma_beta": 1.0, "h2": 0.2}}],
        "causal_selection": {
            "source_pairs_file": str(pairs),
            "variant_id_column": "causal_id",
            "maf_columns_by_population": {"EUR": "maf_eur", "AFR": "maf_afr"},
            "maf_buckets": [
                {
                    "name": "b1",
                    "maf_min": 0.01,
                    "maf_max": 0.03,
                    "fixed_count": 1,
                    "subset_count": 1,
                }
            ],
        },
        "effect_model": {
            "population_order": ["EUR", "AFR"],
            "population_corr_matrix": [[1.0, 0.5], [0.5, 1.0]],
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
        "gwas": {
            "global_pcs": {
                "mode": "precomputed",
                "reuse_cache": True,
                "precomputed_file": str(precomputed),
                "n_components": 2,
            }
        },
        "paths": {"output_root": str(tmp_path / "out")},
        "chromosomes": [1],
    }


def test_refresh_manifest_global_pcs_dedupes_in_memory(monkeypatch, tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    out_dir1 = run_dir / "rep1" / "arch_a1" / "cohort_c1"
    out_dir2 = run_dir / "rep2" / "arch_a1" / "cohort_c1"
    out_dir1.mkdir(parents=True, exist_ok=True)
    out_dir2.mkdir(parents=True, exist_ok=True)

    keep_file = tmp_path / "keep.tsv"
    keep_file.write_text("FID IID\n0 s1\n0 s2\n", encoding="utf-8")

    manifest = pd.DataFrame(
        {
            "run_name": [cfg.name, cfg.name],
            "replicate": [1, 2],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
            "pgen_prefix": ["/tmp/fake", "/tmp/fake"],
            "keep_file": [str(keep_file), str(keep_file)],
            "output_dir": [str(out_dir1), str(out_dir2)],
            "global_pcs_file": ["", ""],
        }
    )
    manifest_path = run_dir / "run_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)

    calls = {"n": 0}

    def fake_write_precomputed_pcs(*, mode_cfg, sample_ids, out_path):  # noqa: ANN001
        del mode_cfg
        calls["n"] += 1
        df = pd.DataFrame(
            {
                "FID": sample_ids,
                "IID": sample_ids,
                "PC1": [0.1, -0.1],
                "PC2": [0.2, 0.3],
            }
        )
        df.to_csv(out_path, sep="\t", index=False)
        return df

    monkeypatch.setattr("forgephast.phenotype.data.global_pcs._write_precomputed_pcs", fake_write_precomputed_pcs)

    runtime_cache: dict[str, object] = {}
    out_manifest = refresh_manifest_global_pcs(cfg, manifest_path, runtime_cache=runtime_cache)
    refreshed = pd.read_csv(out_manifest, sep="\t")

    assert calls["n"] == 1
    assert refreshed["global_pcs_file"].nunique() == 2
    assert refreshed["pca_cache_key"].nunique() == 1
    assert all(Path(path).exists() for path in refreshed["global_pcs_file"].tolist())
    assert runtime_cache["cache_misses"]["global_pcs"] == 1
    assert runtime_cache["cache_hits"]["global_pcs"] == 1


def test_refresh_manifest_global_pcs_reuses_existing_disk_cache(monkeypatch, tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_payload(tmp_path))
    run_dir = Path(cfg.paths.output_root) / cfg.name
    out_dir = run_dir / "rep1" / "arch_a1" / "cohort_c1"
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_file = tmp_path / "keep.tsv"
    keep_file.write_text("FID IID\n0 s1\n0 s2\n", encoding="utf-8")

    pcs_path = out_dir / "global_pcs.tsv"
    pcs_df = pd.DataFrame(
        {
            "FID": ["s1", "s2"],
            "IID": ["s1", "s2"],
            "PC1": [0.1, -0.1],
            "PC2": [0.2, 0.3],
        }
    )
    pcs_df.to_csv(pcs_path, sep="\t", index=False)

    manifest = pd.DataFrame(
        {
            "run_name": [cfg.name],
            "replicate": [1],
            "architecture": ["a1"],
            "cohort": ["c1"],
            "pgen_prefix": ["/tmp/fake"],
            "keep_file": [str(keep_file)],
            "output_dir": [str(out_dir)],
            "global_pcs_file": [str(pcs_path)],
        }
    )
    manifest_path = run_dir / "run_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)

    runtime_cache: dict[str, object] = {}
    first_manifest = refresh_manifest_global_pcs(cfg, manifest_path, runtime_cache=runtime_cache)
    first_df = pd.read_csv(first_manifest, sep="\t")
    key_path = _global_pcs_cache_key_path(pcs_path)
    assert key_path.exists()
    assert first_df.loc[0, "global_pcs_file"] == str(pcs_path)

    def _fail_write(*args, **kwargs):  # noqa: ANN001, ANN202
        raise AssertionError("persistent PCA cache should have been reused")

    monkeypatch.setattr("forgephast.phenotype.data.global_pcs._write_precomputed_pcs", _fail_write)

    runtime_cache_2: dict[str, object] = {}
    second_manifest = refresh_manifest_global_pcs(cfg, manifest_path, runtime_cache=runtime_cache_2)
    second_df = pd.read_csv(second_manifest, sep="\t")

    assert second_df.loc[0, "global_pcs_file"] == str(pcs_path)
    assert second_df.loc[0, "pca_cache_key"]
    assert runtime_cache_2["cache_hits"]["global_pcs"] == 1
    assert runtime_cache_2.get("cache_misses", {}).get("global_pcs", 0) == 0
