"""run_pipeline GWAS parallel scheduling tests."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from forgephast.phenotype.cli import _run_gwas
from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.gwas import GwasResult


def _payload(tmp_path: Path, task_workers: int) -> dict:
    sample_ids = tmp_path / "samples.txt"
    sample_ids.write_text("s1\ns2\n", encoding="utf-8")
    cov = tmp_path / "cov.tsv"
    cov.write_text("IID\tage\ns1\t30\ns2\t40\n", encoding="utf-8")
    cov_fx = tmp_path / "cov_fx.tsv"
    cov_fx.write_text("covariate\tbeta\nage\t0.1\n", encoding="utf-8")
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("causal_id\tmaf_eur\tmaf_afr\nchr1:10\t0.02\t0.02\n", encoding="utf-8")
    return {
        "name": "gwas_parallel_unit",
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
            "task_workers": task_workers,
            "engines": ["plink2", "regenie"],
            "global_pcs": {"mode": "none"},
        },
        "paths": {"output_root": str(tmp_path / "out")},
        "chromosomes": [1],
    }


def _write_manifest(cfg: RunConfig) -> Path:
    run_dir = Path(cfg.paths.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    keep = run_dir / "keep.tsv"
    pheno = run_dir / "pheno.tsv"
    cov = run_dir / "cov.tsv"
    pcs = run_dir / "pcs.tsv"
    pgen_prefix = run_dir / "fake_pgen"
    keep.write_text("FID IID\n0 s1\n0 s2\n", encoding="utf-8")
    pheno.write_text("FID\tIID\tPHENO\n0\ts1\t1\n0\ts2\t2\n", encoding="utf-8")
    cov.write_text("FID\tIID\tage\n0\ts1\t30\n0\ts2\t40\n", encoding="utf-8")
    pcs.write_text("FID\tIID\tPC1\n0\ts1\t0.1\n0\ts2\t-0.1\n", encoding="utf-8")
    Path(f"{pgen_prefix}.psam").write_text("#FID\tIID\n0\ts1\n0\ts2\n", encoding="utf-8")

    manifest = pd.DataFrame(
        {
            "run_name": [cfg.name, cfg.name],
            "replicate": [1, 2],
            "architecture": ["a1", "a1"],
            "cohort": ["c1", "c1"],
            "pgen_prefix": [str(pgen_prefix), str(pgen_prefix)],
            "keep_file": [str(keep), str(keep)],
            "phenotype_file": [str(pheno), str(pheno)],
            "covariates_file": [str(cov), str(cov)],
            "global_pcs_file": [str(pcs), str(pcs)],
            "output_dir": [str(run_dir / "task_1"), str(run_dir / "task_2")],
        }
    )
    manifest_path = run_dir / "run_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


class _SlowEngine:
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.prepared = None

    def prepare_batch(self, tasks, batch_dir):  # noqa: ANN001
        self.prepared = {
            "n_tasks": len(tasks),
            "batch_dir": str(batch_dir),
            "engine": self.engine_name,
        }
        return self.prepared

    def run(self, task, run_dir, prepared_state=None):  # noqa: ANN001
        delay = 0.03 if (self.engine_name == "plink2" and task.replicate == 1) else 0.0
        time.sleep(delay)
        run_dir.mkdir(parents=True, exist_ok=True)
        primary = run_dir / f"{self.engine_name}_rep{task.replicate}.txt"
        primary.write_text("ok\n", encoding="utf-8")
        return GwasResult(
            engine=self.engine_name,
            output_dir=str(run_dir),
            primary_output=str(primary),
            metadata={"prepared_engine": str((prepared_state or {}).get("engine", ""))},
        )


class _FlakyEngine(_SlowEngine):
    def run(self, task, run_dir, prepared_state=None):  # noqa: ANN001
        if self.engine_name == "regenie" and task.replicate == 2:
            raise RuntimeError("intentional failure")
        return super().run(task, run_dir, prepared_state=prepared_state)


def test_run_gwas_parallel_preserves_manifest_order(monkeypatch, tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_payload(tmp_path, task_workers=4))
    manifest_path = _write_manifest(cfg)
    monkeypatch.setattr(
        "forgephast.phenotype.cli._make_engines",
        lambda _cfg: {"plink2": _SlowEngine("plink2"), "regenie": _SlowEngine("regenie")},
    )

    out_path, _ = _run_gwas(cfg, manifest_path, engines_filter=None)
    out = pd.read_csv(out_path, sep="\t")
    assert out["status"].tolist() == ["success", "success", "success", "success"]
    assert list(zip(out["replicate"].tolist(), out["engine"].tolist())) == [
        (1, "plink2"),
        (1, "regenie"),
        (2, "plink2"),
        (2, "regenie"),
    ]
    assert set(out["prepared_engine"].tolist()) == {"plink2", "regenie"}


def test_run_gwas_parallel_records_failures(monkeypatch, tmp_path: Path) -> None:
    cfg = RunConfig.from_dict(_payload(tmp_path, task_workers=4))
    manifest_path = _write_manifest(cfg)
    monkeypatch.setattr(
        "forgephast.phenotype.cli._make_engines",
        lambda _cfg: {"plink2": _FlakyEngine("plink2"), "regenie": _FlakyEngine("regenie")},
    )

    with pytest.raises(RuntimeError, match="failed task"):
        _run_gwas(cfg, manifest_path, engines_filter=None)

    summary_path = Path(cfg.paths.output_root) / cfg.name / "gwas_manifest.tsv"
    out = pd.read_csv(summary_path, sep="\t")
    assert out["status"].tolist().count("failed") == 1
    failed = out[out["status"] == "failed"].iloc[0]
    assert failed["engine"] == "regenie"
    assert failed["replicate"] == 2
    assert "intentional failure" in str(failed["error"])
