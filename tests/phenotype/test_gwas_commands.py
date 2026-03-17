"""GWAS wrapper command/output tests using mocked subprocess calls."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from forgephast.phenotype.config import Plink2EngineConfig, RegenieEngineConfig
from forgephast.phenotype.gwas import (
    GwasResult,
    GwasTask,
    Plink2Engine,
    RegenieEngine,
    merge_covariates_for_gwas,
)


def _make_task(tmp_path: Path) -> GwasTask:
    cov = pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"], "age": [10, 20]})
    pcs = pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"], "PC1": [0.1, -0.1]})
    pheno = pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"], "PHENO": [1.0, 2.0]})
    keep = pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]})

    cov_path = tmp_path / "cov.tsv"
    pcs_path = tmp_path / "pcs.tsv"
    pheno_path = tmp_path / "pheno.tsv"
    keep_path = tmp_path / "keep.tsv"

    cov.to_csv(cov_path, sep="\t", index=False)
    pcs.to_csv(pcs_path, sep="\t", index=False)
    pheno.to_csv(pheno_path, sep="\t", index=False)
    keep.to_csv(keep_path, sep="\t", index=False)

    return GwasTask(
        run_name="r",
        replicate=1,
        architecture="a",
        cohort="c",
        pgen_prefix="/tmp/fake_pgen_prefix",
        keep_file=str(keep_path),
        phenotype_file=str(pheno_path),
        covariates_file=str(cov_path),
        global_pcs_file=str(pcs_path),
        output_dir=str(tmp_path),
        chromosomes=[1, 3],
    )


def test_plink2_engine_outputs(monkeypatch, tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = list(cmd)
        out_idx = cmd.index("--out") + 1
        out_prefix = Path(cmd[out_idx])
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        (out_prefix.parent / "assoc.PHENO1.glm.linear").write_text("# fake\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    engine = Plink2Engine(Plink2EngineConfig(path="plink2", threads=1))
    result = engine.run(task, tmp_path / "run")
    assert result.engine == "plink2"
    assert Path(result.primary_output).exists()
    assert "--chr" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--chr") + 1] == "1,3"


def test_regenie_engine_outputs(monkeypatch, tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    calls = {"n": 0}
    commands = []

    def fake_run(cmd, capture_output, text):
        calls["n"] += 1
        commands.append(list(cmd))
        out_idx = cmd.index("--out") + 1
        out_prefix = Path(cmd[out_idx])
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        if "--step" in cmd and cmd[cmd.index("--step") + 1] == "1":
            Path(f"{out_prefix}_pred.list").write_text("pred\n")
        else:
            Path(f"{out_prefix}_PHENO.regenie").write_text("# fake\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    engine = RegenieEngine(RegenieEngineConfig(path="regenie", threads=1))
    result = engine.run(task, tmp_path / "run")
    assert result.engine == "regenie"
    assert Path(result.primary_output).exists()
    assert calls["n"] == 2
    assert all("--chrList" in cmd for cmd in commands)
    assert all(cmd[cmd.index("--chrList") + 1] == "1,3" for cmd in commands)


def test_merge_covariates_for_gwas_coerces_id_types(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.tsv"
    pcs_path = tmp_path / "pcs.tsv"
    out_path = tmp_path / "merged.tsv"
    pd.DataFrame({"FID": [0, 0], "IID": ["s1", "s2"], "age": [30, 40]}).to_csv(
        cov_path, sep="\t", index=False
    )
    pd.DataFrame({"FID": ["0", "0"], "IID": ["s1", "s2"], "PC1": [0.1, -0.2]}).to_csv(
        pcs_path, sep="\t", index=False
    )

    covars = merge_covariates_for_gwas(str(cov_path), str(pcs_path), out_path)
    out_df = pd.read_csv(out_path, sep="\t")

    assert covars == ["age", "PC1"]
    assert out_df.shape[0] == 2


def test_merge_covariates_for_gwas_falls_back_to_iid_when_fid_differs(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.tsv"
    pcs_path = tmp_path / "pcs.tsv"
    out_path = tmp_path / "merged.tsv"
    pd.DataFrame({"FID": ["0", "0"], "IID": ["s1", "s2"], "age": [30, 40]}).to_csv(
        cov_path, sep="\t", index=False
    )
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"], "PC1": [0.1, -0.2]}).to_csv(
        pcs_path, sep="\t", index=False
    )

    covars = merge_covariates_for_gwas(str(cov_path), str(pcs_path), out_path)
    out_df = pd.read_csv(out_path, sep="\t")

    assert covars == ["age", "PC1"]
    assert out_df["FID"].tolist() == [0, 0] or out_df["FID"].tolist() == ["0", "0"]
    assert out_df.shape[0] == 2
