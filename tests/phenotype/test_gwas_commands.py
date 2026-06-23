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
    prepare_manifest_covariates_for_gwas,
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
        merged_covariates_file="",
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
    assert result.metadata["monitor_log"].endswith("assoc.log")
    assert "assoc*.glm.*" in result.metadata["monitor_primary_glob"]
    assert "--chr" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--chr") + 1] == "1,3"
    assert "--covar-variance-standardize" in captured["cmd"]
    assert "--1" not in captured["cmd"]


def test_plink2_engine_uses_binary_zero_one_flag(monkeypatch, tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    task = GwasTask(**{**task.__dict__, "trait_type": "binary"})
    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = list(cmd)
        out_idx = cmd.index("--out") + 1
        out_prefix = Path(cmd[out_idx])
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        (out_prefix.parent / "assoc.PHENO1.glm.logistic.hybrid").write_text("# fake\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    engine = Plink2Engine(Plink2EngineConfig(path="plink2", threads=1))
    result = engine.run(task, tmp_path / "run")

    assert result.engine == "plink2"
    assert "--1" in captured["cmd"]
    assert Path(result.primary_output).exists()


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
            Path(f"{out_prefix}.regenie.Ydict").write_text("Y1 PHENO\n")
            Path(f"{out_prefix}_PHENO.regenie").write_text("# fake\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    engine = RegenieEngine(RegenieEngineConfig(path="regenie", threads=1))
    result = engine.run(task, tmp_path / "run")
    assert result.engine == "regenie"
    assert Path(result.primary_output).exists()
    assert result.primary_output.endswith("_PHENO.regenie")
    assert result.metadata["monitor_step1_log"].endswith("step1.log")
    assert result.metadata["monitor_step1_pred_list"].endswith("step1_pred.list")
    assert "step2*.regenie*" in result.metadata["monitor_step2_glob"]
    assert calls["n"] == 2
    assert all("--chrList" in cmd for cmd in commands)
    assert all(cmd[cmd.index("--chrList") + 1] == "1,3" for cmd in commands)
    assert all("--qt" in cmd for cmd in commands)
    assert all("--bt" not in cmd for cmd in commands)


def test_regenie_engine_uses_binary_trait_flag(monkeypatch, tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    task = GwasTask(**{**task.__dict__, "trait_type": "binary"})
    commands = []

    def fake_run(cmd, capture_output, text):
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
    assert len(commands) == 2
    assert all("--bt" in cmd for cmd in commands)
    assert all("--qt" not in cmd for cmd in commands)


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


def test_merge_covariates_for_gwas_allows_ids_only(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.tsv"
    pcs_path = tmp_path / "pcs.tsv"
    out_path = tmp_path / "merged.tsv"
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}).to_csv(
        cov_path, sep="\t", index=False
    )
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}).to_csv(
        pcs_path, sep="\t", index=False
    )

    covars = merge_covariates_for_gwas(str(cov_path), str(pcs_path), out_path)
    out_df = pd.read_csv(out_path, sep="\t")

    assert covars == []
    assert list(out_df.columns) == ["FID", "IID"]


def test_merge_covariates_for_gwas_drops_age_when_age_centered_present(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.tsv"
    pcs_path = tmp_path / "pcs.tsv"
    out_path = tmp_path / "merged.tsv"
    pd.DataFrame(
        {
            "FID": ["s1", "s2"],
            "IID": ["s1", "s2"],
            "age": [30, 40],
            "age_centered": [-20, -10],
            "sex": [0, 1],
        }
    ).to_csv(cov_path, sep="\t", index=False)
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"], "PC1": [0.1, -0.2]}).to_csv(
        pcs_path, sep="\t", index=False
    )

    covars = merge_covariates_for_gwas(str(cov_path), str(pcs_path), out_path)
    out_df = pd.read_csv(out_path, sep="\t")

    assert "age" not in covars
    assert covars == ["age_centered", "sex", "PC1"]
    assert "age" not in out_df.columns


def test_merge_covariates_for_gwas_drops_constant_and_population_code_with_pcs(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.tsv"
    pcs_path = tmp_path / "pcs.tsv"
    out_path = tmp_path / "merged.tsv"
    pd.DataFrame(
        {
            "FID": ["s1", "s2", "s3"],
            "IID": ["s1", "s2", "s3"],
            "age_centered": [0, 1, -1],
            "sex": [0, 1, 0],
            "admix_indicator": [0, 0, 0],
            "population_code": [0, 1, 2],
        }
    ).to_csv(cov_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "FID": ["s1", "s2", "s3"],
            "IID": ["s1", "s2", "s3"],
            "PC1": [0.1, -0.2, 0.3],
            "PC2": [0.4, -0.5, 0.6],
        }
    ).to_csv(pcs_path, sep="\t", index=False)

    covars = merge_covariates_for_gwas(str(cov_path), str(pcs_path), out_path)
    out_df = pd.read_csv(out_path, sep="\t")

    assert "admix_indicator" not in covars
    assert "population_code" not in covars
    assert covars == ["age_centered", "sex", "PC1", "PC2"]
    assert "admix_indicator" not in out_df.columns
    assert "population_code" not in out_df.columns


def test_plink2_engine_omits_covar_flags_when_none(monkeypatch, tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}).to_csv(
        task.covariates_file, sep="\t", index=False
    )
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}).to_csv(
        task.global_pcs_file, sep="\t", index=False
    )
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
    assert "--covar" not in captured["cmd"]
    assert "--covar-name" not in captured["cmd"]
    assert "--covar-variance-standardize" not in captured["cmd"]
    assert "--glm" in captured["cmd"]
    glm_idx = captured["cmd"].index("--glm")
    assert captured["cmd"][glm_idx + 1 : glm_idx + 3] == ["hide-covar", "allow-no-covars"]


def test_regenie_engine_omits_covar_flags_when_none(monkeypatch, tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}).to_csv(
        task.covariates_file, sep="\t", index=False
    )
    pd.DataFrame({"FID": ["s1", "s2"], "IID": ["s1", "s2"]}).to_csv(
        task.global_pcs_file, sep="\t", index=False
    )

    commands = []

    def fake_run(cmd, capture_output, text):
        commands.append(list(cmd))
        out_idx = cmd.index("--out") + 1
        out_prefix = Path(cmd[out_idx])
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        if "--step" in cmd and cmd[cmd.index("--step") + 1] == "1":
            Path(f"{out_prefix}_pred.list").write_text("pred\n")
        else:
            Path(f"{out_prefix}.regenie.Ydict").write_text("Y1 PHENO\n")
            Path(f"{out_prefix}_PHENO.regenie").write_text("# fake\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    engine = RegenieEngine(RegenieEngineConfig(path="regenie", threads=1))
    result = engine.run(task, tmp_path / "run")

    assert result.engine == "regenie"
    assert all("--covarFile" not in cmd for cmd in commands)
    assert all("--covarColList" not in cmd for cmd in commands)


def test_prepare_manifest_covariates_for_gwas_reuses_one_file_per_row(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    manifest = pd.DataFrame(
        {
            "covariates_file": [task.covariates_file],
            "global_pcs_file": [task.global_pcs_file],
            "output_dir": [str(tmp_path / "task_1")],
        }
    )

    prepared = prepare_manifest_covariates_for_gwas(manifest)

    merged_path = Path(prepared.loc[0, "merged_covariates_file"])
    assert merged_path.exists()
    merged = pd.read_csv(merged_path, sep="\t")
    assert list(merged.columns) == ["FID", "IID", "age", "PC1"]


def test_plink2_engine_reuses_prepared_covariates(monkeypatch, tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    merged_cov = tmp_path / "prepared.tsv"
    pd.DataFrame(
        {"FID": ["s1", "s2"], "IID": ["s1", "s2"], "age": [10, 20], "PC1": [0.1, -0.1]}
    ).to_csv(merged_cov, sep="\t", index=False)
    task = GwasTask(**{**task.__dict__, "merged_covariates_file": str(merged_cov)})

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = list(cmd)
        out_idx = cmd.index("--out") + 1
        out_prefix = Path(cmd[out_idx])
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        (out_prefix.parent / "assoc.PHENO1.glm.linear").write_text("# fake\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def fail_merge(*args, **kwargs):  # noqa: ANN001, ANN202
        raise AssertionError("prepared covariate file should have been reused")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("forgephast.phenotype.gwas.engines.merge_covariates_for_gwas", fail_merge)

    engine = Plink2Engine(Plink2EngineConfig(path="plink2", threads=1))
    engine.run(task, tmp_path / "run")

    assert "--covar" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--covar") + 1] == str(merged_cov)
