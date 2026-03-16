"""Global PCA computation tests using PLINK2 command mocks."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from forgephast.phenotype.config import GlobalPCConfig
from forgephast.phenotype.data.genotypes import CanonicalPgenAdapter, _read_plink2_eigenvec


def _make_dummy_pgen_triplet(prefix: Path) -> None:
    for ext in (".pgen", ".pvar"):
        (Path(f"{prefix}{ext}")).write_text("", encoding="utf-8")
    Path(f"{prefix}.psam").write_text(
        "#IID\n"
        "s1\n"
        "s2\n",
        encoding="utf-8",
    )


def test_compute_global_pcs_runs_plink2_and_aligns(monkeypatch, tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    _make_dummy_pgen_triplet(prefix)
    adapter = CanonicalPgenAdapter(str(prefix), plink2_path="plink2")

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = list(cmd)
        out_idx = cmd.index("--out") + 1
        out_prefix = Path(cmd[out_idx])
        Path(f"{out_prefix}.eigenvec").write_text(
            "#FID IID PC1 PC2\n"
            "s1 s1 0.10 0.20\n"
            "s2 s2 -0.30 0.40\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    pcs_df = adapter.compute_global_pcs(
        sample_ids=["s2", "s1"],
        sample_populations=["EUR", "EUR"],
        n_components=3,
        out_dir=tmp_path / "pca",
        mode_cfg=GlobalPCConfig(
            mode="plink2_pca",
            n_components=3,
            plink2_approx=True,
            plink2_extra_args=["--freq"],
        ),
        chromosomes=[1, 2],
    )

    cmd = captured["cmd"]
    assert cmd[:2] == ["plink2", "--pfile"]
    assert "--keep" in cmd
    assert "--chr" in cmd
    assert cmd[cmd.index("--chr") + 1] == "1,2"
    assert "--pca" in cmd
    assert "3" in cmd
    assert "approx" in cmd
    assert "--freq" in cmd

    assert list(pcs_df.columns) == ["FID", "IID", "PC1", "PC2", "PC3"]
    assert pcs_df["IID"].tolist() == ["s2", "s1"]
    assert pcs_df["FID"].tolist() == ["s2", "s1"]
    assert float(pcs_df.loc[pcs_df["IID"] == "s2", "PC1"].iloc[0]) == -0.30
    assert pcs_df["PC3"].eq(0.0).all()


def test_read_plink2_eigenvec_no_header() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        eigenvec = Path(tmp) / "x.eigenvec"
        eigenvec.write_text("s1 s1 0.5\ns2 s2 -0.5\n", encoding="utf-8")
        pcs_df = _read_plink2_eigenvec(eigenvec, n_components=2)
        assert list(pcs_df.columns) == ["FID", "IID", "PC1", "PC2"]
        assert pcs_df["PC1"].tolist() == [0.5, -0.5]
        assert pcs_df["PC2"].eq(0.0).all()


def test_read_plink2_eigenvec_header_iid_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        eigenvec = Path(tmp) / "x.eigenvec"
        eigenvec.write_text("IID PC1 PC2\ns1 0.10 0.20\ns2 -0.10 0.30\n", encoding="utf-8")
        pcs_df = _read_plink2_eigenvec(eigenvec, n_components=2)
        assert list(pcs_df.columns) == ["FID", "IID", "PC1", "PC2"]
        assert pcs_df["IID"].tolist() == ["s1", "s2"]
        assert pcs_df["FID"].tolist() == ["s1", "s2"]
        assert pcs_df["PC2"].tolist() == [0.20, 0.30]


def test_compute_global_pcs_plink2_failure_raises(monkeypatch, tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    _make_dummy_pgen_triplet(prefix)
    adapter = CanonicalPgenAdapter(str(prefix), plink2_path="plink2")

    def fake_run(cmd, capture_output, text):
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="PLINK2 PCA failed"):
        adapter.compute_global_pcs(
            sample_ids=["s1", "s2"],
            sample_populations=["EUR", "EUR"],
            n_components=2,
            out_dir=tmp_path / "pca",
            mode_cfg=GlobalPCConfig(mode="plink2_pca", n_components=2),
        )
