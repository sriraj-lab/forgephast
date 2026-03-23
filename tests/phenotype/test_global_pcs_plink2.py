"""Global PCA computation tests using PLINK2 command mocks."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

from forgephast.phenotype.data.global_pcs import (
    _run_plink2_pca,
    _read_plink2_eigenvec,
    _write_plink2_keep_file,
    normalize_psam_fid_column,
)


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


def test_normalize_psam_fid_column_adds_fid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        psam = Path(tmp) / "x.psam"
        psam.write_text(
            "#IID\tSEX\tPHENO\ns1\tNA\tNA\ns2\tNA\t1\n",
            encoding="utf-8",
        )
        normalize_psam_fid_column(psam)
        lines = psam.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "#FID\tIID\tSEX\tPHENO"
        assert lines[1] == "s1\ts1\tNA\tNA"
        assert lines[2] == "s2\ts2\tNA\t1"


def test_write_plink2_keep_file_matches_normalized_psam() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        psam = Path(tmp) / "x.psam"
        keep = Path(tmp) / "keep.txt"
        psam.write_text(
            "#IID\tSEX\tPHENO\ns1\tNA\tNA\ns2\tNA\tNA\n",
            encoding="utf-8",
        )
        _write_plink2_keep_file(keep, ["s1", "s2"], psam_path=psam)
        assert keep.read_text(encoding="utf-8").splitlines() == ["s1 s1", "s2 s2"]
        assert psam.read_text(encoding="utf-8").splitlines()[0] == "#FID\tIID\tSEX\tPHENO"


def test_run_plink2_pca_prunes_before_pca(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        psam = root / "cohort.psam"
        psam.write_text("#FID\tIID\ns1\ts1\ns2\ts2\n", encoding="utf-8")
        calls = []

        def fake_run(cmd, capture_output, text):  # noqa: ANN001
            calls.append(list(cmd))
            out_idx = cmd.index("--out") + 1
            out_prefix = Path(cmd[out_idx])
            if "--indep-pairwise" in cmd:
                Path(f"{out_prefix}.prune.in").write_text("chr1:10\n", encoding="utf-8")
            else:
                Path(f"{out_prefix}.eigenvec").write_text("s1 s1 0.5\ns2 s2 -0.5\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)
        pcs = _run_plink2_pca(
            pgen_prefix=str(root / "cohort"),
            plink2_path="plink2",
            sample_ids=["s1", "s2"],
            n_components=1,
            out_dir=root / "pca",
            mode_cfg=SimpleNamespace(
                plink2_approx=True,
                plink2_extra_args=[],
                prune_before_pca=True,
                prune_window_kb=1000,
                prune_step_variants=50,
                prune_r2=0.1,
            ),
            memory_mb=4096,
            chromosomes=[1, 8, 21],
        )

        assert list(pcs.columns) == ["FID", "IID", "PC1"]
        assert len(calls) == 2
        assert "--indep-pairwise" in calls[0]
        assert "--extract" in calls[1]
        assert "--memory" in calls[0]
        assert "--memory" in calls[1]
