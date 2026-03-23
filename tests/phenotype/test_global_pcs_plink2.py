"""Global PCA computation tests using PLINK2 command mocks."""

from __future__ import annotations

from pathlib import Path
import tempfile

from forgephast.phenotype.data.global_pcs import (
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
