"""Dosage export parsing tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from forgephast.phenotype.data.genotypes import CanonicalPgenAdapter


def _make_dummy_pgen_triplet(prefix: Path) -> None:
    Path(f"{prefix}.pgen").write_text("", encoding="utf-8")
    Path(f"{prefix}.pvar").write_text(
        "#CHROM POS ID REF ALT\n"
        "1 100 v1 A G\n"
        "1 200 v2 C T\n",
        encoding="utf-8",
    )
    Path(f"{prefix}.psam").write_text(
        "FID IID\n"
        "s1 s1\n"
        "s2 s2\n",
        encoding="utf-8",
    )


def test_export_dosages_tracks_effect_allele_map(monkeypatch, tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    _make_dummy_pgen_triplet(prefix)
    adapter = CanonicalPgenAdapter(str(prefix), plink2_path="plink2")

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = list(cmd)
        out_idx = cmd.index("--out") + 1
        out_prefix = Path(cmd[out_idx])
        raw = Path(f"{out_prefix}.raw")
        df = pd.DataFrame(
            {
                "FID": ["s1", "s2"],
                "IID": ["s1", "s2"],
                "PAT": [0, 0],
                "MAT": [0, 0],
                "SEX": [0, 0],
                "PHENOTYPE": [-9, -9],
                "v1_G": [0.0, 1.0],
                "v2_T": [2.0, 1.0],
            }
        )
        df.to_csv(raw, sep="\t", index=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    dosage = adapter.export_dosages(
        sample_ids=["s1", "s2"],
        sample_populations=["EUR", "EUR"],
        variant_ids=["v1", "v2"],
        out_dir=tmp_path / "export",
        chromosomes=[1],
    )

    assert list(dosage.columns) == ["FID", "IID", "v1", "v2"]
    assert dosage["v1"].tolist() == [0.0, 1.0]
    assert dosage["v2"].tolist() == [2.0, 1.0]
    assert adapter.last_export_allele_by_variant == {"v1": "G", "v2": "T"}
    assert "--chr" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--chr") + 1] == "1"
    assert "--export-allele" in captured["cmd"]
    export_allele_path = Path(captured["cmd"][captured["cmd"].index("--export-allele") + 1])
    assert export_allele_path.exists()
    assert export_allele_path.read_text(encoding="utf-8").strip().splitlines() == [
        "v1\tG",
        "v2\tT",
    ]
