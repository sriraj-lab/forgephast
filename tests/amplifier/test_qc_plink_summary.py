from __future__ import annotations

import importlib
from pathlib import Path

from forgephast.amplifier.qc.pipeline import _load_plink_fst_summary


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_plink_fst_summary_hudson(tmp_path):
    summary = tmp_path / "test.fst.summary"
    _write(
        summary,
        "#POP1\tPOP2\tHUDSON_FST\n"
        "P1\tP2\t0.123\n"
        "P1\tP3\t0.045\n",
    )
    parsed = _load_plink_fst_summary(summary)
    assert parsed["P1__P2"] == 0.123
    assert parsed["P1__P3"] == 0.045


def test_load_plink_fst_summary_wc_fallback(tmp_path):
    summary = tmp_path / "test_wc.fst.summary"
    _write(
        summary,
        "POP1 POP2 WC_FST\n"
        "A B 0.250\n",
    )
    parsed = _load_plink_fst_summary(summary)
    assert parsed["A__B"] == 0.25


def test_plink_fst_command_places_pair_file_with_fst_args(monkeypatch, tmp_path):
    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    captured = {"cmd": None}

    def _fake_run_cmd(cmd):
        captured["cmd"] = cmd

    def _fake_load(_path):
        return {"A__B": 0.1}

    monkeypatch.setattr(qc_module, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(qc_module, "_load_plink_fst_summary", _fake_load)

    out = qc_module._plink_fst_for_groups(
        vcf_path=tmp_path / "input.vcf.gz",
        out_prefix=tmp_path / "out",
        plink2_path="plink2",
        threads=4,
        sample_to_group={"s1": "A", "s2": "B"},
        pairs=[("A", "B")],
        keep_samples=["s1", "s2"],
    )
    assert out["A__B"] == 0.1
    cmd = captured["cmd"]
    assert cmd is not None
    file_token = next(x for x in cmd if str(x).startswith("file="))
    assert cmd.index(file_token) < cmd.index("--keep")


def test_plink_fst_summary_path_handles_dotted_out_prefix(monkeypatch, tmp_path):
    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    captured = {"summary_path": None}

    def _fake_run_cmd(_cmd):
        return None

    def _fake_load(path):
        captured["summary_path"] = Path(path)
        return {"A__B": 0.1}

    monkeypatch.setattr(qc_module, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(qc_module, "_load_plink_fst_summary", _fake_load)

    out_prefix = tmp_path / "out.with.dot"
    out = qc_module._plink_fst_for_groups(
        vcf_path=tmp_path / "input.vcf.gz",
        out_prefix=out_prefix,
        plink2_path="plink2",
        threads=1,
        sample_to_group={"s1": "A", "s2": "B"},
        pairs=[("A", "B")],
        keep_samples=["s1", "s2"],
    )
    assert out["A__B"] == 0.1
    assert captured["summary_path"] == Path(f"{out_prefix}.fst.summary")


def test_plink_ld_loader_path_handles_dotted_out_prefix(monkeypatch, tmp_path):
    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")

    def _fake_run_cmd(_cmd):
        ld_path = Path(f"{out_prefix}.ld")
        _write(
            ld_path,
            "BP_A BP_B R2\n"
            "100 200 0.5\n",
        )

    monkeypatch.setattr(qc_module, "_run_cmd", _fake_run_cmd)

    out_prefix = tmp_path / "ld.with.dot"
    extract = tmp_path / "extract.txt"
    _write(extract, "rs1\n")
    df = qc_module._plink_ld_pairs_for_group(
        vcf_path=tmp_path / "input.vcf.gz",
        out_prefix=out_prefix,
        plink1_path="plink",
        threads=1,
        extract_ids_path=extract,
        keep_samples=["s1", "s2"],
        min_maf=0.01,
        max_distance_cm=1.0,
    )
    assert len(df) == 1
    assert {"BP_A", "BP_B", "R2"}.issubset(df.columns)
