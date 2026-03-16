from __future__ import annotations

import importlib

import pandas as pd
import pytest

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.qc import run_donor_within_pop_round_robin

from .conftest import write_test_config


def test_donor_round_robin_qc_outputs(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="donor_rr_qc")
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")

    def _fake_plink_fst_for_groups(**kwargs):
        pairs = kwargs["pairs"]
        out = {}
        for idx, (a, b) in enumerate(pairs):
            out[qc_module._pair_key(a, b)] = 0.01 + idx * 0.001
        return out

    monkeypatch.setattr(qc_module, "_plink_fst_for_groups", _fake_plink_fst_for_groups)

    report = run_donor_within_pop_round_robin(
        cfg,
        chrom=21,
        fold_fraction=0.5,
        seed=11,
        workers=2,
    )

    assert report.diagnostics_path.exists()
    assert report.table_path.exists()
    assert report.n_records == 3
    assert set(report.summary_by_population.keys()) == {"P1", "P2", "P3"}

    table = pd.read_csv(report.table_path, sep="\t")
    assert len(table) == 3
    assert set(table["population"].unique()) == {"P1", "P2", "P3"}
    assert set(table["fold_a"].unique()) == {0}
    assert set(table["fold_b"].unique()) == {1}

    for pop, summary in report.summary_by_population.items():
        assert summary["n_donors"] == 2
        assert summary["n_folds"] == 2
        assert summary["n_pairs"] == 1
        assert summary["fold_min_size"] == 1
        assert summary["fold_max_size"] == 1
        assert pop in {"P1", "P2", "P3"}


def test_donor_round_robin_rejects_invalid_fraction(tmp_path):
    cfg_path = write_test_config(tmp_path, name="donor_rr_qc_invalid_fraction")
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    with pytest.raises(ValueError, match="fold_fraction must be in"):
        run_donor_within_pop_round_robin(
            cfg,
            chrom=21,
            fold_fraction=0.01,
            seed=11,
            workers=1,
        )
