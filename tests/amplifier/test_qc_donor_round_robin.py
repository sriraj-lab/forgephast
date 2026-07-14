from __future__ import annotations

import importlib

import numpy as np
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

    data_support = importlib.import_module("forgephast.amplifier.qc.data_support")
    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    call_counter = {"count": 0}

    def _fake_resolve_source(*args, **kwargs):
        return "vcf", tmp_path / "dummy.vcf.gz"

    def _fake_load_panel(*, config, source_path, chrom, sample_ids, source_kind):
        del config, source_path, chrom, source_kind
        n_samples = len(sample_ids)
        positions = np.asarray([101, 202, 303], dtype=np.int64)
        haplotypes = np.zeros((2 * n_samples, positions.size), dtype=np.uint8)
        return positions, haplotypes

    def _fake_matrix_hudson(ac_a, an_a, ac_b, an_b, *, numba_threads=None):
        del ac_a, an_a, ac_b, an_b, numba_threads
        value = 0.01 + (0.001 * call_counter["count"])
        call_counter["count"] += 1
        return value

    monkeypatch.setattr(data_support, "resolve_donor_panel_source", _fake_resolve_source)
    monkeypatch.setattr(data_support, "load_sample_haplotypes_from_panel", _fake_load_panel)
    monkeypatch.setattr(qc_module, "matrix_hudson_fst_from_counts", _fake_matrix_hudson)

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
    assert set(table["fst"].round(6).tolist()) == {0.01, 0.011, 0.012}

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
