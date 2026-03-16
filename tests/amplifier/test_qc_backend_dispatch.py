from __future__ import annotations

import importlib
import json

import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome

from .conftest import write_test_config


def test_qc_backend_plink_counts_dispatch(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_backend_dispatch")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    payload["qc"]["backend"] = "plink_counts"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)

    called = {"count": 0}

    def _fake_plink_backend(**kwargs):
        called["count"] += 1
        return (
            {"P1__P2": 0.11, "P1__P3": 0.07, "P2__P3": 0.05},
            {"P1": -0.001, "P2": -0.002, "P3": -0.003},
        )

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    monkeypatch.setattr(qc_module, "_run_qc_plink_counts", _fake_plink_backend)
    report = run_qc(cfg, chrom=21, n_shards=2)
    assert called["count"] == 1
    assert report.pairwise_fst["P1__P2"] == 0.11
    assert report.within_split_fst["P3"] == -0.003

    with open(report.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["backend"] == "plink_counts"


def test_qc_backend_plink_native_fst_dispatch(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_backend_dispatch_native")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    payload["qc"]["backend"] = "plink_native_fst"
    payload["qc"]["within_pop_split_checks"] = []
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)

    called = {"count": 0}

    def _fake_plink_native_backend(**kwargs):
        called["count"] += 1
        assert kwargs["within_checks"] == []
        return (
            {"P1__P2": 0.21, "P1__P3": 0.17, "P2__P3": 0.15},
            {"P1": -0.011, "P2": -0.012, "P3": -0.013},
        )

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    monkeypatch.setattr(qc_module, "_run_qc_plink_native_fst", _fake_plink_native_backend)
    report = run_qc(cfg, chrom=21, n_shards=2)
    assert called["count"] == 1
    assert report.pairwise_fst["P1__P2"] == 0.21
    assert report.within_split_fst["P3"] == -0.013

    with open(report.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["backend"] == "plink_native_fst"


def test_qc_ld_decay_gate_updates_overall_gate(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_ld_decay_gate")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    payload["qc"]["backend"] = "plink_native_fst"
    payload["qc"]["within_pop_split_checks"] = []
    payload["qc"]["ld_decay_enabled"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")

    def _fake_plink_native_backend(**kwargs):
        return (
            {"P1__P2": 0.01, "P1__P3": 0.01, "P2__P3": 0.01},
            {},
        )

    def _fake_ld_decay(**kwargs):
        ld_json = tmp_path / "ld_decay_fake.json"
        return (
            {
                "enabled": True,
                "status": "ok",
                "gate_pass": False,
                "table_path": str(tmp_path / "ld_decay_fake.tsv"),
                "population_summary": {"P1": {"pass": False}},
            },
            ld_json,
        )

    monkeypatch.setattr(qc_module, "_run_qc_plink_native_fst", _fake_plink_native_backend)
    monkeypatch.setattr(qc_module, "_compute_ld_decay_qc", _fake_ld_decay)
    report = run_qc(cfg, chrom=21, n_shards=2)

    assert report.gate_pass is False
    assert report.ld_decay is not None
    assert report.ld_decay["gate_pass"] is False
    assert report.ld_decay_table_path is not None


def test_qc_ld_decay_plink_backend_dispatch(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_ld_decay_plink_dispatch")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    payload["qc"]["backend"] = "plink_native_fst"
    payload["qc"]["ld_decay_enabled"] = True
    payload["qc"]["ld_decay_backend"] = "plink"
    payload["qc"]["within_pop_split_checks"] = []
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    called = {"plink_ld": 0}

    def _fake_plink_native_backend(**kwargs):
        return (
            {"P1__P2": 0.01, "P1__P3": 0.01, "P2__P3": 0.01},
            {},
        )

    def _fake_ld_decay_plink(**kwargs):
        called["plink_ld"] += 1
        ld_json = tmp_path / "ld_decay_plink_fake.json"
        return (
            {
                "enabled": True,
                "backend": "plink",
                "status": "ok",
                "gate_pass": True,
                "table_path": str(tmp_path / "ld_decay_plink_fake.tsv"),
                "population_summary": {"P1": {"pass": True}},
            },
            ld_json,
        )

    monkeypatch.setattr(qc_module, "_run_qc_plink_native_fst", _fake_plink_native_backend)
    monkeypatch.setattr(qc_module, "_compute_ld_decay_qc_plink", _fake_ld_decay_plink)
    report = run_qc(cfg, chrom=21, n_shards=2)

    assert called["plink_ld"] == 1
    assert report.ld_decay is not None
    assert report.ld_decay.get("backend") == "plink"


def test_qc_stochastic_fst_gate_replaces_standard_gate_when_available(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_stochastic_replaces_standard")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["qc"]["backend"] = "python"
    payload["qc"]["main_pairwise_mode"] = "sampled"
    payload["qc"]["fst_tolerance_relative"] = 0.0
    payload["qc"]["sampled_gate_tolerance_relative"] = 0.0
    payload["qc"]["fst_stochastic_gate_mode"] = "on"
    payload["qc"]["within_pop_split_checks"] = []
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")

    def _fake_matrix_backend(**kwargs):
        return (
            {"P1__P2": 0.10, "P1__P3": 0.10, "P2__P3": 0.10},
            {},
        )

    def _fake_donor_baseline(*args, **kwargs):
        return {"P1__P2": 0.0, "P1__P3": 0.0, "P2__P3": 0.0}

    def _fake_stochastic_gate(**kwargs):
        return {
            "available": True,
            "gate_pass": True,
            "summary": {"p_ge_observed_abs_diff_mean": 0.9},
        }

    monkeypatch.setattr(qc_module, "_run_qc_matrix_hapstore", _fake_matrix_backend)
    monkeypatch.setattr(qc_module, "donor_baseline_fst", _fake_donor_baseline)
    monkeypatch.setattr(qc_module, "_compute_stochastic_donor_fst_gate", _fake_stochastic_gate)

    report = run_qc(cfg, chrom=21, n_shards=2)
    assert report.gate_pass is True

    with open(report.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["fst_gate_standard_pass"] is False
    assert diag["fst_gate_effective_pass"] is True
    assert diag["fst_gate_strategy_used"] == "stochastic_donor_baseline"


def test_apply_admixed_non_gating_metric_helper():
    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    payload = {
        "gate_pass": False,
        "population_summary": {
            "P1": {"gate_pass": True},
            "P3": {"gate_pass": False},
        },
    }
    out = qc_module._apply_admixed_non_gating_to_metric(  # type: ignore[attr-defined]
        payload,
        admixed_pop_names={"P3"},
        reason="non_gating_admixed_policy",
    )
    assert out["population_summary"]["P1"]["gate_pass"] is True
    assert out["population_summary"]["P3"]["gate_pass"] is None
    assert out["population_summary"]["P3"]["gate_reason"] == "non_gating_admixed_policy"
    assert out["gate_pass"] is True
