from __future__ import annotations

import importlib
import json

import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool

from .conftest import write_test_config


def test_qc_ld_decay_gate_updates_overall_gate(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_ld_decay_gate")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    payload["qc"]["backend"] = "python"
    payload["qc"]["within_pop_split_checks"] = []
    payload["qc"]["ld_decay_enabled"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=2)

    matrix_backend = importlib.import_module("forgephast.amplifier.qc.matrix_backend")
    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")

    def _fake_matrix_backend(**kwargs):
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

    monkeypatch.setattr(matrix_backend, "run_qc_matrix_hapstore_impl", _fake_matrix_backend)
    monkeypatch.setattr(qc_module, "_compute_ld_decay_qc", _fake_ld_decay)
    report = run_qc(cfg, chrom=21, n_shards=2)

    assert report.gate_pass is False
    assert report.ld_decay is not None
    assert report.ld_decay["gate_pass"] is False
    assert report.ld_decay_table_path is not None


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
    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=2)

    matrix_backend = importlib.import_module("forgephast.amplifier.qc.matrix_backend")
    run_support = importlib.import_module("forgephast.amplifier.qc.run_support")

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

    monkeypatch.setattr(matrix_backend, "run_qc_matrix_hapstore_impl", _fake_matrix_backend)
    monkeypatch.setattr(run_support, "donor_baseline_fst", _fake_donor_baseline)
    monkeypatch.setattr(run_support, "_compute_stochastic_donor_fst_gate", _fake_stochastic_gate)

    report = run_qc(cfg, chrom=21, n_shards=2)
    assert report.gate_pass is True

    with open(report.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["fst_gate_standard_pass"] is False
    assert diag["fst_gate_effective_pass"] is True
    assert diag["fst_gate_strategy_used"] == "stochastic_donor_baseline"


def test_apply_admixed_non_gating_metric_helper():
    qc_module = importlib.import_module("forgephast.amplifier.qc.admixed_support")
    payload = {
        "gate_pass": False,
        "population_summary": {
            "P1": {"gate_pass": True},
            "P3": {"gate_pass": False},
        },
    }
    out = qc_module.apply_admixed_non_gating_to_metric(
        payload,
        admixed_pop_names={"P3"},
        reason="non_gating_admixed_policy",
    )
    assert out["population_summary"]["P1"]["gate_pass"] is True
    assert out["population_summary"]["P3"]["gate_pass"] is None
    assert out["population_summary"]["P3"]["gate_reason"] == "non_gating_admixed_policy"
    assert out["gate_pass"] is True
