from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from forgephast.amplifier.cli import _parse_args
from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool

from .conftest import write_test_config


def _load_diag(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_synthetic_shard_round_robin_metric_exact_default(tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_shard_rr_exact")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=2)

    report = run_qc(cfg, chrom=21, n_shards=2)
    diag = _load_diag(report.diagnostics_path)
    metric = diag["synthetic_shard_round_robin"]

    assert metric["status"] == "ok"
    assert metric["gate_pass"] is None
    assert metric["n_records"] == 3
    assert metric["n_candidate_pairs"] == 3
    assert metric["n_selected_pairs"] == 3
    assert metric["pairwise_mode"] == "exact"
    assert set(metric["population_summary"].keys()) == {"P1", "P2", "P3"}

    table = pd.read_csv(metric["table_path"], sep="\t")
    assert len(table) == 3
    assert set(table["population"].unique()) == {"P1", "P2", "P3"}
    assert set(table["shard_a"].unique()) == {0}
    assert set(table["shard_b"].unique()) == {1}


def test_synthetic_shard_round_robin_metric_respects_max_pairs_per_population(tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_shard_rr_max_pairs_cap")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    cfg.qc.synthetic_shard_round_robin_max_pairs_per_population = 2
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=4, task_workers=4, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=4)

    report = run_qc(cfg, chrom=21, n_shards=4)
    metric = _load_diag(report.diagnostics_path)["synthetic_shard_round_robin"]

    assert metric["n_candidate_pairs"] == 18
    assert metric["n_selected_pairs"] == 9
    table = pd.read_csv(metric["table_path"], sep="\t")
    assert len(table) == 9
    assert table.groupby("population").size().to_dict() == {"P1": 3, "P2": 3, "P3": 3}


def test_synthetic_shard_round_robin_metric_sampled_mode_hapstore_only(tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_shard_rr_sampled_hapstore_only")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    cfg.qc.synthetic_shard_round_robin_pairwise_mode = "sampled"
    cfg.outputs.write_phased_vcf = False
    cfg.outputs.write_pgen = False
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=2)

    report = run_qc(cfg, chrom=21, n_shards=2)
    metric = _load_diag(report.diagnostics_path)["synthetic_shard_round_robin"]
    assert metric["status"] == "ok"
    assert metric["pairwise_mode"] == "sampled"
    assert metric["n_records"] == 3


def test_synthetic_shard_round_robin_metric_respects_min_cpus(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_shard_rr_pair_parallel")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    cfg.qc.synthetic_shard_round_robin_pairwise_mode = "sampled"
    cfg.qc.synthetic_shard_round_robin_min_cpus_per_match = 4
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=2)

    metric_module = importlib.import_module("forgephast.amplifier.qc.metrics.synthetic_shard_round_robin")
    observed_threads: list[int] = []

    def _fake_hudson(ac_a, an_a, ac_b, an_b, *, numba_threads=None):
        observed_threads.append(int(numba_threads))
        return 0.01

    monkeypatch.setattr(metric_module, "matrix_hudson_fst_from_counts", _fake_hudson)
    report = run_qc(cfg, chrom=21, n_shards=2)
    metric = _load_diag(report.diagnostics_path)["synthetic_shard_round_robin"]

    assert metric["n_records"] == 3
    assert observed_threads == [4, 4, 4]


def test_synthetic_shard_round_robin_metric_counts_only_needed_shards(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_shard_rr_needed_shards_only")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    cfg.qc.synthetic_shard_round_robin_pairwise_mode = "sampled"
    cfg.qc.synthetic_shard_round_robin_max_pairs_per_population = 1
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=4, task_workers=4, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=4)

    metric_module = importlib.import_module("forgephast.amplifier.qc.metrics.synthetic_shard_round_robin")
    count_calls: list[object] = []

    def _fake_counts_for_samples(*, store, chrom, sample_indices, variant_indices, variant_block_size, numba_threads):
        import numpy as np

        count_calls.append((store, tuple(int(x) for x in sample_indices.tolist())))
        ac = np.zeros(10, dtype=np.int64)
        an = np.full(10, 2, dtype=np.int64)
        return ac, an

    monkeypatch.setattr(metric_module, "hapstore_counts_for_samples", _fake_counts_for_samples)
    monkeypatch.setattr(metric_module, "matrix_hudson_fst_from_counts", lambda *args, **kwargs: 0.01)

    report = run_qc(cfg, chrom=21, n_shards=4)
    metric = _load_diag(report.diagnostics_path)["synthetic_shard_round_robin"]

    assert metric["n_records"] == 3
    assert len(count_calls) == 6


def test_synthetic_shard_round_robin_metric_no_pairs_for_single_shard(tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_shard_rr_single_shard")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=1, task_workers=1, threads_per_task=1, task_start_method="fork")
    merge_chromosome(cfg, chrom=21, n_shards=1)

    report = run_qc(cfg, chrom=21, n_shards=1)
    metric = _load_diag(report.diagnostics_path)["synthetic_shard_round_robin"]

    assert metric["status"] == "insufficient_shards"
    assert metric["gate_pass"] is None
    assert metric["n_records"] == 0


def test_removed_synthetic_premerge_cli_command_is_rejected(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["forgephast", "qc-synthetic-premerge-round-robin", "--config", "cfg.yaml", "--chrom", "21", "--n-shards", "2"],
    )
    with pytest.raises(SystemExit):
        _parse_args()
