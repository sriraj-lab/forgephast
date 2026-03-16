from __future__ import annotations

import importlib

import pandas as pd
import pytest

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.qc import run_synthetic_premerge_within_pop_round_robin
from forgephast.amplifier.simulation import simulate_chromosome

from .conftest import write_test_config



def test_synthetic_premerge_round_robin_exact_default(tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_premerge_rr_exact")
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)

    report = run_synthetic_premerge_within_pop_round_robin(
        cfg,
        chrom=21,
        n_shards=2,
        seed=7,
        workers=2,
    )
    assert report.diagnostics_path.exists()
    assert report.table_path.exists()
    assert report.n_records == 3
    assert report.n_candidate_pairs == 3
    assert report.n_selected_pairs == 3
    assert set(report.summary_by_population.keys()) == {"P1", "P2", "P3"}

    table = pd.read_csv(report.table_path, sep="\t")
    assert len(table) == 3
    assert set(table["population"].unique()) == {"P1", "P2", "P3"}
    assert set(table["shard_a"].unique()) == {0}
    assert set(table["shard_b"].unique()) == {1}


def test_synthetic_premerge_round_robin_respects_max_pairs_per_population(tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_premerge_rr_max_pairs_cap")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_round_robin_max_pairs_per_population = 2
    build_manifest_from_config(cfg)

    for shard_id in range(4):
        simulate_chromosome(cfg, chrom=21, shard_id=shard_id, n_shards=4)

    report = run_synthetic_premerge_within_pop_round_robin(
        cfg,
        chrom=21,
        n_shards=4,
        seed=13,
        workers=2,
    )
    assert report.n_candidate_pairs == 18
    # max_pairs_per_population is treated as a target; we choose the smallest shard subset
    # whose full nC2 comparisons meets/exceeds the target.
    # target=2 => subset_shards=3 => 3 comparisons per population.
    assert report.n_selected_pairs == 9

    table = pd.read_csv(report.table_path, sep="\t")
    assert len(table) == 9
    assert table.groupby("population").size().to_dict() == {"P1": 3, "P2": 3, "P3": 3}


def test_synthetic_premerge_round_robin_sampled_mode_uses_plink_hook(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_premerge_rr_sampled_mode")
    cfg = from_yaml(cfg_path)
    cfg.qc.rr_pairwise_mode = "sampled"
    cfg.qc.backend = "plink_native_fst"
    build_manifest_from_config(cfg)

    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")

    def _fake_plink_fst_for_groups(**kwargs):
        pairs = kwargs["pairs"]
        out = {}
        for idx, (a, b) in enumerate(pairs):
            out[qc_module._pair_key(a, b)] = 0.02 + idx * 0.001
        return out

    monkeypatch.setattr(qc_module, "_plink_fst_for_groups", _fake_plink_fst_for_groups)

    report = run_synthetic_premerge_within_pop_round_robin(
        cfg,
        chrom=21,
        n_shards=2,
        seed=7,
        workers=2,
    )

    assert report.n_records == 3
    table = pd.read_csv(report.table_path, sep="\t")
    assert set(table["fst"].round(6).tolist()) == {0.02}


def test_synthetic_premerge_round_robin_sampled_mode_python_backend_without_vcfs(tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_premerge_rr_sampled_python_no_vcf")
    cfg = from_yaml(cfg_path)
    cfg.qc.backend = "python"
    cfg.qc.rr_pairwise_mode = "sampled"
    cfg.outputs.write_phased_vcf = False
    cfg.outputs.write_pgen = False
    build_manifest_from_config(cfg)

    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)

    for shard_id in (0, 1):
        shard_dir = cfg.run_root / "synthetic" / "chr21" / f"shard_{shard_id}_of_2"
        assert not (shard_dir / "P1.chr21.vcf.gz").exists()
        assert (shard_dir / "P1.chr21.hapstore.zarr").exists()

    report = run_synthetic_premerge_within_pop_round_robin(
        cfg,
        chrom=21,
        n_shards=2,
        seed=7,
        workers=2,
    )
    assert report.n_records == 3
    assert set(report.summary_by_population.keys()) == {"P1", "P2", "P3"}


def test_synthetic_premerge_round_robin_python_pair_parallel_respects_min_cpus(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_premerge_rr_python_pair_parallel")
    cfg = from_yaml(cfg_path)
    cfg.qc.backend = "python"
    cfg.qc.rr_pairwise_mode = "sampled"
    cfg.qc.synthetic_rr_min_cpus_per_match = 4
    cfg.outputs.write_phased_vcf = False
    cfg.outputs.write_pgen = False
    build_manifest_from_config(cfg)

    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    observed_threads: list[int] = []

    def _fake_hudson(ac_a, an_a, ac_b, an_b, *, numba_threads=None):
        observed_threads.append(int(numba_threads))
        return 0.01

    monkeypatch.setattr(qc_module, "matrix_hudson_fst_from_counts", _fake_hudson)
    report = run_synthetic_premerge_within_pop_round_robin(
        cfg,
        chrom=21,
        n_shards=2,
        seed=7,
        workers=8,
    )
    assert report.n_records == 3
    assert len(observed_threads) == 3
    assert observed_threads.count(4) == 3


def test_synthetic_premerge_round_robin_matrix_counts_only_needed_shards(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="syn_premerge_rr_needed_shards_only")
    cfg = from_yaml(cfg_path)
    cfg.qc.backend = "python"
    cfg.qc.rr_pairwise_mode = "sampled"
    cfg.qc.synthetic_round_robin_max_pairs_per_population = 1
    cfg.outputs.write_phased_vcf = False
    cfg.outputs.write_pgen = False
    build_manifest_from_config(cfg)

    for shard_id in range(4):
        simulate_chromosome(cfg, chrom=21, shard_id=shard_id, n_shards=4)

    qc_module = importlib.import_module("forgephast.amplifier.qc.pipeline")
    count_calls: list[object] = []

    def _fake_counts_for_samples(*, store, chrom, sample_indices, variant_indices, variant_block_size, numba_threads):
        import numpy as np

        count_calls.append(store)
        ac = np.zeros(10, dtype=np.int64)
        an = np.full(10, 2, dtype=np.int64)
        return ac, an

    monkeypatch.setattr(qc_module, "hapstore_counts_for_samples", _fake_counts_for_samples)
    monkeypatch.setattr(qc_module, "matrix_hudson_fst_from_counts", lambda *args, **kwargs: 0.01)

    report = run_synthetic_premerge_within_pop_round_robin(
        cfg,
        chrom=21,
        n_shards=4,
        seed=7,
        workers=2,
    )

    assert report.n_records == 3
    # 3 populations, max_pairs_per_population=1 => one pair per pop => 2 unique shards per pop.
    assert len(count_calls) == 6
