from __future__ import annotations

import numpy as np

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome

from .conftest import write_test_config



def _prepare_two_shard_run(tmp_path, name: str):
    cfg_path = write_test_config(tmp_path, name=name)
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)
    return cfg



def test_qc_python_hapstore_sampled_mode_is_deterministic(tmp_path):
    cfg = _prepare_two_shard_run(tmp_path, name="qc_matrix_sampled")
    cfg.qc.backend = "python"
    cfg.qc.main_pairwise_mode = "sampled"
    cfg.qc.seed = 123
    cfg.qc.sampled_max_diploids_per_population = 2
    cfg.qc.sampled_max_variants = 4

    out1 = run_qc(cfg, chrom=21, n_shards=2)
    out2 = run_qc(cfg, chrom=21, n_shards=2)

    assert out1.mode.endswith("hapstore")
    assert out1.pairwise_fst == out2.pairwise_fst
    assert set(out1.within_split_fst.keys()) == set(out2.within_split_fst.keys())
    for pop in out1.within_split_fst:
        a = out1.within_split_fst[pop]
        b = out2.within_split_fst[pop]
        if np.isnan(a):
            assert np.isnan(b)
        else:
            assert a == b



def test_qc_python_hapstore_exact_mode_runs(tmp_path):
    cfg = _prepare_two_shard_run(tmp_path, name="qc_matrix_exact")
    cfg.qc.backend = "python"
    cfg.qc.main_pairwise_mode = "exact"

    out = run_qc(cfg, chrom=21, n_shards=2)

    assert out.mode.endswith("hapstore")
    assert set(out.pairwise_fst.keys()) == {"P1__P2", "P1__P3", "P2__P3"}
    assert set(out.within_split_fst.keys()) == {"P1", "P2", "P3"}
