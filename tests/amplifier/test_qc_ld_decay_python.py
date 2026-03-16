from __future__ import annotations

import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome

from .conftest import write_test_config


def test_ld_decay_runs_with_python_qc_backend(tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_ld_decay_python")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    payload["qc"]["backend"] = "python"
    payload["qc"]["ld_decay_enabled"] = True
    payload["qc"]["ext_max_diploids_per_population"] = 4
    payload["qc"]["ext_max_variants"] = 8
    payload["qc"]["ld_decay_min_pairs_per_bin"] = 1
    payload["qc"]["ld_decay_max_distance_cm"] = 1.0
    payload["qc"]["ld_decay_bin_width_cm"] = 0.2
    payload["qc"]["ld_decay_rmse_max"] = 1.0
    payload["qc"]["ld_decay_half_decay_rel_diff_max"] = 1.0
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)
    report = run_qc(cfg, chrom=21, n_shards=2)

    assert report.ld_decay is not None
    assert "status" in report.ld_decay
    assert report.ld_decay_table_path is not None
    assert report.ld_decay_table_path.exists()
