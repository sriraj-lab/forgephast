from __future__ import annotations

import yaml

from forgephast.amplifier.api import build_cache
from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.simulation import simulate_chromosome

from .conftest import write_test_config


def test_build_cache_first_run_then_cache_hit(tmp_path):
    cfg_path = write_test_config(tmp_path, name="cache_warm")

    first = build_cache(cfg_path, chroms=[21])
    assert first.entries
    assert all(not entry.cache_hit for entry in first.entries)
    assert all(entry.cache_path.exists() for entry in first.entries)

    second = build_cache(cfg_path, chroms=[21])
    assert second.entries
    assert all(entry.cache_hit for entry in second.entries)


def test_simulate_chromosome_uses_stage0_cache(tmp_path):
    cfg_path = write_test_config(tmp_path, name="cache_sim")
    _ = build_cache(cfg_path, chroms=[21])

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=1)
    assert result.stage_timings["preload_panel_cache_hits"] >= 1
    assert result.stage_timings["preload_panel_cache_misses"] == 0
