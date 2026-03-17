from __future__ import annotations

import logging

import yaml

from forgephast.amplifier.api import simulate as simulate_api
from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool

from .conftest import write_test_config


def test_simulate_pool_population_shard_outputs(tmp_path):
    cfg_path = write_test_config(tmp_path, name="sim_pool")
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=2,
        task_workers=2,
        threads_per_task=1,
        task_start_method="fork",
    )
    assert result.task_count == 6
    assert len(result.sample_manifest_paths) == 2
    assert all(path.exists() for path in result.sample_manifest_paths)

    for shard_id in range(2):
        shard_dir = cfg.run_root / "synthetic" / "chr21" / f"shard_{shard_id}_of_2"
        assert (shard_dir / "P1.chr21.hapstore.zarr").exists()
        assert (shard_dir / "P2.chr21.hapstore.zarr").exists()
        assert (shard_dir / "P3.chr21.hapstore.zarr").exists()
        assert (shard_dir / "combined.chr21.hapstore.zarr").exists()

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert merge_result.merged_manifest.exists()
    assert merge_result.combined_hapstore is not None
    assert merge_result.combined_hapstore.exists()
    qc = run_qc(cfg, chrom=21, n_shards=2)
    assert qc.diagnostics_path.exists()


def test_simulate_pool_logs_intermediate_progress(tmp_path, caplog):
    cfg_path = write_test_config(tmp_path, name="sim_pool_logs")
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    caplog.set_level(logging.INFO, logger="forgephast.amplifier.simulation")
    simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=2,
        task_workers=2,
        threads_per_task=1,
        task_start_method="fork",
    )

    messages = [rec.getMessage() for rec in caplog.records if rec.name == "forgephast.amplifier.simulation"]
    assert any("simulate-pool task grid" in msg for msg in messages)
    assert any("simulate-pool preload complete" in msg for msg in messages)
    assert any("simulate-pool donor prep complete" in msg for msg in messages)
    assert any("simulate-pool progress" in msg for msg in messages)


def test_simulate_pool_can_skip_vcf_outputs(tmp_path):
    cfg_path = write_test_config(tmp_path, name="sim_pool_no_vcf")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_phased_vcf"] = False
    payload["outputs"]["write_pgen"] = False
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=2,
        task_workers=2,
        threads_per_task=1,
        task_start_method="fork",
    )
    assert result.task_count == 6
    for shard_id in range(2):
        shard_dir = cfg.run_root / "synthetic" / "chr21" / f"shard_{shard_id}_of_2"
        assert not (shard_dir / "P1.chr21.vcf.gz").exists()
        assert not (shard_dir / "P2.chr21.vcf.gz").exists()
        assert not (shard_dir / "P3.chr21.vcf.gz").exists()
        assert (shard_dir / "P1.chr21.hapstore.zarr").exists()
        assert (shard_dir / "P2.chr21.hapstore.zarr").exists()
        assert (shard_dir / "P3.chr21.hapstore.zarr").exists()

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert set(merge_result.population_merged_hapstores.keys()) == {"P1", "P2", "P3"}
    qc = run_qc(cfg, chrom=21, n_shards=2)
    assert qc.mode == "population_merged_hapstore"


def test_simulate_api_uses_yaml_defaults(tmp_path):
    cfg_path = write_test_config(tmp_path, name="sim_pool_yaml_defaults")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("simulation", {})["n_shards"] = 2
    payload.setdefault("simulation", {})["task_workers"] = 2
    payload.setdefault("simulation", {})["threads_per_task"] = 1
    payload.setdefault("simulation", {})["task_start_method"] = "fork"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    _ = simulate_api(
        config_path=cfg_path,
        chrom=21,
        n_shards=None,
        workers=None,
        threads_per_task=None,
        task_start_method=None,
    )
    cfg = from_yaml(cfg_path)
    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert merge_result.merged_manifest.exists()
