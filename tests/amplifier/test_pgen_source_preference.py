from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool

from .conftest import write_test_config


merge_mod = importlib.import_module("forgephast.amplifier.merge")
sim_mod = importlib.import_module("forgephast.amplifier.simulation")


def _touch_triplet(prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pgen", ".pvar", ".psam"):
        Path(f"{prefix}{ext}").write_text("", encoding="utf-8")


def test_simulate_prefers_hapstore_direct_pgen(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="sim_direct_pgen")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_phased_vcf"] = False
    payload["outputs"]["write_hapstore"] = True
    payload["outputs"]["write_pgen"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    calls: list[Path] = []

    def _fake_direct(*, hapstore_path, out_prefix, chrom, variant_block_size=8192):
        assert Path(hapstore_path).exists()
        assert str(chrom) in {"21", "chr21"}
        calls.append(Path(out_prefix))
        _touch_triplet(Path(out_prefix))
        return Path(out_prefix)

    monkeypatch.setattr(sim_mod, "convert_hapstore_to_pgen", _fake_direct)

    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=1,
        threads_per_task=1,
        task_start_method="fork",
    )

    assert result.combined_vcf is None
    assert result.combined_hapstore is not None and result.combined_hapstore.exists()
    assert result.combined_pgen_prefix is not None
    assert len(calls) == 4  # 3 populations + 1 combined output
    for artifact in result.population_outputs:
        assert artifact.vcf_path is None
        assert artifact.pgen_prefix is not None
        assert Path(f"{artifact.pgen_prefix}.pgen").exists()


def test_simulate_pool_prefers_hapstore_direct_pgen(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="sim_pool_direct_pgen")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_phased_vcf"] = False
    payload["outputs"]["write_hapstore"] = True
    payload["outputs"]["write_pgen"] = True
    payload["outputs"]["write_combined_shard_output"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    calls: list[Path] = []

    def _fake_direct(*, hapstore_path, out_prefix, chrom, variant_block_size=8192):
        assert Path(hapstore_path).exists()
        calls.append(Path(out_prefix))
        _touch_triplet(Path(out_prefix))
        return Path(out_prefix)

    monkeypatch.setattr(sim_mod, "convert_hapstore_to_pgen", _fake_direct)

    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=2,
        task_workers=2,
        threads_per_task=1,
        task_start_method="fork",
    )

    assert result.task_count == 6
    # Only parent-process calls are observable here; worker-process calls do not
    # mutate this list under multiprocessing.
    assert len(calls) == 2  # combined shard outputs
    for shard_id in range(2):
        shard_dir = cfg.run_root / "synthetic" / "chr21" / f"shard_{shard_id}_of_2"
        assert (shard_dir / "P1.chr21.pgen").exists()
        assert (shard_dir / "P2.chr21.pgen").exists()
        assert (shard_dir / "P3.chr21.pgen").exists()
        assert (shard_dir / "combined.chr21.pgen").exists()
        assert not (shard_dir / "combined.chr21.vcf.gz").exists()


def test_merge_writes_pgen_from_combined_hapstore(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="merge_direct_pgen")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_phased_vcf"] = False
    payload["outputs"]["write_hapstore"] = True
    payload["outputs"]["write_pgen"] = False
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=2,
        task_workers=2,
        threads_per_task=1,
        task_start_method="fork",
    )

    cfg.outputs.write_pgen = True

    direct_calls: list[Path] = []

    def _fake_direct(*, hapstore_path, out_prefix, chrom, variant_block_size=8192):
        assert Path(hapstore_path).exists()
        direct_calls.append(Path(out_prefix))
        _touch_triplet(Path(out_prefix))
        return Path(out_prefix)

    monkeypatch.setattr(merge_mod, "convert_hapstore_to_pgen", _fake_direct)

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)

    assert merge_result.combined_hapstore is not None and merge_result.combined_hapstore.exists()
    assert merge_result.merged_pgen_prefix is not None
    assert direct_calls == [merge_result.merged_pgen_prefix]
    assert Path(f"{merge_result.merged_pgen_prefix}.pgen").exists()
