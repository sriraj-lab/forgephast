from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import pytest
from hapstore import ingest_vcf

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donor_panels import load_donor_haplotypes_from_source
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.maps import load_genetic_map, switch_probs_from_positions
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool, simulate_haplotypes
import forgephast.amplifier.donor_panels as donor_panel_module
import forgephast.amplifier.simulation as simulation_module

from .conftest import write_test_config



def test_haplotype_simulation_seed_determinism(tmp_path):
    cfg_path = write_test_config(tmp_path)
    cfg = from_yaml(cfg_path)
    manifest = build_manifest_from_config(cfg)

    donors = manifest.donors_by_population["P1"]
    panel, _, _ = load_donor_haplotypes_from_source(
        source_path=Path(cfg.input.panel_vcf_pattern.format(chrom=21, population="P1")),
        source_kind="vcf",
        chrom=21,
        donor_ids=donors,
        cache_dir=cfg.donor_cache_root,
    )
    map_bp, map_cm = load_genetic_map(cfg.input.genetic_map_pattern.format(chrom=21, population="P1"))
    switch_probs = switch_probs_from_positions(panel.variants.pos, map_bp, map_cm, 1.0)

    out_a = simulate_haplotypes(
        donor_haps=panel.donor_haps,
        switch_probs=switch_probs,
        n_haps=8,
        seed=999,
        threads=2,
    )
    out_b = simulate_haplotypes(
        donor_haps=panel.donor_haps,
        switch_probs=switch_probs,
        n_haps=8,
        seed=999,
        threads=2,
    )
    out_c = simulate_haplotypes(
        donor_haps=panel.donor_haps,
        switch_probs=switch_probs,
        n_haps=8,
        seed=1000,
        threads=2,
    )

    assert np.array_equal(out_a, out_b)
    assert not np.array_equal(out_a, out_c)



def test_simulate_emits_outputs(tmp_path):
    cfg_path = write_test_config(tmp_path)
    cfg = from_yaml(cfg_path)

    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=1,
        threads_per_task=1,
        task_start_method="fork",
    )

    assert result.combined_vcf is None
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()
    assert result.sample_manifest_path.exists()


def test_simulate_emits_vcf_via_hapstore_export(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="emit_vcf")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_phased_vcf"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)

    def _fake_export(*, hapstore_path, output_vcf, chrom, variant_block_size=1024, plink2_location=None, keep_intermediate_files=False, n_shards=1):
        assert Path(hapstore_path).exists()
        output_vcf = Path(output_vcf)
        output_vcf.parent.mkdir(parents=True, exist_ok=True)
        output_vcf.write_text("dummy", encoding="utf-8")
        Path(f"{output_vcf}.tbi").write_text("dummy", encoding="utf-8")
        return output_vcf

    monkeypatch.setattr(simulation_module, "convert_hapstore_to_vcf", _fake_export)

    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=1,
        threads_per_task=1,
        task_start_method="fork",
    )

    assert result.combined_vcf is not None
    assert result.combined_vcf.exists()
    assert Path(f"{result.combined_vcf}.tbi").exists()
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()


def test_simulate_can_skip_combined_output(tmp_path):
    cfg_path = write_test_config(tmp_path, name="skip_combined")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=1,
        threads_per_task=1,
        task_start_method="fork",
    )

    assert result.combined_vcf is None
    assert result.combined_hapstore is None
    assert result.sample_manifest_path.exists()
    pop_hapstores = [item.hapstore_path for item in result.population_outputs if item.hapstore_path is not None]
    assert len(pop_hapstores) == 3
    assert all(path.exists() for path in pop_hapstores)


def test_simulate_with_parallel_donor_io(tmp_path):
    cfg_path = write_test_config(tmp_path, name="parallel_donor_io")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["simulation"]["donor_io_workers"] = 2
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=1,
        threads_per_task=1,
        task_start_method="fork",
    )
    assert result.combined_vcf is None
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()


def test_simulate_materializes_vcf_donor_hapstore_cache(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="vcf_to_hapstore_runtime")
    ingest_calls = {"count": 0}

    def _counting_ingest(*args, **kwargs):
        ingest_calls["count"] += 1
        return ingest_vcf(*args, **kwargs)

    monkeypatch.setattr(donor_panel_module, "ingest_vcf", _counting_ingest)

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=1,
        threads_per_task=1,
        task_start_method="fork",
    )

    cache_entries = list((cfg.donor_cache_root / "chr21").glob("*.hapstore.zarr"))
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()
    assert ingest_calls["count"] >= 1
    assert cache_entries


def test_require_chrom_sharded_panel_validation(tmp_path):
    cfg_path = write_test_config(tmp_path, name="require_chrom_panel")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["input"]["require_chrom_sharded_panel"] = True
    payload["input"]["panel_vcf_pattern"] = str(tmp_path / "panel" / "fixed.vcf.gz")
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="require_chrom_sharded_panel"):
        _ = from_yaml(cfg_path)


def test_simulate_with_hapstore_donor_input(tmp_path):
    cfg_path = write_test_config(tmp_path, name="hapstore_donor_input")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    panel_vcf = Path(payload["input"]["panel_vcf_pattern"].format(chrom=21, population="P1"))
    panel_hapstore = tmp_path / "panel" / "chr21.hapstore.zarr"
    ingest_vcf(vcf_path=panel_vcf, output_zarr=panel_hapstore, chrom=21)
    payload["input"].pop("panel_vcf_pattern", None)
    payload["input"]["panel_hapstore_pattern"] = str(tmp_path / "panel" / "chr{chrom}.hapstore.zarr")
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=1,
        threads_per_task=1,
        task_start_method="fork",
    )
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()
