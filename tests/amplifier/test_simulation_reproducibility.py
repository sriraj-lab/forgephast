from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import pytest

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.maps import load_genetic_map, switch_probs_from_positions
from forgephast.amplifier.simulation import simulate_haplotypes, simulate_chromosome
from forgephast.amplifier.vcfio import load_donor_haplotypes
import forgephast.amplifier.vcfio as vcfio_module

from .conftest import write_test_config



def test_haplotype_simulation_seed_determinism(tmp_path):
    cfg_path = write_test_config(tmp_path)
    cfg = from_yaml(cfg_path)
    manifest = build_manifest_from_config(cfg)

    donors = manifest.donors_by_population["P1"]
    panel = load_donor_haplotypes(
        vcf_path=Path(cfg.input.panel_vcf_pattern.format(chrom=21, population="P1")),
        chrom=21,
        donor_ids=donors,
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



def test_simulate_chromosome_emits_outputs(tmp_path):
    cfg_path = write_test_config(tmp_path)
    cfg = from_yaml(cfg_path)

    result = simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=1)

    assert result.combined_vcf.exists()
    assert (Path(str(result.combined_vcf) + ".tbi")).exists()
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()
    assert result.sample_manifest_path.exists()


def test_simulate_chromosome_can_skip_combined_output(tmp_path):
    cfg_path = write_test_config(tmp_path, name="skip_combined")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=1)

    assert result.combined_vcf is None
    assert result.combined_hapstore is None
    assert result.sample_manifest_path.exists()
    pop_vcfs = [item.vcf_path for item in result.population_outputs if item.vcf_path is not None]
    pop_hapstores = [item.hapstore_path for item in result.population_outputs if item.hapstore_path is not None]
    assert len(pop_vcfs) == 3
    assert len(pop_hapstores) == 3
    assert all(path.exists() for path in pop_vcfs)
    assert all(path.exists() for path in pop_hapstores)


def test_simulate_chromosome_with_parallel_donor_io(tmp_path):
    cfg_path = write_test_config(tmp_path, name="parallel_donor_io")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["simulation"]["donor_io_workers"] = 2
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=1)
    assert result.combined_vcf is not None
    assert result.combined_vcf.exists()
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()


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


def test_simulate_chromosome_with_hapstore_donor_input(monkeypatch, tmp_path):
    cfg_path = write_test_config(tmp_path, name="hapstore_donor_input")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    panel_vcf = Path(payload["input"]["panel_vcf_pattern"].format(chrom=21, population="P1"))
    payload["input"].pop("panel_vcf_pattern", None)
    payload["input"]["panel_hapstore_pattern"] = str(tmp_path / "panel" / "chr{chrom}.hapstore.zarr")
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    def _fake_hapstore_loader(hapstore_path, chrom, donor_ids):
        return load_donor_haplotypes(
            vcf_path=panel_vcf,
            chrom=chrom,
            donor_ids=donor_ids,
        )

    monkeypatch.setattr(vcfio_module, "load_donor_haplotypes_hapstore", _fake_hapstore_loader)

    cfg = from_yaml(cfg_path)
    result = simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=1)
    assert result.combined_hapstore is not None
    assert result.combined_hapstore.exists()
