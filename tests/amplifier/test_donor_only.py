from __future__ import annotations

import pandas as pd
import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool

from .conftest import write_test_config


def test_donor_only_population_is_available_but_not_simulated(tmp_path, monkeypatch):
    cfg_path = write_test_config(tmp_path, name="donor_only")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][0]["donor_only"] = True
    payload["populations"][0]["n_diploids"] = 0
    payload["outputs"]["scratch_root"] = str(tmp_path / "scratch")
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    donor_manifest = build_manifest_from_config(cfg)
    donor_rows = pd.read_csv(donor_manifest.manifest_path, sep="\t")
    assert set(donor_rows.loc[donor_rows["population"] == "P1", "sample_id"]) == {"A1", "A2"}
    assert donor_rows.loc[donor_rows["population"] == "P1", "donor_only"].all()

    monkeypatch.setenv("FORGEPHAST_CPU_BUDGET", "2")
    batch = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=1,
        task_workers=3,
        threads_per_task=1,
        task_start_method="fork",
    )
    assert {item.population for item in batch.population_outputs} == {"P2", "P3"}

    sample_manifest = pd.read_csv(batch.sample_manifest_path, sep="\t")
    assert set(sample_manifest["population"]) == {"P2", "P3"}

    merged = merge_chromosome(cfg, chrom=21, n_shards=1)
    assert set(merged.population_merged_hapstores) == {"P2", "P3"}
    assert not list((tmp_path / "scratch").rglob("*.hapstore.zarr"))
