from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import _read_sample_manifests, merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool

from .conftest import write_test_config


def test_read_sample_manifests_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    first.write_text("sample_id\tshard_id\ns1\t0\n", encoding="utf-8")
    second.write_text("sample_id\tshard_id\ns1\t1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        _read_sample_manifests([first, second])


def test_read_sample_manifests_rejects_invalid_shard_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample_id\tshard_id\ns1\tnot-an-integer\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _read_sample_manifests([manifest])



def test_merge_and_qc_pipeline(tmp_path):
    cfg_path = write_test_config(tmp_path, name="merge_qc")
    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert merge_result.combined_hapstore is not None
    assert merge_result.combined_hapstore.exists()
    assert merge_result.merged_manifest.exists()

    qc = run_qc(cfg, chrom=21, n_shards=2)
    assert qc.diagnostics_path.exists()
    assert any(key.endswith("P2") or "P1__P2" in key for key in qc.pairwise_fst.keys())
    assert set(qc.within_split_fst.keys()) == {"P1", "P2", "P3"}
    with open(qc.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["synthetic_shard_round_robin"]["status"] == "ok"


def test_merge_and_qc_pipeline_with_extended_metrics(tmp_path: Path):
    cfg_path = write_test_config(tmp_path, name="merge_qc_extended")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    payload.setdefault("qc", {})
    payload["qc"].update(
        {
            "king_enabled": True,
            "afs_enabled": True,
            "hwe_enabled": True,
            "inbreeding_f_enabled": True,
            "pca_enabled": True,
            "roh_enabled": True,
            "plots_enabled": True,
            "plots_dirname": "plots",
            "ext_max_diploids_per_population": 50,
            "ext_max_variants": 50,
            "king_max_within_pairs_per_population": 50,
            "king_max_cross_pairs_per_pop_pair": 50,
            "afs_n_bins": 10,
            "pca_n_components": 5,
            "pca_auc_n_components": 5,
            "roh_max_diploids_per_population": 50,
            "roh_max_variants": 100,
            "roh_min_segment_cm": 0.1,
            "roh_max_gap_cm": 0.5,
        }
    )

    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")
    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert merge_result.combined_hapstore is not None
    assert merge_result.combined_hapstore.exists()

    qc = run_qc(cfg, chrom=21, n_shards=2)
    assert qc.diagnostics_path.exists()

    with open(qc.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)

    for key in ("king", "afs", "hwe", "inbreeding_f", "pca", "roh"):
        assert key in diag
        assert diag[key] is not None

    assert diag["king"]["status"] == "ok"
    assert Path(diag["king"]["summary_path"]).exists()
    assert Path(diag["king"]["hist_path"]).exists()
    if diag["king"]["plot_path"] is not None:
        assert Path(diag["king"]["plot_path"]).exists()

    assert diag["afs"]["status"] == "ok"
    assert Path(diag["afs"]["table_path"]).exists()
    if diag["afs"]["plot_path"] is not None:
        assert Path(diag["afs"]["plot_path"]).exists()

    assert diag["hwe"]["status"] == "ok"
    assert Path(diag["hwe"]["table_path"]).exists()
    for path in (diag["hwe"].get("plot_paths") or {}).values():
        assert Path(path).exists()

    assert diag["inbreeding_f"]["status"] == "ok"
    assert Path(diag["inbreeding_f"]["table_path"]).exists()
    for path in (diag["inbreeding_f"].get("plot_paths") or {}).values():
        assert Path(path).exists()

    assert diag["pca"]["status"] == "ok"
    assert Path(diag["pca"]["table_path"]).exists()
    if diag["pca"]["plot_path"] is not None:
        assert Path(diag["pca"]["plot_path"]).exists()

    assert diag["roh"]["status"] == "ok"
    assert Path(diag["roh"]["table_path"]).exists()
    if diag["roh"]["plot_path"] is not None:
        assert Path(diag["roh"]["plot_path"]).exists()


def test_merge_from_population_shards_when_combined_shard_output_disabled(tmp_path):
    cfg_path = write_test_config(tmp_path, name="merge_no_combined_shard")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    cfg.qc.synthetic_shard_round_robin_enabled = True
    build_manifest_from_config(cfg)
    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert merge_result.combined_hapstore is None
    assert set(merge_result.population_merged_hapstores.keys()) == {"P1", "P2", "P3"}
    assert all(path.exists() for path in merge_result.population_merged_hapstores.values())

    qc = run_qc(cfg, chrom=21, n_shards=2)
    assert qc.mode == "population_merged_hapstore"
    assert qc.vcf_path is None
    assert set(qc.population_hapstore_paths.keys()) == {"P1", "P2", "P3"}
    assert qc.diagnostics_path.exists()
    with open(qc.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["synthetic_shard_round_robin"]["status"] == "ok"


def test_merge_can_restrict_to_population_subset(tmp_path):
    cfg_path = write_test_config(tmp_path, name="merge_subset")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
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

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2, populations=["P2"])
    assert merge_result.combined_hapstore is None
    assert set(merge_result.population_merged_hapstores.keys()) == {"P2"}
    assert merge_result.population_merged_hapstores["P2"].exists()
    assert merge_result.merged_manifest.exists()


def test_merge_and_qc_hapstore_only_when_vcf_output_disabled(tmp_path):
    cfg_path = write_test_config(tmp_path, name="merge_hapstore_only")
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
    simulate_chromosome_population_shard_pool(cfg, chrom=21, n_shards=2, task_workers=2, threads_per_task=1, task_start_method="fork")

    for shard_id in (0, 1):
        shard_dir = cfg.run_root / "synthetic" / "chr21" / f"shard_{shard_id}_of_2"
        for pop in ("P1", "P2", "P3"):
            assert not (shard_dir / f"{pop}.chr21.vcf.gz").exists()
            assert (shard_dir / f"{pop}.chr21.hapstore.zarr").exists()

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert set(merge_result.population_merged_hapstores.keys()) == {"P1", "P2", "P3"}
    assert merge_result.combined_hapstore is None

    qc = run_qc(cfg, chrom=21, n_shards=2)
    assert qc.mode == "population_merged_hapstore"
    assert qc.vcf_path is None
    assert set(qc.population_hapstore_paths.keys()) == {"P1", "P2", "P3"}
    assert qc.diagnostics_path.exists()


def test_config_rejects_vcf_without_hapstore_runtime(tmp_path):
    cfg_path = write_test_config(tmp_path, name="rejects_vcf_without_hapstore")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_hapstore"] = False
    payload["outputs"]["write_phased_vcf"] = True
    payload["outputs"]["merge_combined_populations"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="write_hapstore"):
        _ = from_yaml(cfg_path)
