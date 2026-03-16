from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.simulation import simulate_chromosome, simulate_chromosome_population_shard_pool
from forgephast.amplifier.vcfio import load_donor_haplotypes

from .conftest import write_test_config


msprime = pytest.importorskip("msprime")
assert msprime is not None


def _write_msprime_config(tmp_path, *, name: str, admixed: bool, n_shards: int) -> str:
    cfg_path = write_test_config(tmp_path, name=name, n_shards_workers=1)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    payload["simulation"]["n_shards"] = int(n_shards)
    payload["simulation"]["task_workers"] = 2
    payload["simulation"]["threads_per_task"] = 1
    payload["outputs"]["write_pgen"] = False
    payload["outputs"]["write_phased_vcf"] = True
    payload["outputs"]["write_hapstore"] = True
    payload.setdefault("qc", {})
    payload["qc"].update(
        {
            "admixed_tract_length_min_tracts": 1,
            "admixed_min_individuals_for_variance_gate": 1,
        }
    )

    if admixed:
        payload["populations"][2] = {
            "name": "P3",
            "n_diploids": 4,
            "engine": "msprime_tracts",
            "selector": None,
            "msprime": {
                "sources": [
                    {"population": "P1", "weight": 0.6},
                    {"population": "P2", "weight": 0.4},
                ],
                "admixture_time_generations": 8.0,
                "write_full_tract_tsv": True,
            },
        }
    else:
        payload["populations"][2] = {
            "name": "P3",
            "n_diploids": 4,
            "engine": "msprime_tracts",
            "selector": None,
            "msprime": {
                "sources": [{"population": "P1", "weight": 1.0}],
                "admixture_time_generations": None,
                "write_full_tract_tsv": True,
            },
        }

    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return str(cfg_path)


def test_msprime_deterministic_output_fixed_seed(tmp_path):
    cfg_a_path = _write_msprime_config(tmp_path, name="msprime_det_a", admixed=True, n_shards=1)
    cfg_a = from_yaml(cfg_a_path)
    build_manifest_from_config(cfg_a)
    res_a = simulate_chromosome(cfg_a, chrom=21, shard_id=0, n_shards=1)

    cfg_b_path = _write_msprime_config(tmp_path, name="msprime_det_b", admixed=True, n_shards=1)
    cfg_b = from_yaml(cfg_b_path)
    build_manifest_from_config(cfg_b)
    res_b = simulate_chromosome(cfg_b, chrom=21, shard_id=0, n_shards=1)

    pop_a = next(item for item in res_a.population_outputs if item.population == "P3")
    pop_b = next(item for item in res_b.population_outputs if item.population == "P3")
    assert pop_a.vcf_path is not None and pop_b.vcf_path is not None
    assert pop_a.tract_summary_path is not None and pop_b.tract_summary_path is not None
    assert pop_a.tract_summary_path.exists() and pop_b.tract_summary_path.exists()
    assert pop_a.individual_ancestry_path is not None and pop_b.individual_ancestry_path is not None
    assert pop_a.individual_ancestry_path.exists() and pop_b.individual_ancestry_path.exists()

    with open(pop_a.tract_summary_path, "r", encoding="utf-8") as handle:
        summary_a = json.load(handle)
    with open(pop_b.tract_summary_path, "r", encoding="utf-8") as handle:
        summary_b = json.load(handle)
    assert summary_a["configured_source_weights"] == summary_b["configured_source_weights"]
    assert summary_a["observed_source_proportions"] == summary_b["observed_source_proportions"]
    assert summary_a["tract_count"] == summary_b["tract_count"]

    haps_a = load_donor_haplotypes(
        vcf_path=pop_a.vcf_path,
        chrom=21,
        donor_ids=pop_a.sample_ids,
    ).donor_haps
    haps_b = load_donor_haplotypes(
        vcf_path=pop_b.vcf_path,
        chrom=21,
        donor_ids=pop_b.sample_ids,
    ).donor_haps
    assert np.array_equal(haps_a, haps_b)


def test_msprime_mixed_simulate_pool_merge_qc_integration(tmp_path):
    cfg_path = _write_msprime_config(tmp_path, name="msprime_pool", admixed=True, n_shards=2)
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)

    batch = simulate_chromosome_population_shard_pool(
        cfg,
        chrom=21,
        n_shards=2,
        task_workers=2,
    )
    assert batch.task_count > 0
    assert len(batch.sample_manifest_paths) == 2

    merge_result = merge_chromosome(cfg, chrom=21, n_shards=2)
    assert merge_result.merged_manifest.exists()

    qc = run_qc(cfg, chrom=21, n_shards=2)
    with open(qc.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["msprime_tract_composition_gate"]["enabled"] is True
    assert diag["admixed_qc"]["enabled"] is True
    assert diag["admixed_qc"]["population_status"]["P3"]["classification"] == "admixed_msprime_tracts"
    assert "tract_composition" in diag["admixed_qc"]["hard_gate_components"]
    assert "tract_length_distribution" in diag["admixed_qc"]["hard_gate_components"]
    assert "ancestry_variance" in diag["admixed_qc"]["hard_gate_components"]
    var_gate = diag["admixed_qc"]["hard_gate_components"]["ancestry_variance"]["population_summary"]["P3"]
    assert var_gate["per_source"]["P1"]["measure"] == "cm"
    assert var_gate["per_source"]["P2"]["measure"] == "cm"
    assert diag["admixed_qc"]["skipped_components"]["donor_baseline_pairs"]["reason"] == "skipped_admixed_pair_policy"
    assert sorted((diag.get("donor_pairwise_fst") or {}).keys()) == ["P1__P2"]
    assert diag["within_split_gate_status"]["P3"]["gated"] is False
    tract_len_gate = diag["admixed_qc"]["hard_gate_components"]["tract_length_distribution"]["population_summary"]["P3"]
    assert "plot_path" in tract_len_gate
    if tract_len_gate["plot_path"] is not None:
        assert Path(tract_len_gate["plot_path"]).exists()
    pca_cline = diag["admixed_qc"]["optional_diagnostics"]["pca_cline"]["population_summary"]["P3"]
    assert "plot_path" in pca_cline
    if pca_cline["plot_path"] is not None:
        assert Path(pca_cline["plot_path"]).exists()

    p3_pair_status = [
        value
        for value in diag["fst_pairwise_gate_status"].values()
        if isinstance(value.get("pair"), list) and "P3" in value["pair"]
    ]
    assert p3_pair_status
    assert all(status["gated"] is False for status in p3_pair_status)
    assert all(status["reason"] == "skipped_admixed_pair_policy" for status in p3_pair_status)


def test_single_source_msprime_remains_hard_gated(tmp_path):
    cfg_path = _write_msprime_config(tmp_path, name="msprime_single_source", admixed=False, n_shards=1)
    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=1)
    merge_chromosome(cfg, chrom=21, n_shards=1)
    qc = run_qc(cfg, chrom=21, n_shards=1)

    with open(qc.diagnostics_path, "r", encoding="utf-8") as handle:
        diag = json.load(handle)
    assert diag["within_split_gate_status"]["P3"]["gated"] is True
    p3_pair_status = [
        value
        for value in diag["fst_pairwise_gate_status"].values()
        if isinstance(value.get("pair"), list) and "P3" in value["pair"]
    ]
    assert p3_pair_status
    assert all(status["gated"] is True for status in p3_pair_status)
