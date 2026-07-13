from __future__ import annotations

import yaml
import pytest

from forgephast.amplifier.config import from_yaml

from .conftest import write_test_config



def test_config_parses_k_populations(tmp_path):
    cfg_path = write_test_config(tmp_path)
    cfg = from_yaml(cfg_path)

    assert len(cfg.populations) == 3
    assert [p.name for p in cfg.populations] == ["P1", "P2", "P3"]
    assert cfg.simulation.chromosomes == [21]


def test_config_rejects_donor_only_msprime_population(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][0].update(
        {
            "donor_only": True,
            "engine": "msprime_tracts",
            "selector": None,
            "msprime": {
                "sources": [{"population": "P2", "weight": 1.0}],
                "admixture_time_generations": None,
            },
        }
    )
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="donor_only requires engine='mosaic'"):
        from_yaml(cfg_path)



def test_config_rejects_empty_population_list(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"] = []

    broken = tmp_path / "broken.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    try:
        from_yaml(broken)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "populations cannot be empty" in str(exc)


def test_config_rejects_removed_synthetic_rr_match_percentage_key(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["synthetic_round_robin_match_percentage"] = 1.0

    broken = tmp_path / "broken_match_percentage.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="Invalid qc config keys"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_synthetic_rr_max_pairs(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["synthetic_round_robin_max_pairs_per_population"] = 0

    broken = tmp_path / "broken_max_pairs.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="synthetic_round_robin_max_pairs_per_population was removed"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_synthetic_rr_min_cpus_per_match(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["synthetic_rr_min_cpus_per_match"] = 0

    broken = tmp_path / "broken_pair_parallel_unset_min_cpu_set.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="synthetic_rr_min_cpus_per_match was removed"):
        _ = from_yaml(broken)


def test_config_rejects_removed_rr_pairwise_mode_key(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["rr_pairwise_mode"] = "sampled"

    broken = tmp_path / "broken_rr_pairwise_mode.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="qc.rr_pairwise_mode was removed"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_synthetic_shard_round_robin_pairwise_mode(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["synthetic_shard_round_robin_pairwise_mode"] = "bad_mode"

    broken = tmp_path / "broken_synthetic_shard_round_robin_pairwise_mode.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="synthetic_shard_round_robin_pairwise_mode"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_synthetic_shard_round_robin_max_pairs(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["synthetic_shard_round_robin_max_pairs_per_population"] = 0

    broken = tmp_path / "broken_synthetic_shard_round_robin_max_pairs.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="synthetic_shard_round_robin_max_pairs_per_population"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_synthetic_shard_round_robin_min_cpus_per_match(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["synthetic_shard_round_robin_min_cpus_per_match"] = 0

    broken = tmp_path / "broken_synthetic_shard_round_robin_min_cpus_per_match.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="synthetic_shard_round_robin_min_cpus_per_match"):
        _ = from_yaml(broken)


@pytest.mark.parametrize(
    "removed_key",
    [
        "ld_decay_sample_size_per_population",
        "ld_decay_max_variants",
        "plink2_parity_max_variants",
        "plink2_parity_pca_components",
        "plink_threads",
        "plink2_parity_enabled",
        "plink2_parity_max_pairs_per_population",
    ],
)
def test_config_rejects_removed_qc_keys(tmp_path, removed_key: str):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})[removed_key] = 1

    broken = tmp_path / f"broken_{removed_key}.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="Invalid qc config keys"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_ld_decay_backend(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["ld_decay_backend"] = "not_a_backend"

    broken = tmp_path / "broken_ld_decay_backend.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="ld_decay_backend"):
        _ = from_yaml(broken)


@pytest.mark.parametrize("backend", ["plink_counts", "plink_native_fst"])
def test_config_rejects_removed_qc_backends(tmp_path, backend: str):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["backend"] = backend

    broken = tmp_path / f"broken_backend_{backend}.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="qc.backend must be one of"):
        _ = from_yaml(broken)


def test_config_rejects_removed_plink_ld_decay_backend(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["ld_decay_backend"] = "plink"

    broken = tmp_path / "broken_ld_decay_backend_plink.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="qc.ld_decay_backend must be one of"):
        _ = from_yaml(broken)


def test_config_rejects_removed_ld_decay_auc_key(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["ld_decay_auc_rel_diff_floor"] = 0.01

    broken = tmp_path / "broken_ld_decay_auc_rel_diff_floor.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="Invalid qc config keys"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_main_pairwise_mode(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["main_pairwise_mode"] = "bad_mode"

    broken = tmp_path / "broken_main_pairwise_mode.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="main_pairwise_mode"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_numba_threads(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["numba_threads"] = 0

    broken = tmp_path / "broken_numba_threads.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="numba_threads"):
        _ = from_yaml(broken)


def test_config_rejects_invalid_fst_stochastic_gate_mode(tmp_path):
    cfg_path = write_test_config(tmp_path, name="bad_fst_stochastic_mode")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["fst_stochastic_gate_mode"] = "bad_mode"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="fst_stochastic_gate_mode"):
        _ = from_yaml(cfg_path)


def test_config_rejects_invalid_fst_stochastic_small_pop_fraction_threshold(tmp_path):
    cfg_path = write_test_config(tmp_path, name="bad_fst_stochastic_threshold")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})["fst_stochastic_small_pop_fraction_threshold"] = 1.2
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="fst_stochastic_small_pop_fraction_threshold"):
        _ = from_yaml(cfg_path)


def test_config_rejects_all_outputs_disabled(tmp_path):
    cfg_path = write_test_config(tmp_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("outputs", {})["write_phased_vcf"] = False
    payload.setdefault("outputs", {})["write_pgen"] = False
    payload.setdefault("outputs", {})["write_hapstore"] = False

    broken = tmp_path / "broken_all_outputs_disabled.yaml"
    with open(broken, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    with pytest.raises(ValueError, match="write_hapstore"):
        _ = from_yaml(broken)


def test_config_accepts_hapstore_2d_chunking_keys(tmp_path):
    cfg_path = write_test_config(tmp_path, name="hap_chunks_ok")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    outputs = payload.setdefault("outputs", {})
    outputs["hapstore_chunk_haplotype_size"] = 15000
    outputs["hapstore_chunk_sample_size"] = 7500
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    assert cfg.outputs.hapstore_chunk_haplotype_size == 15000
    assert cfg.outputs.hapstore_chunk_sample_size == 7500


def test_config_rejects_removed_hapstore_merge_strategy_key(tmp_path):
    cfg_path = write_test_config(tmp_path, name="hap_merge_strategy_removed")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("outputs", {})["hapstore_merge_strategy"] = "legacy_materialize"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="Invalid outputs config keys"):
        _ = from_yaml(cfg_path)


def test_config_accepts_hapstore_only_donor_input(tmp_path):
    cfg_path = write_test_config(tmp_path, name="hapstore_only_donor_input")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["input"].pop("panel_vcf_pattern", None)
    payload["input"]["panel_hapstore_pattern"] = str(tmp_path / "panel" / "chr{chrom}.hapstore.zarr")
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    assert cfg.input.panel_vcf_pattern is None
    assert cfg.input.panel_hapstore_pattern == str(tmp_path / "panel" / "chr{chrom}.hapstore.zarr")


def test_config_rejects_missing_all_donor_panel_patterns(tmp_path):
    cfg_path = write_test_config(tmp_path, name="missing_panel_patterns")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["input"].pop("panel_vcf_pattern", None)
    payload["input"].pop("panel_hapstore_pattern", None)
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="at least one donor panel pattern"):
        _ = from_yaml(cfg_path)


def test_config_rejects_non_sharded_hapstore_pattern_when_required(tmp_path):
    cfg_path = write_test_config(tmp_path, name="nonsharded_hapstore_required")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["input"].pop("panel_vcf_pattern", None)
    payload["input"]["panel_hapstore_pattern"] = str(tmp_path / "panel" / "fixed.hapstore.zarr")
    payload["input"]["require_chrom_sharded_panel"] = True
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="panel_hapstore_pattern"):
        _ = from_yaml(cfg_path)


@pytest.mark.parametrize(
    "field_name",
    [
        "hapstore_chunk_haplotype_size",
        "hapstore_chunk_sample_size",
    ],
)
def test_config_rejects_invalid_hapstore_2d_chunking_values(tmp_path, field_name: str):
    cfg_path = write_test_config(tmp_path, name=f"hap_chunks_bad_{field_name}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("outputs", {})[field_name] = 0
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match=field_name):
        _ = from_yaml(cfg_path)


@pytest.mark.parametrize(
    "removed_key",
    [
        "pgen_source_preference",
        "merge_backend",
        "bcftools_path",
        "bcftools_threads",
        "bcftools_no_index",
        "plink1_path",
    ],
)
def test_config_rejects_removed_output_keys(tmp_path, removed_key: str):
    cfg_path = write_test_config(tmp_path, name=f"{removed_key}_removed")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("outputs", {})[removed_key] = "removed"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match=removed_key):
        _ = from_yaml(cfg_path)


def test_config_rejects_invalid_hapstore_vcf_n_shards(tmp_path):
    cfg_path = write_test_config(tmp_path, name="hapstore_vcf_n_shards_bad")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("outputs", {})["hapstore_vcf_n_shards"] = 0
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="hapstore_vcf_n_shards"):
        _ = from_yaml(cfg_path)


def test_config_rejects_empty_hapstore_vcf_plink2_location(tmp_path):
    cfg_path = write_test_config(tmp_path, name="hapstore_vcf_plink2_location_bad")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("outputs", {})["hapstore_vcf_plink2_location"] = ""
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="hapstore_vcf_plink2_location"):
        _ = from_yaml(cfg_path)


def test_config_accepts_simulation_pool_defaults(tmp_path):
    cfg_path = write_test_config(tmp_path, name="pool_defaults")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("simulation", {})["n_shards"] = 64
    payload.setdefault("simulation", {})["task_workers"] = 64
    payload.setdefault("simulation", {})["task_start_method"] = "fork"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    assert cfg.simulation.n_shards == 64
    assert cfg.simulation.task_workers == 64


def test_config_rejects_invalid_simulation_pool_values(tmp_path):
    cfg_path = write_test_config(tmp_path, name="pool_invalid")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("simulation", {})["n_shards"] = 0
    payload.setdefault("simulation", {})["task_workers"] = -1
    payload.setdefault("simulation", {})["task_start_method"] = "bad"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="simulation\\.(n_shards|task_workers|task_start_method)"):
        _ = from_yaml(cfg_path)


def test_config_rejects_population_with_invalid_engine(tmp_path):
    cfg_path = write_test_config(tmp_path, name="bad_engine")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][0]["engine"] = "not_a_real_engine"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="engine must be one of"):
        _ = from_yaml(cfg_path)


def test_config_rejects_mosaic_population_without_selector(tmp_path):
    cfg_path = write_test_config(tmp_path, name="mosaic_selector_missing")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][0]["selector"] = None
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="selector is required for engine='mosaic'"):
        _ = from_yaml(cfg_path)


def test_config_rejects_selector_weights_that_do_not_sum_to_one(tmp_path):
    cfg_path = write_test_config(tmp_path, name="selector_weights_bad_sum")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][0]["selector_weights"] = {"POP_A": 0.7, "POP_B": 0.1}
    payload["populations"][0]["selector"] = {"values": ["POP_A", "POP_B"]}
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="selector_weights must sum to 1.0"):
        _ = from_yaml(cfg_path)


def test_config_rejects_msprime_population_without_msprime_block(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_missing_block")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][0]["engine"] = "msprime_tracts"
    payload["populations"][0]["selector"] = None
    payload["populations"][0]["msprime"] = None
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="msprime config is required"):
        _ = from_yaml(cfg_path)


def test_config_rejects_msprime_weights_not_summing_to_one(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_bad_weights")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [
                {"population": "P1", "weight": 0.8},
                {"population": "P2", "weight": 0.3},
            ],
            "admixture_time_generations": 8.0,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        _ = from_yaml(cfg_path)


def test_config_accepts_msprime_ancestry_source_zarr_flag(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_source_zarr")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [
                {"population": "P1", "weight": 0.6},
                {"population": "P2", "weight": 0.4},
            ],
            "admixture_time_generations": 8.0,
            "write_ancestry_source_zarr": True,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)

    assert cfg.populations[2].msprime is not None
    assert cfg.populations[2].msprime.write_ancestry_source_zarr is True


def test_config_rejects_msprime_source_population_not_defined(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_unknown_source")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [{"population": "DOES_NOT_EXIST", "weight": 1.0}],
            "admixture_time_generations": None,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="is not defined"):
        _ = from_yaml(cfg_path)


def test_config_rejects_msprime_source_population_that_is_not_mosaic(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_non_mosaic_source")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][1] = {
        "name": "P2",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [{"population": "P1", "weight": 1.0}],
            "admixture_time_generations": None,
        },
    }
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [{"population": "P2", "weight": 1.0}],
            "admixture_time_generations": None,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="must use engine='mosaic'"):
        _ = from_yaml(cfg_path)


def test_config_rejects_admixture_time_for_single_source_msprime(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_single_source_bad_time")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [{"population": "P1", "weight": 1.0}],
            "admixture_time_generations": 5.0,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="must be null when len\\(msprime.sources\\) == 1"):
        _ = from_yaml(cfg_path)


def test_config_rejects_missing_admixture_time_for_multi_source_msprime(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_multi_source_missing_time")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [
                {"population": "P1", "weight": 0.5},
                {"population": "P2", "weight": 0.5},
            ],
            "admixture_time_generations": None,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="admixture_time_generations is required"):
        _ = from_yaml(cfg_path)


def test_config_msprime_end_time_defaults_to_10(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_end_time_default")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [
                {"population": "P1", "weight": 0.5},
                {"population": "P2", "weight": 0.5},
            ],
            "admixture_time_generations": 8.0,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    assert cfg.populations[2].msprime is not None
    assert cfg.populations[2].msprime.end_time_generations == 10.0


def test_config_rejects_nonpositive_msprime_end_time(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_bad_end_time")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [
                {"population": "P1", "weight": 0.5},
                {"population": "P2", "weight": 0.5},
            ],
            "admixture_time_generations": 8.0,
            "end_time_generations": 0.0,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="end_time_generations must be > 0"):
        _ = from_yaml(cfg_path)


def test_config_rejects_msprime_end_time_before_admixture_time(tmp_path):
    cfg_path = write_test_config(tmp_path, name="msprime_end_before_admixture")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["populations"][2] = {
        "name": "P3",
        "engine": "msprime_tracts",
        "selector": None,
        "n_diploids": 4,
        "msprime": {
            "sources": [
                {"population": "P1", "weight": 0.5},
                {"population": "P2", "weight": 0.5},
            ],
            "admixture_time_generations": 8.0,
            "end_time_generations": 7.0,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="end_time_generations must be >= msprime.admixture_time_generations"):
        _ = from_yaml(cfg_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("admixed_qc_mode", "bad_mode", "admixed_qc_mode"),
        ("admixed_hwe_inbreeding_mode", "bad_mode", "admixed_hwe_inbreeding_mode"),
        ("admixed_tract_length_ks_max", -0.01, "admixed_tract_length_ks_max"),
        ("admixed_tract_length_ks_max", 1.01, "admixed_tract_length_ks_max"),
        ("admixed_tract_length_min_tracts", 0, "admixed_tract_length_min_tracts"),
        ("admixed_ancestry_variance_rel_tolerance", -0.1, "admixed_ancestry_variance_rel_tolerance"),
        ("admixed_min_individuals_for_variance_gate", 0, "admixed_min_individuals_for_variance_gate"),
    ],
)
def test_config_rejects_invalid_admixed_qc_fields(tmp_path, field: str, value: object, match: str):
    cfg_path = write_test_config(tmp_path, name=f"bad_admixed_qc_{field}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {})[field] = value
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match=match):
        _ = from_yaml(cfg_path)


def test_config_accepts_explicit_admixed_qc_fields(tmp_path):
    cfg_path = write_test_config(tmp_path, name="admixed_qc_fields_ok")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload.setdefault("qc", {}).update(
        {
            "admixed_qc_mode": "strict",
            "admixed_skip_donor_baseline_pairs": True,
            "admixed_tract_length_gate_enabled": True,
            "admixed_tract_length_ks_max": 0.12,
            "admixed_tract_length_min_tracts": 100,
            "admixed_ancestry_variance_gate_enabled": True,
            "admixed_ancestry_variance_rel_tolerance": 0.5,
            "admixed_min_individuals_for_variance_gate": 10,
            "admixed_hwe_inbreeding_mode": "legacy",
            "admixed_diag_pca_cline_enabled": False,
            "admixed_diag_king_enabled": False,
            "admixed_diag_stratified_ld_enabled": False,
        }
    )
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    assert cfg.qc.admixed_qc_mode == "strict"
    assert cfg.qc.admixed_hwe_inbreeding_mode == "legacy"
    assert cfg.qc.admixed_tract_length_min_tracts == 100


def test_config_rejects_unknown_top_level_key(tmp_path):
    cfg_path = write_test_config(tmp_path, name="unknown_top_level")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["simulaton"] = {}
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    with pytest.raises(ValueError, match="Unknown top-level config key"):
        from_yaml(cfg_path)
