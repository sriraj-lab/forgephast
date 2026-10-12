from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from forgephast.amplifier.config import MsprimePopulationConfig, MsprimeSourceSpec, from_yaml
from forgephast.amplifier.qc import fst_gate

from .conftest import write_test_config


def single_source_config(tmp_path):
    config = from_yaml(write_test_config(tmp_path, name="single_source_fst"))
    output = config.populations[2]
    output.engine = "msprime_tracts"
    output.selector = None
    output.msprime = MsprimePopulationConfig(sources=[MsprimeSourceSpec(population="P1", weight=1.0)])
    config.validate()
    return config


def test_single_source_baseline_resolves_source_panel_and_uniform_weights(tmp_path, monkeypatch):
    config = single_source_config(tmp_path)
    manifest = SimpleNamespace(
        donors_by_population={"P1": ["a", "b"], "P2": ["c", "d"], "P3": []},
        donor_weights_by_population={"P1": {"a": 0.9, "b": 0.1}, "P2": {}, "P3": {}},
    )
    loaded = []
    haps = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.uint8)

    def load(**kwargs):
        loaded.append((kwargs["source_path"], kwargs["sample_ids"]))
        return np.array([10, 20]), haps

    captured = {}
    def compare(**kwargs):
        captured.update(kwargs["counts_by_population"])
        return {"P1__P3": 0.1}

    monkeypatch.setattr(fst_gate, "pairwise_fst_from_counts_map_impl", compare)
    fst_gate.donor_baseline_fst_impl(
        config=config, chrom=21, pairs=[("P1", "P3")],
        load_manifest_fn=lambda cfg: manifest,
        resolve_pairwise_sets_fn=lambda *args: [],
        resolve_donor_panel_source_for_population_fn=lambda cfg, chrom, pop: ("hapstore", pop),
        load_sample_haplotypes_from_panel_fn=load,
        resolve_python_qc_numba_threads_fn=lambda cfg: 1,
    )
    assert loaded == [("P1", ["a", "b"]), ("P1", ["a", "b"])]
    ac, an = captured["P1"]
    np.testing.assert_allclose(ac / an, [0.1, 0.1])
    ac, an = captured["P3"]
    np.testing.assert_allclose(ac / an, [0.5, 0.5])


def test_empty_baseline_fails_before_reader(tmp_path):
    config = single_source_config(tmp_path)
    def must_not_load(**kwargs):
        pytest.fail("empty samples reached the reader")
    with pytest.raises(ValueError, match="No donor samples for FST baseline"):
        fst_gate.donor_baseline_fst_impl(
            config=config, chrom=21, pairs=[("P1", "P3")],
            load_manifest_fn=lambda cfg: SimpleNamespace(donors_by_population={"P1": [], "P3": []}, donor_weights_by_population={}),
            resolve_pairwise_sets_fn=lambda *args: [],
            resolve_donor_panel_source_for_population_fn=lambda *args: pytest.fail("empty panel resolution"),
            load_sample_haplotypes_from_panel_fn=must_not_load,
            resolve_python_qc_numba_threads_fn=lambda cfg: 1,
        )


def test_single_source_stochastic_gate_uses_source_donor_count(tmp_path):
    config = single_source_config(tmp_path)
    result = fst_gate.resolve_fst_stochastic_gate_activation_impl(
        config=config, groups={"P1": ["x", "y"], "P3": ["z", "w"]}, pairs=[("P1", "P3")],
        donors_by_population={"P1": ["a", "b"], "P3": []},
    )
    assert result["donor_sizes"] == {"P1": 2, "P3": 2}


def test_single_source_stochastic_gate_resolves_source_panel(tmp_path):
    config = single_source_config(tmp_path)
    config.qc.fst_stochastic_reps = 2
    loaded = []
    def load(**kwargs):
        loaded.append((kwargs["source_path"], kwargs["sample_ids"]))
        return np.array([10, 20]), np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    result = fst_gate.compute_stochastic_donor_fst_gate_impl(
        config=config, chrom=21, groups={"P1": ["x", "y"], "P3": ["z", "w"]},
        pairs=[("P1", "P3")], pairwise_fst={"P1__P3": 0.0},
        donors_by_population={"P1": ["a", "b"], "P3": []},
        donor_weights_by_population={}, sampled_diploids_by_population={"P1": 2, "P3": 2}, numba_threads=1,
        resolve_donor_panel_source_for_population_fn=lambda cfg, chrom, pop: ("hapstore", pop),
        load_sample_haplotypes_from_panel_fn=load,
    )
    assert loaded == [("P1", ["a", "b"]), ("P1", ["a", "b"])]
    assert result["available"] is True
