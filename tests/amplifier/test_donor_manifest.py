from __future__ import annotations

import pandas as pd

from forgephast.amplifier.config import MsprimePopulationConfig, MsprimeSourceSpec, PopulationSpec, from_yaml
from forgephast.amplifier.donors import build_manifest_from_config

from .conftest import write_test_config



def test_manifest_selectors_and_counts(tmp_path):
    cfg_path = write_test_config(tmp_path)
    cfg = from_yaml(cfg_path)
    manifest = build_manifest_from_config(cfg)

    counts = pd.read_csv(manifest.donor_counts_path, sep="\t")
    count_map = dict(zip(counts["population"], counts["n_donors"]))

    assert count_map["P1"] == 2
    assert count_map["P2"] == 2
    assert count_map["P3"] == 2

    df = pd.read_csv(manifest.manifest_path, sep="\t")
    assert set(df.columns) == {"sample_id", "population"}
    assert df["sample_id"].nunique() == 6


def test_manifest_allows_msprime_target_with_zero_direct_donors(tmp_path):
    cfg_path = write_test_config(tmp_path)
    cfg = from_yaml(cfg_path)
    cfg.populations.append(
        PopulationSpec(
            name="ADMIX",
            n_diploids=4,
            selector=None,
            engine="msprime_tracts",
            msprime=MsprimePopulationConfig(
                sources=[
                    MsprimeSourceSpec(population="P1", weight=0.6),
                    MsprimeSourceSpec(population="P2", weight=0.4),
                ],
                admixture_time_generations=8.0,
            ),
        )
    )
    manifest = build_manifest_from_config(cfg)

    counts = pd.read_csv(manifest.donor_counts_path, sep="\t")
    count_map = dict(zip(counts["population"], counts["n_donors"]))
    assert count_map["ADMIX"] == 0
    assert manifest.donors_by_population["ADMIX"] == []
