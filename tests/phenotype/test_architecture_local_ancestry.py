"""Architecture local-ancestry-aware genetic component tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgephast.phenotype.architectures.polygenic_no_pleiotropy import PolygenicNoPleiotropyPlugin
from forgephast.phenotype.architectures.base import ArchitectureInput


def test_polygenic_no_pleiotropy_genotype_matrix_fast_path_matches_dataframe_path() -> None:
    plugin = PolygenicNoPleiotropyPlugin()
    sample_ids = ["eur1", "afr1", "eur2"]
    sample_populations = ["EUR", "AFR", "EUR"]
    variants = ["chr21:10", "chr21:20"]

    membership_df = pd.DataFrame(
        {
            "variant_id": variants,
            "maf_bucket": ["b1", "b2"],
            "membership_type": ["fixed", "subset"],
            "replicate": [1, 1],
        }
    )
    beta_df = pd.DataFrame(
        [
            {"variant_id": "chr21:10", "population": "EUR", "beta": 2.0},
            {"variant_id": "chr21:10", "population": "AFR", "beta": 5.0},
            {"variant_id": "chr21:20", "population": "EUR", "beta": 1.0},
            {"variant_id": "chr21:20", "population": "AFR", "beta": 3.0},
        ]
    )
    genotype_matrix = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    dosage_df = pd.DataFrame(genotype_matrix, columns=variants)
    dosage_df.insert(0, "IID", sample_ids)
    dosage_df.insert(0, "FID", sample_ids)

    baseline_genetic, baseline_effects = plugin.compute_genetic_component(
        ArchitectureInput(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            dosage_df=dosage_df,
            membership_df=membership_df,
            beta_df=beta_df,
            variant_id_column="variant_id",
        )
    )
    fast_genetic, fast_effects = plugin.compute_genetic_component(
        ArchitectureInput(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            dosage_df=dosage_df[["FID", "IID"]].copy(),
            genotype_matrix=genotype_matrix,
            membership_df=membership_df,
            beta_df=beta_df,
            variant_id_column="variant_id",
        )
    )

    assert np.allclose(baseline_genetic, fast_genetic, atol=1e-12, rtol=0.0)
    pd.testing.assert_frame_equal(baseline_effects, fast_effects)


def test_polygenic_no_pleiotropy_local_ancestry_component_matches_hand_calc() -> None:
    plugin = PolygenicNoPleiotropyPlugin()
    sample_ids = ["adm1"]
    variants = ["chr21:10", "chr21:20"]

    membership_df = pd.DataFrame(
        {
            "variant_id": variants,
            "maf_bucket": ["b1", "b1"],
            "membership_type": ["fixed", "fixed"],
            "replicate": [1, 1],
        }
    )

    beta_df = pd.DataFrame(
        [
            {"variant_id": "chr21:10", "population": "EUR", "beta": 2.0},
            {"variant_id": "chr21:10", "population": "AFR", "beta": 10.0},
            {"variant_id": "chr21:20", "population": "EUR", "beta": 1.0},
            {"variant_id": "chr21:20", "population": "AFR", "beta": 3.0},
        ]
    )

    haplotypes = np.asarray([[[1, 0], [1, 1]]], dtype=np.uint8)  # (1 sample, 2 variants, 2 haps)
    local_ancestry_codes = np.asarray([[[0, 1], [1, 1]]], dtype=np.int16)  # EUR then AFR
    # dosage is only used for schema checks in the plugin; construct consistent values
    dosage = haplotypes[:, :, 0].astype(float) + haplotypes[:, :, 1].astype(float)
    dosage_df = pd.DataFrame(dosage, columns=variants)
    dosage_df.insert(0, "IID", sample_ids)
    dosage_df.insert(0, "FID", sample_ids)

    genetic, _ = plugin.compute_genetic_component(
        ArchitectureInput(
            sample_ids=sample_ids,
            sample_populations=["ADMIX"],
            dosage_df=dosage_df,
            haplotypes=haplotypes,
            local_ancestry_codes=local_ancestry_codes,
            ancestry_order=["EUR", "AFR"],
            la_chunk_n_samples=10,
            la_chunk_n_variants=10,
            membership_df=membership_df,
            beta_df=beta_df,
            variant_id_column="variant_id",
        )
    )

    # Hand calc:
    # chr21:10: hap0 EUR (1*2) + hap1 AFR (0*10) = 2
    # chr21:20: hap0 AFR (1*3) + hap1 AFR (1*3) = 6
    # total = 8
    assert float(genetic[0]) == 8.0


def test_polygenic_local_ancestry_numba_flag_matches_baseline() -> None:
    plugin = PolygenicNoPleiotropyPlugin()
    sample_ids = ["adm1", "adm2"]
    variants = ["chr21:10", "chr21:20"]

    membership_df = pd.DataFrame(
        {
            "variant_id": variants,
            "maf_bucket": ["b1", "b1"],
            "membership_type": ["fixed", "fixed"],
            "replicate": [1, 1],
        }
    )
    beta_df = pd.DataFrame(
        [
            {"variant_id": "chr21:10", "population": "EUR", "beta": 2.0},
            {"variant_id": "chr21:10", "population": "AFR", "beta": 10.0},
            {"variant_id": "chr21:20", "population": "EUR", "beta": 1.0},
            {"variant_id": "chr21:20", "population": "AFR", "beta": 3.0},
        ]
    )
    haplotypes = np.asarray(
        [
            [[1, 0], [1, 1]],
            [[0, 1], [1, 0]],
        ],
        dtype=np.uint8,
    )
    local_ancestry_codes = np.asarray(
        [
            [[0, 1], [1, 1]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.int16,
    )
    dosage = haplotypes[:, :, 0].astype(float) + haplotypes[:, :, 1].astype(float)
    dosage_df = pd.DataFrame(dosage, columns=variants)
    dosage_df.insert(0, "IID", sample_ids)
    dosage_df.insert(0, "FID", sample_ids)

    base_input = dict(
        sample_ids=sample_ids,
        sample_populations=["ADMIX", "ADMIX"],
        dosage_df=dosage_df,
        haplotypes=haplotypes,
        local_ancestry_codes=local_ancestry_codes,
        ancestry_order=["EUR", "AFR"],
        la_chunk_n_samples=10,
        la_chunk_n_variants=10,
        membership_df=membership_df,
        beta_df=beta_df,
        variant_id_column="variant_id",
    )

    baseline, _ = plugin.compute_genetic_component(ArchitectureInput(**base_input, use_numba=False))
    accelerated, _ = plugin.compute_genetic_component(ArchitectureInput(**base_input, use_numba=True))
    assert np.allclose(baseline, accelerated, atol=1e-12, rtol=0.0)
