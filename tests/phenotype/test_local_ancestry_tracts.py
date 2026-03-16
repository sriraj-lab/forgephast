"""Local ancestry tract parsing and rasterization tests."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from forgephast.phenotype.local_ancestry.tracts import load_tracts_tsv_gz, rasterize_tracts_to_variant_ancestry


def _write_tracts(path: Path, rows: list[tuple[str, int, int, int, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("sample_id\thap_idx\tleft_bp\tright_bp\tancestry\n")
        for sid, hap_idx, left_bp, right_bp, anc in rows:
            handle.write(f"{sid}\t{hap_idx}\t{left_bp}\t{right_bp}\t{anc}\n")


def test_rasterize_tracts_assigns_expected_codes(tmp_path: Path) -> None:
    tracts_path = tmp_path / "tracts.tsv.gz"
    _write_tracts(
        tracts_path,
        rows=[
            ("s1", 0, 0, 100, "EUR"),
            ("s1", 0, 100, 200, "AFR"),
            ("s1", 1, 0, 200, "AFR"),
            ("s2", 0, 0, 200, "AFR"),
            ("s2", 1, 0, 50, "EUR"),
            ("s2", 1, 50, 200, "AFR"),
        ],
    )

    tracts = load_tracts_tsv_gz(tracts_path, sample_ids=["s1", "s2"], label_map={})
    ancestry_to_index = {"EUR": 0, "AFR": 1}
    positions = np.asarray([10, 150], dtype=np.int64)
    codes = rasterize_tracts_to_variant_ancestry(
        sample_ids=["s1", "s2"],
        tracts_by_sample_hap=tracts,
        variant_positions_bp=positions,
        ancestry_to_index=ancestry_to_index,
        chunk_n_samples=10,
    )

    assert codes.shape == (2, 2, 2)
    # s1 hap0: 10->EUR, 150->AFR; hap1: always AFR
    assert codes[0, :, 0].tolist() == [0, 1]
    assert codes[0, :, 1].tolist() == [1, 1]
    # s2 hap0 always AFR; hap1 switches at 50
    assert codes[1, :, 0].tolist() == [1, 1]
    assert codes[1, :, 1].tolist() == [0, 1]


def test_rasterize_tracts_errors_on_missing_coverage(tmp_path: Path) -> None:
    tracts_path = tmp_path / "tracts.tsv.gz"
    _write_tracts(
        tracts_path,
        rows=[
            ("s1", 0, 0, 50, "EUR"),
            ("s1", 1, 0, 50, "EUR"),
        ],
    )
    tracts = load_tracts_tsv_gz(tracts_path, sample_ids=["s1"], label_map={})
    ancestry_to_index = {"EUR": 0, "AFR": 1}
    positions = np.asarray([10, 60], dtype=np.int64)

    with pytest.raises(ValueError, match="missing codes"):
        rasterize_tracts_to_variant_ancestry(
            sample_ids=["s1"],
            tracts_by_sample_hap=tracts,
            variant_positions_bp=positions,
            ancestry_to_index=ancestry_to_index,
            chunk_n_samples=10,
        )


def test_load_tracts_supports_haplotype_index_schema(tmp_path: Path) -> None:
    tracts_path = tmp_path / "tracts.tsv.gz"
    header = (
        "population\tchrom\thaplotype_index\ttract_index\tstart_marker_idx\tend_marker_idx\t"
        "start_bp\tend_bp\tmarker_count\tbp_length\tsource_population\tdonor_haplotype_index\tcm_length\n"
    )
    rows = [
        # haplotype_index 0 (ADMIX_syn1 hap0): EUR then AFR
        ("ADMIX", 1, 0, 0, 0, 0, 0.0, 100.0, 0, 0.0, "EUR", 0, 0.0),
        ("ADMIX", 1, 0, 1, 0, 0, 100.0, 200.0, 0, 0.0, "AFR", 0, 0.0),
        # haplotype_index 1 (ADMIX_syn1 hap1): AFR
        ("ADMIX", 1, 1, 0, 0, 0, 0.0, 200.0, 0, 0.0, "AFR", 0, 0.0),
        # haplotype_index 2 (ADMIX_syn2 hap0): AFR
        ("ADMIX", 1, 2, 0, 0, 0, 0.0, 200.0, 0, 0.0, "AFR", 0, 0.0),
        # haplotype_index 3 (ADMIX_syn2 hap1): EUR then AFR
        ("ADMIX", 1, 3, 0, 0, 0, 0.0, 50.0, 0, 0.0, "EUR", 0, 0.0),
        ("ADMIX", 1, 3, 1, 0, 0, 50.0, 200.0, 0, 0.0, "AFR", 0, 0.0),
    ]

    with gzip.open(tracts_path, "wt", encoding="utf-8") as handle:
        handle.write(header)
        for row in rows:
            handle.write("\t".join(str(x) for x in row) + "\n")

    tracts = load_tracts_tsv_gz(tracts_path, sample_ids=["ADMIX_syn1", "ADMIX_syn2"], label_map={})
    ancestry_to_index = {"EUR": 0, "AFR": 1}
    positions = np.asarray([10, 150], dtype=np.int64)
    codes = rasterize_tracts_to_variant_ancestry(
        sample_ids=["ADMIX_syn1", "ADMIX_syn2"],
        tracts_by_sample_hap=tracts,
        variant_positions_bp=positions,
        ancestry_to_index=ancestry_to_index,
        chunk_n_samples=10,
    )

    assert codes.shape == (2, 2, 2)
    assert codes[0, :, 0].tolist() == [0, 1]
    assert codes[0, :, 1].tolist() == [1, 1]
    assert codes[1, :, 0].tolist() == [1, 1]
    assert codes[1, :, 1].tolist() == [0, 1]


def test_rasterize_tracts_numba_flag_matches_baseline(tmp_path: Path) -> None:
    tracts_path = tmp_path / "tracts.tsv.gz"
    _write_tracts(
        tracts_path,
        rows=[
            ("s1", 0, 0, 100, "EUR"),
            ("s1", 0, 100, 200, "AFR"),
            ("s1", 1, 0, 200, "AFR"),
            ("s2", 0, 0, 200, "AFR"),
            ("s2", 1, 0, 50, "EUR"),
            ("s2", 1, 50, 200, "AFR"),
        ],
    )
    tracts = load_tracts_tsv_gz(tracts_path, sample_ids=["s1", "s2"], label_map={})
    ancestry_to_index = {"EUR": 0, "AFR": 1}
    positions = np.asarray([10, 150], dtype=np.int64)

    baseline = rasterize_tracts_to_variant_ancestry(
        sample_ids=["s1", "s2"],
        tracts_by_sample_hap=tracts,
        variant_positions_bp=positions,
        ancestry_to_index=ancestry_to_index,
        chunk_n_samples=10,
        use_numba=False,
    )
    accelerated = rasterize_tracts_to_variant_ancestry(
        sample_ids=["s1", "s2"],
        tracts_by_sample_hap=tracts,
        variant_positions_bp=positions,
        ancestry_to_index=ancestry_to_index,
        chunk_n_samples=10,
        use_numba=True,
    )

    assert np.array_equal(baseline, accelerated)
