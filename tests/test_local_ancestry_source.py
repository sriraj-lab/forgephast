"""Local ancestry source object tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forgephast.local_ancestry import (
    LOCAL_ANCESTRY_SOURCE_SCHEMA_VERSION,
    LocalAncestrySource,
    LocalAncestryTract,
    write_local_ancestry_source,
    write_local_ancestry_source_from_tract_rows,
)


def test_source_object_round_trips_rle_and_rasterizes(tmp_path: Path) -> None:
    source_path = tmp_path / "local_ancestry.zarr"
    tracts = {
        ("s1", 0): [
            LocalAncestryTract(0, 100, "EUR"),
            LocalAncestryTract(100, 200, "AFR"),
        ],
        ("s1", 1): [LocalAncestryTract(0, 200, "AFR")],
        ("s2", 0): [LocalAncestryTract(0, 200, "AFR")],
        ("s2", 1): [
            LocalAncestryTract(0, 50, "EUR"),
            LocalAncestryTract(50, 200, "AFR"),
        ],
    }

    write_local_ancestry_source(
        source_path,
        sample_ids=["s1", "s2"],
        tracts_by_sample_hap=tracts,
        ancestry_order=["EUR", "AFR"],
        chrom=22,
        metadata={"population": "ADMIX"},
    )

    source = LocalAncestrySource.open(source_path)
    assert source.chrom == "22"
    assert source.sample_ids == ["s1", "s2"]
    assert source.ancestry_order == ["EUR", "AFR"]
    assert source.tract_start_bp.shape[0] == 6

    codes = source.rasterize(
        sample_ids=["s2", "s1"],
        positions_bp=np.asarray([150, 10], dtype=np.int64),
    )

    assert codes.shape == (2, 2, 2)
    # Requested sample and variant order are preserved.
    assert codes[0, :, 0].tolist() == [1, 1]
    assert codes[0, :, 1].tolist() == [1, 0]
    assert codes[1, :, 0].tolist() == [1, 0]
    assert codes[1, :, 1].tolist() == [1, 1]


def test_source_object_remaps_to_requested_ancestry_order(tmp_path: Path) -> None:
    source_path = tmp_path / "local_ancestry.zarr"
    tracts = {
        ("s1", 0): [LocalAncestryTract(0, 200, "AFR")],
        ("s1", 1): [LocalAncestryTract(0, 200, "EUR")],
    }
    write_local_ancestry_source(
        source_path,
        sample_ids=["s1"],
        tracts_by_sample_hap=tracts,
        ancestry_order=["AFR", "EUR"],
        chrom=22,
    )

    source = LocalAncestrySource.open(source_path)
    codes = source.rasterize(
        sample_ids=["s1"],
        positions_bp=np.asarray([10, 150], dtype=np.int64),
        ancestry_to_index={"EUR": 0, "AFR": 1},
    )

    assert codes[:, :, 0].tolist() == [[1, 1]]
    assert codes[:, :, 1].tolist() == [[0, 0]]


def test_source_object_from_msprime_rows_records_schema(tmp_path: Path) -> None:
    source_path = tmp_path / "msprime.local_ancestry.zarr"
    rows = [
        {"haplotype_index": 0, "start_bp": 0.0, "end_bp": 100.0, "source_population": "EUR"},
        {"haplotype_index": 0, "start_bp": 100.0, "end_bp": 200.0, "source_population": "AFR"},
        {"haplotype_index": 1, "start_bp": 0.0, "end_bp": 200.0, "source_population": "AFR"},
        {"haplotype_index": 2, "start_bp": 0.0, "end_bp": 200.0, "source_population": "AFR"},
        {"haplotype_index": 3, "start_bp": 0.0, "end_bp": 50.0, "source_population": "EUR"},
        {"haplotype_index": 3, "start_bp": 50.0, "end_bp": 200.0, "source_population": "AFR"},
    ]

    write_local_ancestry_source_from_tract_rows(
        source_path,
        sample_ids=["ADMIX_syn1", "ADMIX_syn2"],
        tract_rows=rows,
        ancestry_order=["EUR", "AFR"],
        chrom=22,
        metadata={"population": "ADMIX"},
    )

    source = LocalAncestrySource.open(source_path)
    codes = source.rasterize(
        sample_ids=["ADMIX_syn1", "ADMIX_syn2"],
        positions_bp=np.asarray([10, 150], dtype=np.int64),
    )

    assert source_path.joinpath(".zattrs").exists()
    assert source.ancestry_order == ["EUR", "AFR"]
    assert codes[0, :, 0].tolist() == [0, 1]
    assert codes[0, :, 1].tolist() == [1, 1]
    assert codes[1, :, 0].tolist() == [1, 1]
    assert codes[1, :, 1].tolist() == [0, 1]

    import zarr

    group = zarr.open_group(str(source_path), mode="r")
    assert group.attrs["schema_version"] == LOCAL_ANCESTRY_SOURCE_SCHEMA_VERSION
    assert group.attrs["storage"] == "rle_per_haplotype"
    assert group.attrs["population"] == "ADMIX"


def test_source_object_errors_on_missing_coverage(tmp_path: Path) -> None:
    source_path = tmp_path / "local_ancestry.zarr"
    tracts = {
        ("s1", 0): [LocalAncestryTract(0, 50, "EUR")],
        ("s1", 1): [LocalAncestryTract(0, 50, "EUR")],
    }
    write_local_ancestry_source(
        source_path,
        sample_ids=["s1"],
        tracts_by_sample_hap=tracts,
        ancestry_order=["EUR", "AFR"],
        chrom=22,
    )

    source = LocalAncestrySource.open(source_path)
    with pytest.raises(ValueError, match="missing codes"):
        source.rasterize(
            sample_ids=["s1"],
            positions_bp=np.asarray([10, 60], dtype=np.int64),
        )
