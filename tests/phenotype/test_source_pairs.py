"""Tests for native HapStore-backed source-pair table generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forgephast.phenotype.causal.source_pairs import (
    SourcePairBuildConfig,
    build_source_pairs_from_hapstore,
    write_source_pairs_from_hapstore,
)


class FakeStore:
    def __init__(self, sample_ids, positions, vids, genotypes):
        self._sample_ids = [str(iid) for iid in sample_ids]
        self._positions = np.asarray(positions, dtype=np.int64)
        self._vids = [str(vid) for vid in vids]
        self._genotypes = np.asarray(genotypes, dtype=np.uint8)
        self.closed = False
        self.read_slices: list[tuple[int, int]] = []

    def sample_ids(self, chrom):
        return list(self._sample_ids)

    def variant_info(self, chrom):
        return {
            "pos": np.asarray(self._positions, dtype=np.int64),
            "vid": list(self._vids),
            "ref": ["A"] * len(self._positions),
            "alt": ["G"] * len(self._positions),
        }

    def genotypes(self, chrom, variant_slice=None, sample_mask=None):
        assert sample_mask is None
        start, end = variant_slice
        self.read_slices.append((int(start), int(end)))
        return np.asarray(self._genotypes[:, int(start) : int(end)], dtype=np.uint8)

    def has_chromosome(self, chrom):
        return True

    def attrs(self, chrom):
        return {"missing_sentinel": 255}

    def close(self):
        self.closed = True


def _write_samples(tmp_path: Path) -> Path:
    path = tmp_path / "samples.tsv"
    pd.DataFrame(
        {
            "IID": ["s1", "s2", "s3"],
            "population": ["EUR", "AFR", "AFR"],
            "source_population": ["EUR", "AFR_GULF", "AFR_EAST"],
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def _base_payload(tmp_path: Path, sample_path: Path) -> dict:
    return {
        "hapstore": {
            "layout": "per_chr",
            "root_dir": str(tmp_path / "stores"),
            "file_pattern": "chr{chrom}.hapstore.zarr",
            "missing_sentinel": 255,
        },
        "samples": {
            "file": str(sample_path),
            "id_column": "IID",
            "group_column": "population",
        },
        "chromosomes": [1],
        "groups": [
            {"name": "EUR", "column": "population", "value": "EUR", "maf_column": "maf_eur"},
            {"name": "AFR", "column": "population", "value": "AFR", "maf_column": "maf_afr"},
            {
                "name": "AFR_GULF",
                "column": "source_population",
                "value": "AFR_GULF",
                "filters": {"population": "AFR"},
                "maf_column": "maf_afr_gulf",
            },
        ],
        "source_pairs": {
            "pooled_maf_min": 0.30,
            "variant_block_size": 2,
        },
        "output": {
            "source_pairs_file": str(tmp_path / "source_pairs.tsv"),
            "qc_file": str(tmp_path / "source_pairs_qc.tsv"),
        },
    }


def test_build_source_pairs_per_chr_computes_group_maf_from_genotypes(tmp_path: Path) -> None:
    sample_path = _write_samples(tmp_path)
    config = SourcePairBuildConfig.from_dict(_base_payload(tmp_path, sample_path))
    store_path = tmp_path / "stores" / "chr1.hapstore.zarr"
    store = FakeStore(
        sample_ids=["s1", "s2", "s3", "extra"],
        positions=[10, 20, 30, 40],
        vids=["rs10", "rs20", "rs30", "rs40"],
        genotypes=[
            [0, 1, 2, 255],
            [2, 1, 0, 1],
            [0, 0, 0, 0],
            [2, 2, 2, 2],
        ],
    )

    frame, qc = build_source_pairs_from_hapstore(
        config,
        opener=lambda path: {store_path: store}[path],
    )

    assert frame["variant_id"].tolist() == ["chr1:10", "chr1:20", "chr1:30"]
    assert frame["chrom"].astype(str).tolist() == ["1", "1", "1"]
    assert frame["pos"].tolist() == [10, 20, 30]
    assert frame["source_variant_id"].tolist() == ["rs10", "rs20", "rs30"]
    assert np.allclose(frame["maf_pooled"].to_numpy(), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(frame["maf_eur"].to_numpy(), [0.0, 0.5, 0.0])
    assert np.allclose(frame["maf_afr"].to_numpy(), [0.5, 0.25, 0.0])
    assert np.allclose(frame["maf_afr_gulf"].to_numpy(), [0.0, 0.5, 0.0])
    assert store.read_slices == [(0, 2), (2, 4)]
    assert store.closed
    total = qc.loc[qc["scope"] == "total"].iloc[0]
    assert int(total["n_variants_total"]) == 4
    assert int(total["n_variants_written"]) == 3


def test_source_pairs_rejects_store_with_noncanonical_missing_sentinel(
    tmp_path: Path,
) -> None:
    sample_path = _write_samples(tmp_path)
    config = SourcePairBuildConfig.from_dict(_base_payload(tmp_path, sample_path))
    store_path = tmp_path / "stores" / "chr1.hapstore.zarr"
    store = FakeStore(
        sample_ids=["s1", "s2", "s3"],
        positions=[10],
        vids=["rs10"],
        genotypes=[[0], [1], [2]],
    )
    store.attrs = lambda chrom: {"missing_sentinel": 254}  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="missing_sentinel=254, expected 255"):
        build_source_pairs_from_hapstore(
            config,
            opener=lambda path: {store_path: store}[path],
        )


def test_source_pairs_rejects_noncanonical_configured_missing_sentinel(
    tmp_path: Path,
) -> None:
    payload = _base_payload(tmp_path, _write_samples(tmp_path))
    payload["hapstore"]["missing_sentinel"] = 254
    with pytest.raises(ValueError, match="missing_sentinel must be 255"):
        SourcePairBuildConfig.from_dict(payload)


def test_write_source_pairs_derives_population_groups_and_qc(tmp_path: Path) -> None:
    sample_path = _write_samples(tmp_path)
    payload = _base_payload(tmp_path, sample_path)
    payload.pop("groups")
    payload["source_pairs"]["pooled_maf_min"] = 0.0
    config = SourcePairBuildConfig.from_dict(payload)
    store_path = tmp_path / "stores" / "chr1.hapstore.zarr"
    store = FakeStore(
        sample_ids=["s1", "s2", "s3"],
        positions=[10, 20],
        vids=["rs10", "rs20"],
        genotypes=[
            [0, 1],
            [2, 1],
            [0, 0],
        ],
    )

    result = write_source_pairs_from_hapstore(
        config,
        opener=lambda path: {store_path: store}[path],
    )

    frame = pd.read_csv(result.source_pairs_file, sep="\t")
    qc = pd.read_csv(result.qc_file, sep="\t")
    assert {"maf_pooled", "maf_eur", "maf_afr"}.issubset(frame.columns)
    assert int(result.n_variants_total) == 2
    assert int(result.n_variants_written) == 2
    assert int(qc.loc[qc["scope"] == "total", "n_variants_written"].iloc[0]) == 2


def test_build_source_pairs_per_pop_per_chr_aggregates_stores(tmp_path: Path) -> None:
    sample_path = _write_samples(tmp_path)
    payload = _base_payload(tmp_path, sample_path)
    payload["hapstore"] = {
        "layout": "per_pop_per_chr",
        "root_dir": str(tmp_path / "stores"),
        "file_pattern": "{population}/chr{chrom}.hapstore.zarr",
    }
    payload["samples"]["store_population_column"] = "population"
    payload["source_pairs"]["pooled_maf_min"] = 0.0
    config = SourcePairBuildConfig.from_dict(payload)
    eur_path = tmp_path / "stores" / "EUR" / "chr1.hapstore.zarr"
    afr_path = tmp_path / "stores" / "AFR" / "chr1.hapstore.zarr"
    stores = {
        eur_path: FakeStore(
            sample_ids=["s1"],
            positions=[10, 20],
            vids=["rs10", "rs20"],
            genotypes=[[0, 1]],
        ),
        afr_path: FakeStore(
            sample_ids=["s2", "s3"],
            positions=[10, 20],
            vids=["rs10", "rs20"],
            genotypes=[[2, 1], [0, 0]],
        ),
    }

    frame, _qc = build_source_pairs_from_hapstore(config, opener=lambda path: stores[path])

    assert frame["variant_id"].tolist() == ["chr1:10", "chr1:20"]
    assert np.allclose(frame["maf_pooled"].to_numpy(), [1 / 3, 1 / 3])
    assert np.allclose(frame["maf_eur"].to_numpy(), [0.0, 0.5])
    assert np.allclose(frame["maf_afr"].to_numpy(), [0.5, 0.25])


def test_build_source_pairs_raises_when_selected_sample_missing(tmp_path: Path) -> None:
    sample_path = _write_samples(tmp_path)
    config = SourcePairBuildConfig.from_dict(_base_payload(tmp_path, sample_path))
    store_path = tmp_path / "stores" / "chr1.hapstore.zarr"
    store = FakeStore(
        sample_ids=["s1", "s2"],
        positions=[10],
        vids=["rs10"],
        genotypes=[[0], [2]],
    )

    with pytest.raises(ValueError, match="missing 1 selected sample"):
        build_source_pairs_from_hapstore(config, opener=lambda path: {store_path: store}[path])


def test_source_pair_config_rejects_unknown_keys(tmp_path: Path) -> None:
    sample_path = _write_samples(tmp_path)
    payload = _base_payload(tmp_path, sample_path)
    payload["source_pairs"]["variant_blok_size"] = 10
    with pytest.raises(ValueError, match="Unknown source_pairs config key"):
        SourcePairBuildConfig.from_dict(payload)


def test_source_pairs_reject_duplicate_positions(tmp_path: Path) -> None:
    sample_path = _write_samples(tmp_path)
    config = SourcePairBuildConfig.from_dict(_base_payload(tmp_path, sample_path))
    store_path = tmp_path / "stores" / "chr1.hapstore.zarr"
    store = FakeStore(
        sample_ids=["s1", "s2", "s3"],
        positions=[10, 10],
        vids=["rs10a", "rs10b"],
        genotypes=[[0, 1], [2, 1], [0, 0]],
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        build_source_pairs_from_hapstore(
            config,
            opener=lambda path: {store_path: store}[path],
        )
