"""HapStore adapter unit tests with in-memory fake stores."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forgephast.phenotype.config import HapStoreInputConfig
from forgephast.phenotype.data.hapstore_adapter import HapStoreAdapter


class FakeStore:
    def __init__(self, sample_ids, positions, refs, alts, genotypes):
        self._sample_ids = list(sample_ids)
        self._positions = np.asarray(positions, dtype=np.int64)
        self._refs = list(refs)
        self._alts = list(alts)
        self._genotypes = np.asarray(genotypes, dtype=np.uint8)
        self._haplotypes = self._build_haplotypes_from_genotypes(self._genotypes)
        self.genotype_sample_masks: list[np.ndarray | None] = []
        self.haplotype_sample_masks: list[np.ndarray | None] = []

    @staticmethod
    def _build_haplotypes_from_genotypes(genotypes: np.ndarray) -> np.ndarray:
        g = np.asarray(genotypes, dtype=np.uint8)
        n_samples, n_variants = g.shape
        out = np.zeros((2 * n_samples, n_variants), dtype=np.uint8)
        missing = g == 255
        out[0::2, :] = np.where(missing, 255, np.where(g >= 1, 1, 0)).astype(np.uint8)
        out[1::2, :] = np.where(missing, 255, np.where(g == 2, 1, 0)).astype(np.uint8)
        return out

    def sample_ids(self, chrom):
        return list(self._sample_ids)

    def variant_positions(self, chrom):
        return np.asarray(self._positions, dtype=np.int64)

    def variant_info(self, chrom):
        return {
            "pos": np.asarray(self._positions, dtype=np.int64),
            "ref": list(self._refs),
            "alt": list(self._alts),
        }

    def attrs(self, chrom):
        return {"missing_sentinel": 255}

    def genotypes(self, chrom, variant_slice=None, sample_mask=None):
        self.genotype_sample_masks.append(
            None if sample_mask is None else np.asarray(sample_mask, dtype=bool).copy()
        )
        start, end = variant_slice
        out = self._genotypes[:, int(start):int(end)]
        if sample_mask is not None:
            out = out[np.asarray(sample_mask, dtype=bool)]
        return np.asarray(out, dtype=np.uint8)

    def haplotypes(self, chrom, variant_slice=None, sample_mask=None):
        self.haplotype_sample_masks.append(
            None if sample_mask is None else np.asarray(sample_mask, dtype=bool).copy()
        )
        start, end = variant_slice
        out = self._haplotypes[:, int(start):int(end)]
        if sample_mask is not None:
            sample_indices = np.flatnonzero(np.asarray(sample_mask, dtype=bool))
            hap_indices = np.empty(sample_indices.size * 2, dtype=np.int64)
            hap_indices[0::2] = 2 * sample_indices
            hap_indices[1::2] = (2 * sample_indices) + 1
            out = out[hap_indices, :]
        return np.asarray(out, dtype=np.uint8)

    def n_variants(self, chrom):
        return int(self._genotypes.shape[1])


@pytest.fixture
def fake_adapter() -> HapStoreAdapter:
    cfg = HapStoreInputConfig(
        root_dir="/tmp/hapstores",
        file_pattern="{population}/chr{chrom}.hapstore.zarr",
        missing_sentinel=255,
        missing_policy="fill",
        missing_fill_value=0.25,
    )
    return HapStoreAdapter(cfg)


def _wire_fake_stores(adapter: HapStoreAdapter, stores: dict[tuple[str, int], FakeStore]) -> None:
    adapter._open_store = lambda population, chrom: stores[(str(population), int(chrom))]  # type: ignore[method-assign]


def test_export_to_pgen_passes_native_filter_and_scratch_settings(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_export_pgen(**kwargs):
        calls.append(dict(kwargs))
        prefix = Path(str(kwargs["out_prefix"]))
        Path(f"{prefix}.pgen").write_bytes(b"pgen")
        Path(f"{prefix}.pvar").write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n1\t10\tv1\tA\tG\n",
            encoding="utf-8",
        )
        Path(f"{prefix}.psam").write_text(
            "#FID\tIID\n0\ts1\n0\ts2\n",
            encoding="utf-8",
        )
        return prefix

    monkeypatch.setattr("hapstore.pgen.export_pgen", fake_export_pgen)
    adapter = HapStoreAdapter(
        HapStoreInputConfig(
            layout="single_store",
            store_path=str(tmp_path / "source.zarr"),
        )
    )
    scratch = tmp_path / "scratch"
    out_prefix = tmp_path / "cohort.maf_gt0"
    result = adapter.export_to_pgen(
        sample_ids=["s1", "s2"],
        sample_populations=["EUR", "EUR"],
        out_prefix=out_prefix,
        chromosomes=[1],
        export_cfg=SimpleNamespace(
            plink2_path="plink2",
            threads=3,
            concat_workers=2,
            pgen_shard_workers=4,
            filter_monomorphic=True,
            min_maf=0.01,
            shard_tmp_dir=str(scratch),
        ),
    )

    assert result == str(out_prefix)
    assert len(calls) == 1
    assert calls[0]["filter_monomorphic"] is True
    assert calls[0]["min_maf"] == 0.01
    assert calls[0]["shard_tmp_dir"] == str(scratch)
    assert calls[0]["n_workers"] == 4
    assert calls[0]["plink2_workers"] == 3


def test_export_dosages_reads_per_pop_per_chr_and_tracks_alt(fake_adapter: HapStoreAdapter) -> None:
    stores = {
        ("EUR", 1): FakeStore(
            sample_ids=["id1", "id3"],
            positions=[10, 20, 30],
            refs=["A", "C", "G"],
            alts=["G", "T", "A"],
            genotypes=[[0, 1, 255], [2, 0, 1]],
        ),
        ("AFR", 1): FakeStore(
            sample_ids=["id2"],
            positions=[10, 20, 30],
            refs=["A", "C", "G"],
            alts=["G", "T", "A"],
            genotypes=[[1, 2, 0]],
        ),
        ("EUR", 2): FakeStore(
            sample_ids=["id1", "id3"],
            positions=[15],
            refs=["T"],
            alts=["C"],
            genotypes=[[1], [2]],
        ),
        ("AFR", 2): FakeStore(
            sample_ids=["id2"],
            positions=[15],
            refs=["T"],
            alts=["C"],
            genotypes=[[0]],
        ),
    }
    _wire_fake_stores(fake_adapter, stores)

    out = fake_adapter.export_dosages(
        sample_ids=["id1", "id2", "id3"],
        sample_populations=["EUR", "AFR", "EUR"],
        variant_ids=["chr1:10", "chr1:30", "chr2:15"],
        out_dir=Path("/tmp"),
    )

    assert list(out.columns) == ["FID", "IID", "chr1:10", "chr1:30", "chr2:15"]
    assert out["IID"].tolist() == ["id1", "id2", "id3"]
    assert out["chr1:10"].tolist() == [0.0, 1.0, 2.0]
    assert out["chr1:30"].tolist() == [0.25, 0.0, 1.0]
    assert out["chr2:15"].tolist() == [1.0, 0.0, 2.0]
    assert fake_adapter.last_export_allele_by_variant == {
        "chr1:10": "G",
        "chr1:30": "A",
        "chr2:15": "C",
    }
    assert fake_adapter.last_export_other_allele_by_variant == {
        "chr1:10": "A",
        "chr1:30": "G",
        "chr2:15": "T",
    }


def test_export_dosages_preserves_requested_sample_order(fake_adapter: HapStoreAdapter) -> None:
    stores = {
        ("EUR", 1): FakeStore(
            sample_ids=["id1", "id2", "id3"],
            positions=[10],
            refs=["A"],
            alts=["G"],
            genotypes=[[0], [1], [2]],
        ),
    }
    _wire_fake_stores(fake_adapter, stores)

    out = fake_adapter.export_dosages(
        sample_ids=["id3", "id1"],
        sample_populations=["EUR", "EUR"],
        variant_ids=["chr1:10"],
        out_dir=Path("/tmp"),
    )

    assert out["IID"].tolist() == ["id3", "id1"]
    assert out["chr1:10"].tolist() == [2.0, 0.0]


def test_export_haplotypes_reads_per_pop_per_chr(fake_adapter: HapStoreAdapter) -> None:
    stores = {
        ("EUR", 1): FakeStore(
            sample_ids=["id1", "id2"],
            positions=[10, 20],
            refs=["A", "C"],
            alts=["G", "T"],
            genotypes=[[1, 2], [0, 1]],
        ),
    }
    _wire_fake_stores(fake_adapter, stores)

    hap = fake_adapter.export_haplotypes(
        sample_ids=["id2", "id1"],
        sample_populations=["EUR", "EUR"],
        variant_ids=["chr1:20", "chr1:10"],
        out_dir=Path("/tmp"),
    )

    assert hap.shape == (2, 2, 2)
    # id2: chr1:20 genotype=1 -> hap0=1 hap1=0; chr1:10 genotype=0 -> 0/0
    assert hap[0, :, :].tolist() == [[1, 0], [0, 0]]
    # id1: chr1:20 genotype=2 -> 1/1; chr1:10 genotype=1 -> 1/0
    assert hap[1, :, :].tolist() == [[1, 1], [1, 0]]


def test_export_genotypes_u8_matches_export_dosages(fake_adapter: HapStoreAdapter) -> None:
    fake_adapter.missing_fill_value = 0.0
    stores = {
        ("EUR", 1): FakeStore(
            sample_ids=["id1", "id3"],
            positions=[10, 30],
            refs=["A", "G"],
            alts=["G", "A"],
            genotypes=[[0, 255], [2, 1]],
        ),
        ("AFR", 1): FakeStore(
            sample_ids=["id2"],
            positions=[10, 30],
            refs=["A", "G"],
            alts=["G", "A"],
            genotypes=[[1, 0]],
        ),
    }
    _wire_fake_stores(fake_adapter, stores)

    dosage = fake_adapter.export_dosages(
        sample_ids=["id1", "id2", "id3"],
        sample_populations=["EUR", "AFR", "EUR"],
        variant_ids=["chr1:10", "chr1:30"],
        out_dir=Path("/tmp"),
    )
    geno = fake_adapter.export_genotypes_u8(
        sample_ids=["id1", "id2", "id3"],
        sample_populations=["EUR", "AFR", "EUR"],
        variant_ids=["chr1:10", "chr1:30"],
        out_dir=Path("/tmp"),
    )

    expected = dosage[["chr1:10", "chr1:30"]].to_numpy(dtype=np.float32, copy=False)
    expected[expected == np.float32(fake_adapter.missing_fill_value)] = 0.0
    assert np.array_equal(geno, expected.astype(np.uint8))


def test_missing_genotypes_fail_without_explicit_fill_policy() -> None:
    adapter = HapStoreAdapter(
        HapStoreInputConfig(
            root_dir="/tmp/hapstores",
            file_pattern="{population}/chr{chrom}.hapstore.zarr",
        )
    )
    _wire_fake_stores(
        adapter,
        {
            ("EUR", 1): FakeStore(
                sample_ids=["id1"],
                positions=[10],
                refs=["A"],
                alts=["G"],
                genotypes=[[255]],
            )
        },
    )

    with pytest.raises(ValueError, match="missing genotype values"):
        adapter.export_dosages(
            sample_ids=["id1"],
            sample_populations=["EUR"],
            variant_ids=["chr1:10"],
            out_dir=Path("/tmp"),
        )


def test_adapter_rejects_store_with_noncanonical_missing_sentinel(
    fake_adapter: HapStoreAdapter,
) -> None:
    store = FakeStore(
        sample_ids=["id1"],
        positions=[10],
        refs=["A"],
        alts=["G"],
        genotypes=[[0]],
    )
    store.attrs = lambda chrom: {"missing_sentinel": 254}  # type: ignore[method-assign]
    _wire_fake_stores(fake_adapter, {("EUR", 1): store})

    with pytest.raises(ValueError, match="missing_sentinel=254, expected 255"):
        fake_adapter.export_genotypes_u8(
            sample_ids=["id1"],
            sample_populations=["EUR"],
            variant_ids=["chr1:10"],
            out_dir=Path("/tmp"),
        )


def test_high_fraction_genotype_read_uses_contiguous_store_read(fake_adapter: HapStoreAdapter) -> None:
    store = FakeStore(
        sample_ids=["id1", "id2", "id3", "id4"],
        positions=[10, 20],
        refs=["A", "C"],
        alts=["G", "T"],
        genotypes=[[0, 1], [1, 2], [2, 0], [0, 2]],
    )
    _wire_fake_stores(fake_adapter, {("EUR", 1): store})

    out = fake_adapter.export_genotypes_u8(
        sample_ids=["id4", "id1", "id2"],
        sample_populations=["EUR", "EUR", "EUR"],
        variant_ids=["chr1:20", "chr1:10"],
        out_dir=Path("/tmp"),
    )

    assert store.genotype_sample_masks == [None]
    assert out.tolist() == [[2, 0], [1, 0], [2, 1]]


def test_high_fraction_haplotype_read_uses_contiguous_store_read(fake_adapter: HapStoreAdapter) -> None:
    store = FakeStore(
        sample_ids=["id1", "id2", "id3", "id4"],
        positions=[10, 20],
        refs=["A", "C"],
        alts=["G", "T"],
        genotypes=[[0, 1], [1, 2], [2, 0], [0, 2]],
    )
    _wire_fake_stores(fake_adapter, {("EUR", 1): store})

    hap = fake_adapter.export_haplotypes(
        sample_ids=["id4", "id1", "id2"],
        sample_populations=["EUR", "EUR", "EUR"],
        variant_ids=["chr1:20", "chr1:10"],
        out_dir=Path("/tmp"),
    )

    assert store.haplotype_sample_masks == [None]
    assert hap.tolist() == [
        [[1, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[1, 1], [1, 0]],
    ]


def test_export_dosages_single_store_ignores_sample_populations(tmp_path: Path) -> None:
    cfg = HapStoreInputConfig(
        root_dir="/tmp/ignored",
        layout="single_store",
        store_path="/tmp/fake_store.hapstore.zarr",
        missing_sentinel=255,
        missing_fill_value=0.0,
    )
    adapter = HapStoreAdapter(cfg)
    store = FakeStore(
        sample_ids=["id1", "id2"],
        positions=[10],
        refs=["A"],
        alts=["G"],
        genotypes=[[0], [2]],
    )
    adapter._open_store = lambda population, chrom: store  # type: ignore[method-assign]

    out = adapter.export_dosages(
        sample_ids=["id2", "id1"],
        sample_populations=["EUR", "AFR"],
        variant_ids=["chr1:10"],
        out_dir=tmp_path,
    )
    assert out["IID"].tolist() == ["id2", "id1"]
    assert out["chr1:10"].tolist() == [2.0, 0.0]


def test_export_dosages_requires_chr_pos_variant_ids(fake_adapter: HapStoreAdapter) -> None:
    stores = {
        ("EUR", 1): FakeStore(
            sample_ids=["id1"],
            positions=[10],
            refs=["A"],
            alts=["G"],
            genotypes=[[0]],
        ),
    }
    _wire_fake_stores(fake_adapter, stores)

    with pytest.raises(ValueError, match="chrN:POS"):
        fake_adapter.export_dosages(
            sample_ids=["id1"],
            sample_populations=["EUR"],
            variant_ids=["rs1"],
            out_dir=Path("/tmp"),
        )


def test_export_dosages_rejects_alt_mismatch_between_populations(fake_adapter: HapStoreAdapter) -> None:
    stores = {
        ("EUR", 1): FakeStore(
            sample_ids=["id1"],
            positions=[10],
            refs=["A"],
            alts=["G"],
            genotypes=[[0]],
        ),
        ("AFR", 1): FakeStore(
            sample_ids=["id2"],
            positions=[10],
            refs=["A"],
            alts=["T"],
            genotypes=[[1]],
        ),
    }
    _wire_fake_stores(fake_adapter, stores)

    with pytest.raises(ValueError, match="ALT allele mismatch"):
        fake_adapter.export_dosages(
            sample_ids=["id1", "id2"],
            sample_populations=["EUR", "AFR"],
            variant_ids=["chr1:10"],
            out_dir=Path("/tmp"),
        )


def test_compute_global_pcs_is_deterministic(fake_adapter: HapStoreAdapter) -> None:
    stores = {
        ("EUR", 1): FakeStore(
            sample_ids=["id1", "id3"],
            positions=[10, 20, 30, 40],
            refs=["A", "A", "A", "A"],
            alts=["G", "G", "G", "G"],
            genotypes=[[0, 1, 2, 0], [2, 1, 0, 1]],
        ),
        ("AFR", 1): FakeStore(
            sample_ids=["id2"],
            positions=[10, 20, 30, 40],
            refs=["A", "A", "A", "A"],
            alts=["G", "G", "G", "G"],
            genotypes=[[1, 1, 1, 1]],
        ),
    }
    _wire_fake_stores(fake_adapter, stores)

    mode_cfg = SimpleNamespace(svd_max_variants=4, svd_min_maf=0.0, svd_random_seed=123)
    pcs1 = fake_adapter.compute_global_pcs(
        sample_ids=["id1", "id2", "id3"],
        sample_populations=["EUR", "AFR", "EUR"],
        n_components=2,
        out_dir=Path("/tmp"),
        mode_cfg=mode_cfg,
        chromosomes=[1],
    )
    pcs2 = fake_adapter.compute_global_pcs(
        sample_ids=["id1", "id2", "id3"],
        sample_populations=["EUR", "AFR", "EUR"],
        n_components=2,
        out_dir=Path("/tmp"),
        mode_cfg=mode_cfg,
        chromosomes=[1],
    )

    assert list(pcs1.columns) == ["FID", "IID", "PC1", "PC2"]
    assert pcs1["IID"].tolist() == ["id1", "id2", "id3"]
    assert np.allclose(pcs1[["PC1", "PC2"]].to_numpy(), pcs2[["PC1", "PC2"]].to_numpy())
