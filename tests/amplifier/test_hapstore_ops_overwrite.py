from __future__ import annotations

import numpy as np
import pytest

import forgephast.amplifier.hapstore_ops as hop


def _dummy_ingest_kwargs(output_zarr):
    return {
        "output_zarr": output_zarr,
        "chrom": "chr21",
        "haplotypes": np.zeros((2, 2), dtype=np.uint8),
        "variant_pos": np.asarray([100, 200], dtype=np.int64),
        "sample_ids": ["s1"],
        "ref": ["A", "A"],
        "alt": ["G", "G"],
        "vid": ["v1", "v2"],
    }


def test_ingest_arrays_retries_on_stale_store_error(tmp_path, monkeypatch):
    out = tmp_path / "stale.hapstore.zarr"
    out.mkdir(parents=True, exist_ok=True)

    calls = {"n": 0}

    def _fake_ingest(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("chromosome 'chr21' already exists")
        return "ok"

    monkeypatch.setattr(hop, "_pkg_ingest_arrays", _fake_ingest)
    result = hop.ingest_arrays(**_dummy_ingest_kwargs(out))

    assert result == "ok"
    assert calls["n"] == 2
    assert not out.exists()


def test_ingest_arrays_does_not_retry_on_non_stale_error(tmp_path, monkeypatch):
    out = tmp_path / "not_stale.hapstore.zarr"
    out.mkdir(parents=True, exist_ok=True)

    calls = {"n": 0}

    def _fake_ingest(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("permission denied")

    monkeypatch.setattr(hop, "_pkg_ingest_arrays", _fake_ingest)
    with pytest.raises(RuntimeError, match="permission denied"):
        hop.ingest_arrays(**_dummy_ingest_kwargs(out))

    assert calls["n"] == 1
    assert out.exists()


def test_merge_legacy_materialize_strategy_uses_fallback(monkeypatch):
    calls = {"concat": 0, "pkg": 0, "fallback": 0}

    def _pkg_concat(**kwargs):
        calls["concat"] += 1
        return "concat"

    def _pkg_merge(**kwargs):
        calls["pkg"] += 1
        return "pkg"

    def _fallback_merge(**kwargs):
        calls["fallback"] += 1
        assert kwargs["chunk_haplotype_size"] == 15000
        assert kwargs["chunk_sample_size"] is None
        return "fallback"

    monkeypatch.setattr(hop, "_pkg_concat_hapstores", _pkg_concat)
    monkeypatch.setattr(hop, "_pkg_merge_chromosomes_sample_axis", _pkg_merge)
    monkeypatch.setattr(hop, "_fallback_merge_chromosomes_sample_axis", _fallback_merge)

    out = hop.merge_chromosomes_sample_axis(
        input_stores=["a.zarr", "b.zarr"],
        output_store="out.zarr",
        chrom="chr21",
        chunk_variant_size=500,
        chunk_haplotype_size=15000,
        merge_strategy="legacy_materialize",
    )

    assert out == "fallback"
    assert calls["concat"] == 0
    assert calls["pkg"] == 0
    assert calls["fallback"] == 1


def test_merge_streaming_uses_concat_with_n_workers(monkeypatch):
    calls: dict[str, object] = {}

    def _pkg_concat(**kwargs):
        calls.update(kwargs)
        return "concat"

    monkeypatch.setattr(hop, "_pkg_concat_hapstores", _pkg_concat)
    monkeypatch.setattr(hop, "_pkg_merge_chromosomes_sample_axis", None)

    out = hop.merge_chromosomes_sample_axis(
        input_stores=["a.zarr", "b.zarr"],
        output_store="out.zarr",
        chrom="chr21",
        chunk_variant_size=500,
        n_workers=8,
    )

    assert out == "concat"
    assert calls["axis"] == "samples"
    assert calls["n_workers"] == 8
    assert calls["overwrite"] is True


def test_merge_invalid_strategy_raises():
    with pytest.raises(ValueError, match="merge_strategy"):
        hop.merge_chromosomes_sample_axis(
            input_stores=["a.zarr", "b.zarr"],
            output_store="out.zarr",
            chrom="chr21",
            merge_strategy="not_a_strategy",
        )
