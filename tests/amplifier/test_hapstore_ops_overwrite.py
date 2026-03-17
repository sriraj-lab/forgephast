from __future__ import annotations

import numpy as np
import pytest

import forgephast.amplifier.simulation as sim
import importlib
merge_mod = importlib.import_module("forgephast.amplifier.merge")


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

    monkeypatch.setattr(sim, "_hapstore_ingest_arrays", _fake_ingest)
    result = sim._ingest_arrays_with_retry(**_dummy_ingest_kwargs(out))

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

    monkeypatch.setattr(sim, "_hapstore_ingest_arrays", _fake_ingest)
    with pytest.raises(RuntimeError, match="permission denied"):
        sim._ingest_arrays_with_retry(**_dummy_ingest_kwargs(out))

    assert calls["n"] == 1
    assert out.exists()


def test_merge_streaming_uses_concat_with_n_workers(monkeypatch):
    calls: dict[str, object] = {}

    def _pkg_concat(**kwargs):
        calls.update(kwargs)
        return "concat"

    monkeypatch.setattr(merge_mod, "_pkg_concat_hapstores", _pkg_concat)

    out = merge_mod.merge_chromosomes_sample_axis(
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


def test_merge_rejects_non_overwrite_mode():
    with pytest.raises(ValueError, match="output_mode='w'"):
        merge_mod.merge_chromosomes_sample_axis(
            input_stores=["a.zarr", "b.zarr"],
            output_store="out.zarr",
            chrom="chr21",
            output_mode="a",
        )
