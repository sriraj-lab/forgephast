"""Stable local ancestry source object storage.

The source object is a Zarr-backed run-length encoding of per-haplotype
ancestry tracts. Python classes here are readers/writers around the on-disk
schema; the object itself is not a pickled Python object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from numcodecs import Blosc

LOCAL_ANCESTRY_SOURCE_SCHEMA_VERSION = "forgephast.local_ancestry_source.v1"


@dataclass(frozen=True)
class LocalAncestryTract:
    left_bp: int
    right_bp: int
    ancestry: str


def _string_array(values: Sequence[str]) -> np.ndarray:
    encoded = [str(value).encode("utf-8") for value in values]
    width = max(1, *(len(value) for value in encoded))
    return np.asarray(encoded, dtype=f"S{width}")


def _decode_string_array(values: np.ndarray) -> list[str]:
    arr = np.asarray(values)
    out: list[str] = []
    for value in arr:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        elif hasattr(value, "decode"):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _numeric_compressor(compression_level: int) -> Blosc:
    return Blosc(
        cname="zstd",
        clevel=int(compression_level),
        shuffle=Blosc.BITSHUFFLE,
    )


def _create_dataset(group: zarr.hierarchy.Group, name: str, data: np.ndarray, compressor: Blosc) -> None:
    group.create_dataset(
        name,
        data=np.asarray(data),
        shape=np.asarray(data).shape,
        dtype=np.asarray(data).dtype,
        chunks=True,
        compressor=compressor,
        overwrite=True,
    )


def _extract_tract_value(tract: Any, key: str) -> Any:
    if isinstance(tract, Mapping):
        return tract[key]
    return getattr(tract, key)


def _coerce_tract(tract: Any) -> LocalAncestryTract:
    return LocalAncestryTract(
        left_bp=int(float(_extract_tract_value(tract, "left_bp"))),
        right_bp=int(float(_extract_tract_value(tract, "right_bp"))),
        ancestry=str(_extract_tract_value(tract, "ancestry")),
    )


def _validate_sample_ids(sample_ids: Sequence[str]) -> list[str]:
    out = [str(sample_id) for sample_id in sample_ids]
    if not out:
        raise ValueError("sample_ids cannot be empty")
    if len(set(out)) != len(out):
        raise ValueError("sample_ids must be unique")
    if any(not sample_id for sample_id in out):
        raise ValueError("sample_ids cannot contain empty IDs")
    return out


def _validate_ancestry_order(ancestry_order: Sequence[str]) -> list[str]:
    out = [str(label) for label in ancestry_order]
    if not out:
        raise ValueError("ancestry_order cannot be empty")
    if len(set(out)) != len(out):
        raise ValueError("ancestry_order must contain unique labels")
    if any(not label for label in out):
        raise ValueError("ancestry_order cannot contain empty labels")
    return out


def _build_rle_arrays(
    *,
    sample_ids: list[str],
    tracts_by_sample_hap: Mapping[tuple[str, int], Sequence[Any]],
    ancestry_order: list[str],
) -> dict[str, np.ndarray]:
    ancestry_to_code = {label: idx for idx, label in enumerate(ancestry_order)}
    hap_sample_index: list[int] = []
    hap_phase: list[int] = []
    hap_offsets: list[int] = [0]
    tract_start_bp: list[int] = []
    tract_end_bp: list[int] = []
    source_code: list[int] = []

    for sample_idx, sample_id in enumerate(sample_ids):
        for phase in (0, 1):
            key = (sample_id, phase)
            if key not in tracts_by_sample_hap:
                raise ValueError(f"Missing local ancestry tracts for sample={sample_id!r} hap={phase}")
            tracts = sorted(
                (_coerce_tract(tract) for tract in tracts_by_sample_hap[key]),
                key=lambda tract: (int(tract.left_bp), int(tract.right_bp)),
            )
            last_right: int | None = None
            for tract in tracts:
                if int(tract.right_bp) <= int(tract.left_bp):
                    raise ValueError(
                        f"Invalid local ancestry tract for sample={sample_id!r} hap={phase}: "
                        f"right_bp must be > left_bp"
                    )
                if last_right is not None and int(tract.left_bp) < int(last_right):
                    raise ValueError(
                        f"Overlapping local ancestry tracts for sample={sample_id!r} hap={phase}"
                    )
                if str(tract.ancestry) not in ancestry_to_code:
                    raise ValueError(
                        f"Unknown ancestry label {tract.ancestry!r}; expected one of {ancestry_order}"
                    )
                tract_start_bp.append(int(tract.left_bp))
                tract_end_bp.append(int(tract.right_bp))
                source_code.append(int(ancestry_to_code[str(tract.ancestry)]))
                last_right = int(tract.right_bp)
            hap_sample_index.append(int(sample_idx))
            hap_phase.append(int(phase))
            hap_offsets.append(len(tract_start_bp))

    return {
        "hap_sample_index": np.asarray(hap_sample_index, dtype=np.int32),
        "hap_phase": np.asarray(hap_phase, dtype=np.int8),
        "hap_offsets": np.asarray(hap_offsets, dtype=np.int64),
        "tract_start_bp": np.asarray(tract_start_bp, dtype=np.int64),
        "tract_end_bp": np.asarray(tract_end_bp, dtype=np.int64),
        "source_code": np.asarray(source_code, dtype=np.int16),
    }


def write_local_ancestry_source(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    tracts_by_sample_hap: Mapping[tuple[str, int], Sequence[Any]],
    ancestry_order: Sequence[str],
    chrom: int | str,
    metadata: Mapping[str, Any] | None = None,
    compression_level: int = 5,
) -> Path:
    """Write a Zarr local ancestry source object.

    The written object stores one RLE stream per haplotype. Intervals use
    left-closed, right-open base-pair coordinates, matching the phenotype
    rasterization contract.
    """
    out_path = Path(path)
    ordered_sample_ids = _validate_sample_ids(sample_ids)
    ordered_ancestries = _validate_ancestry_order(ancestry_order)
    arrays = _build_rle_arrays(
        sample_ids=ordered_sample_ids,
        tracts_by_sample_hap=tracts_by_sample_hap,
        ancestry_order=ordered_ancestries,
    )

    compressor = _numeric_compressor(compression_level)
    group = zarr.open_group(str(out_path), mode="w")
    group.attrs.update(
        {
            "schema_version": LOCAL_ANCESTRY_SOURCE_SCHEMA_VERSION,
            "chrom": str(chrom),
            "coordinate_system": "bp_left_closed_right_open",
            "storage": "rle_per_haplotype",
            "n_samples": int(len(ordered_sample_ids)),
            "n_haplotypes": int(2 * len(ordered_sample_ids)),
            "n_tracts": int(arrays["tract_start_bp"].shape[0]),
            "ancestry_order": list(ordered_ancestries),
        }
    )
    if metadata:
        for key, value in metadata.items():
            group.attrs[str(key)] = value

    _create_dataset(group, "sample_id", _string_array(ordered_sample_ids), compressor)
    _create_dataset(group, "ancestry_label", _string_array(ordered_ancestries), compressor)
    for name, array in arrays.items():
        _create_dataset(group, name, array, compressor)
    return out_path


def write_local_ancestry_source_from_tract_rows(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    tract_rows: Sequence[Mapping[str, Any]],
    ancestry_order: Sequence[str],
    chrom: int | str,
    metadata: Mapping[str, Any] | None = None,
    compression_level: int = 5,
) -> Path:
    """Write a source object from msprime-style tract row records."""
    ordered_sample_ids = _validate_sample_ids(sample_ids)
    tracts_by_sample_hap: dict[tuple[str, int], list[LocalAncestryTract]] = {
        (sample_id, phase): []
        for sample_id in ordered_sample_ids
        for phase in (0, 1)
    }
    for row in tract_rows:
        haplotype_index = int(row["haplotype_index"])
        sample_idx = haplotype_index // 2
        phase = haplotype_index % 2
        if sample_idx < 0 or sample_idx >= len(ordered_sample_ids):
            continue
        sample_id = ordered_sample_ids[sample_idx]
        tracts_by_sample_hap[(sample_id, phase)].append(
            LocalAncestryTract(
                left_bp=int(float(row["start_bp"])),
                right_bp=int(float(row["end_bp"])),
                ancestry=str(row["source_population"]),
            )
        )
    return write_local_ancestry_source(
        path,
        sample_ids=ordered_sample_ids,
        tracts_by_sample_hap=tracts_by_sample_hap,
        ancestry_order=ancestry_order,
        chrom=chrom,
        metadata=metadata,
        compression_level=compression_level,
    )


def _rasterize_one_haplotype(
    *,
    positions_bp_sorted: np.ndarray,
    tract_start_bp: np.ndarray,
    tract_end_bp: np.ndarray,
    source_code: np.ndarray,
    code_map: np.ndarray,
) -> np.ndarray:
    codes = np.full(int(positions_bp_sorted.shape[0]), -1, dtype=np.int16)
    if positions_bp_sorted.size == 0 or tract_start_bp.size == 0:
        return codes
    idx = np.searchsorted(tract_end_bp, positions_bp_sorted, side="right")
    valid = idx < tract_end_bp.shape[0]
    if not np.any(valid):
        return codes
    valid_idx = idx[valid]
    valid_positions = positions_bp_sorted[valid]
    covered = (
        (valid_positions >= tract_start_bp[valid_idx])
        & (valid_positions < tract_end_bp[valid_idx])
    )
    if not np.any(covered):
        return codes
    out_positions = np.where(valid)[0][covered]
    raw_codes = source_code[valid_idx[covered]]
    codes[out_positions] = code_map[raw_codes]
    return codes


@dataclass(frozen=True)
class LocalAncestrySource:
    """In-memory reader facade for a Zarr local ancestry source object."""

    path: Path
    chrom: str
    sample_ids: list[str]
    ancestry_order: list[str]
    hap_sample_index: np.ndarray
    hap_phase: np.ndarray
    hap_offsets: np.ndarray
    tract_start_bp: np.ndarray
    tract_end_bp: np.ndarray
    source_code: np.ndarray

    @classmethod
    def open(cls, path: str | Path) -> "LocalAncestrySource":
        source_path = Path(path)
        group = zarr.open_group(str(source_path), mode="r")
        schema_version = group.attrs.get("schema_version")
        if schema_version != LOCAL_ANCESTRY_SOURCE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported local ancestry source schema {schema_version!r}; "
                f"expected {LOCAL_ANCESTRY_SOURCE_SCHEMA_VERSION!r}"
            )
        required = {
            "sample_id",
            "ancestry_label",
            "hap_sample_index",
            "hap_phase",
            "hap_offsets",
            "tract_start_bp",
            "tract_end_bp",
            "source_code",
        }
        missing = sorted(name for name in required if name not in group)
        if missing:
            raise ValueError(f"Local ancestry source missing required arrays: {missing}")

        sample_ids = _decode_string_array(group["sample_id"][:])
        ancestry_order = _decode_string_array(group["ancestry_label"][:])
        hap_sample_index = np.asarray(group["hap_sample_index"][:], dtype=np.int32)
        hap_phase = np.asarray(group["hap_phase"][:], dtype=np.int8)
        hap_offsets = np.asarray(group["hap_offsets"][:], dtype=np.int64)
        tract_start_bp = np.asarray(group["tract_start_bp"][:], dtype=np.int64)
        tract_end_bp = np.asarray(group["tract_end_bp"][:], dtype=np.int64)
        source_code = np.asarray(group["source_code"][:], dtype=np.int16)

        if hap_offsets.shape[0] != hap_sample_index.shape[0] + 1:
            raise ValueError("Invalid local ancestry source: hap_offsets length mismatch")
        if hap_phase.shape[0] != hap_sample_index.shape[0]:
            raise ValueError("Invalid local ancestry source: hap_phase length mismatch")
        if tract_start_bp.shape != tract_end_bp.shape or tract_start_bp.shape != source_code.shape:
            raise ValueError("Invalid local ancestry source: tract arrays must have matching shapes")
        return cls(
            path=source_path,
            chrom=str(group.attrs.get("chrom", "")),
            sample_ids=sample_ids,
            ancestry_order=ancestry_order,
            hap_sample_index=hap_sample_index,
            hap_phase=hap_phase,
            hap_offsets=hap_offsets,
            tract_start_bp=tract_start_bp,
            tract_end_bp=tract_end_bp,
            source_code=source_code,
        )

    def rasterize(
        self,
        *,
        sample_ids: Sequence[str],
        positions_bp: np.ndarray,
        ancestry_to_index: Mapping[str, int] | None = None,
        chunk_n_samples: int = 5000,
    ) -> np.ndarray:
        """Rasterize ancestry codes to ``samples x variants x haplotypes``."""
        positions = np.asarray(positions_bp, dtype=np.int64)
        order = np.argsort(positions, kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(order.shape[0])
        positions_sorted = positions[order]

        requested_sample_ids = [str(sample_id) for sample_id in sample_ids]
        sample_index_by_id = {sample_id: idx for idx, sample_id in enumerate(self.sample_ids)}
        hap_index_by_sample_phase = {
            (int(sample_idx), int(phase)): hap_idx
            for hap_idx, (sample_idx, phase) in enumerate(
                zip(self.hap_sample_index, self.hap_phase)
            )
        }

        if ancestry_to_index is None:
            code_map = np.arange(len(self.ancestry_order), dtype=np.int16)
        else:
            code_map = np.empty(len(self.ancestry_order), dtype=np.int16)
            for source_idx, label in enumerate(self.ancestry_order):
                if label not in ancestry_to_index:
                    raise ValueError(
                        f"Source ancestry label {label!r} is not present in requested ancestry order"
                    )
                code_map[source_idx] = int(ancestry_to_index[label])

        out_sorted = np.full(
            (len(requested_sample_ids), int(positions.shape[0]), 2),
            -1,
            dtype=np.int16,
        )
        chunk = max(1, int(chunk_n_samples))
        for start in range(0, len(requested_sample_ids), chunk):
            end = min(len(requested_sample_ids), start + chunk)
            for out_sample_idx in range(start, end):
                sample_id = requested_sample_ids[out_sample_idx]
                if sample_id not in sample_index_by_id:
                    raise ValueError(
                        f"Sample {sample_id!r} is not present in local ancestry source {self.path}"
                    )
                source_sample_idx = int(sample_index_by_id[sample_id])
                for phase in (0, 1):
                    hap_idx = hap_index_by_sample_phase.get((source_sample_idx, phase))
                    if hap_idx is None:
                        raise ValueError(
                            f"Local ancestry source missing haplotype for sample={sample_id!r} phase={phase}"
                        )
                    tract_start = int(self.hap_offsets[hap_idx])
                    tract_end = int(self.hap_offsets[hap_idx + 1])
                    out_sorted[out_sample_idx, :, phase] = _rasterize_one_haplotype(
                        positions_bp_sorted=positions_sorted,
                        tract_start_bp=self.tract_start_bp[tract_start:tract_end],
                        tract_end_bp=self.tract_end_bp[tract_start:tract_end],
                        source_code=self.source_code[tract_start:tract_end],
                        code_map=code_map,
                    )

        out = out_sorted[:, inv, :]
        if np.any(out < 0):
            bad = np.where(out < 0)
            first = (int(bad[0][0]), int(bad[1][0]), int(bad[2][0]))
            raise ValueError(
                "Local ancestry source rasterization produced missing codes. "
                f"First missing sample_idx={first[0]} variant_idx={first[1]} hap_idx={first[2]}"
            )
        return out
