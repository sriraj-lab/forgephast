"""Truth local ancestry tract parsing and rasterization."""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
try:  # pragma: no cover - optional dependency
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    njit = None
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tract:
    left_bp: int
    right_bp: int
    ancestry: str


def _canonicalize_label(label: str, label_map: Dict[str, str]) -> str:
    raw = str(label).strip()
    if raw in label_map:
        return str(label_map[raw])
    upper = raw.upper()
    if upper in label_map:
        return str(label_map[upper])
    return raw


def _parse_header(line: str) -> Dict[str, int]:
    cols = [c.strip() for c in line.strip().split("\t") if c.strip()]
    index = {c: i for i, c in enumerate(cols)}
    return index


def _detect_schema(index: Dict[str, int]) -> str:
    explicit_required = {"sample_id", "hap_idx", "left_bp", "right_bp", "ancestry"}
    if explicit_required.issubset(index):
        return "explicit_sample_hap"

    haplotype_required = {"haplotype_index", "start_bp", "end_bp", "source_population"}
    if haplotype_required.issubset(index):
        return "haplotype_index_combined"

    raise ValueError(
        "Tracts header missing required columns. "
        f"Need either {sorted(explicit_required)} or {sorted(haplotype_required)}; "
        f"got={sorted(index)}"
    )


def load_tracts_tsv_gz(
    tracts_path: str | Path,
    *,
    sample_ids: Iterable[str],
    label_map: Dict[str, str],
) -> Dict[Tuple[str, int], List[Tract]]:
    """Load tracts for the given sample IDs.

    Supports either of these gzip-compressed TSV schemas:

    1) Explicit sample/haplotype (preferred):
       sample_id, hap_idx, left_bp, right_bp, ancestry

    2) Combined haplotype-indexed tracts (as produced by some PopAmp exports):
       haplotype_index, start_bp, end_bp, source_population

       In this schema, haplotype_index is interpreted as:
         sample_idx = haplotype_index // 2
         hap_idx    = haplotype_index % 2

       This assumes the provided sample_ids are ordered consistently with the
       haplotype_index ordering in the tracts file.
    """
    path = Path(tracts_path)
    ordered_sample_ids = [str(s) for s in sample_ids]
    need = set(ordered_sample_ids)
    out: Dict[Tuple[str, int], List[Tract]] = {}

    header_index: Dict[str, int] | None = None
    schema: str | None = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if header_index is None:
                header_index = _parse_header(line)
                schema = _detect_schema(header_index)
                if schema == "haplotype_index_combined":
                    logger.info(
                        "Detected haplotype_index tracts schema for %s; mapping haplotype_index to sample_ids order",
                        path,
                    )
                continue
            if header_index is None:
                raise RuntimeError("header_index not initialized")
            if schema is None:
                raise RuntimeError("schema not initialized")
            fields = line.split("\t")

            if schema == "explicit_sample_hap":
                sid = str(fields[header_index["sample_id"]])
                if sid not in need:
                    continue
                hap_idx = int(fields[header_index["hap_idx"]])
                if hap_idx not in (0, 1):
                    raise ValueError(f"Invalid hap_idx={hap_idx} at {path}:{line_no}")
                left_bp = int(fields[header_index["left_bp"]])
                right_bp = int(fields[header_index["right_bp"]])
                anc_raw = fields[header_index["ancestry"]]
            elif schema == "haplotype_index_combined":
                haplotype_index = int(fields[header_index["haplotype_index"]])
                pop_name = fields[header_index["population"]] if "population" in header_index else None
                if pop_name:
                    sample_number = int(haplotype_index) // 2 + 1
                    sid = f"{pop_name}_syn{sample_number}"
                else:
                    sample_idx = int(haplotype_index) // 2
                    if sample_idx < 0 or sample_idx >= len(ordered_sample_ids):
                        continue
                    sid = ordered_sample_ids[sample_idx]
                if sid not in need:
                    continue
                hap_idx = int(haplotype_index) % 2
                left_bp = int(float(fields[header_index["start_bp"]]))
                right_bp = int(float(fields[header_index["end_bp"]]))
                anc_raw = fields[header_index["source_population"]]
            else:
                raise RuntimeError(f"Unsupported schema={schema}")

            if right_bp <= left_bp:
                continue
            anc = _canonicalize_label(anc_raw, label_map=label_map)
            out.setdefault((sid, hap_idx), []).append(
                Tract(left_bp=left_bp, right_bp=right_bp, ancestry=anc)
            )

    missing = [(sid, h) for sid in need for h in (0, 1) if (sid, h) not in out]
    if missing:
        first = missing[0]
        raise ValueError(
            f"Tracts file missing sample/haplotype rows; first missing={first} (n={len(missing)})"
        )

    for key, tracts in out.items():
        last_left = None
        last_right = None
        for tract in tracts:
            if last_left is not None and tract.left_bp < int(last_left):
                raise ValueError(f"Tracts not sorted by left_bp for {key} in {path}")
            if last_right is not None and tract.left_bp < int(last_right):
                raise ValueError(f"Overlapping tracts detected for {key} in {path}")
            last_left = tract.left_bp
            last_right = tract.right_bp

    return out


def _fill_codes_for_one_haplotype(
    positions_bp_sorted: np.ndarray,
    tracts: List[Tract],
    ancestry_to_index: Dict[str, int],
) -> np.ndarray:
    codes = np.full(int(positions_bp_sorted.shape[0]), -1, dtype=np.int16)
    if positions_bp_sorted.size == 0:
        return codes
    if not tracts:
        return codes

    j = 0
    for k, pos in enumerate(positions_bp_sorted.astype(np.int64)):
        while j < len(tracts) and int(pos) >= int(tracts[j].right_bp):
            j += 1
        if j >= len(tracts):
            break
        tract = tracts[j]
        if int(pos) < int(tract.left_bp) or int(pos) >= int(tract.right_bp):
            continue
        anc = str(tract.ancestry)
        if anc not in ancestry_to_index:
            raise ValueError(f"Unknown ancestry label {anc!r} (not in ancestry_order)")
        codes[k] = int(ancestry_to_index[anc])
    return codes


if _HAS_NUMBA:

    @njit(cache=True)
    def _fill_codes_numba(
        positions_bp_sorted: np.ndarray,
        left_bp: np.ndarray,
        right_bp: np.ndarray,
        ancestry_codes: np.ndarray,
    ) -> np.ndarray:
        codes = np.full(int(positions_bp_sorted.shape[0]), -1, dtype=np.int16)
        if positions_bp_sorted.size == 0:
            return codes
        if left_bp.size == 0:
            return codes
        j = 0
        for k in range(positions_bp_sorted.shape[0]):
            pos = int(positions_bp_sorted[k])
            while j < left_bp.shape[0] and pos >= int(right_bp[j]):
                j += 1
            if j >= left_bp.shape[0]:
                break
            if pos < int(left_bp[j]) or pos >= int(right_bp[j]):
                continue
            codes[k] = int(ancestry_codes[j])
        return codes


def rasterize_tracts_to_variant_ancestry(
    *,
    sample_ids: List[str],
    tracts_by_sample_hap: Dict[Tuple[str, int], List[Tract]],
    variant_positions_bp: np.ndarray,
    ancestry_to_index: Dict[str, int],
    chunk_n_samples: int = 5000,
    use_numba: bool = False,
) -> np.ndarray:
    """Rasterize per-haplotype ancestry codes for each sample at each variant.

    Returns int16 codes of shape (n_samples, n_variants, 2), aligned to the input
    sample_ids and variant_positions_bp order.
    """
    positions = np.asarray(variant_positions_bp, dtype=np.int64)
    order = np.argsort(positions, kind="mergesort")
    inv = np.empty_like(order)
    inv[order] = np.arange(order.shape[0])
    pos_sorted = positions[order]

    n_samples = len(sample_ids)
    n_variants = int(positions.shape[0])
    out_sorted = np.full((n_samples, n_variants, 2), -1, dtype=np.int16)
    use_numba_kernel = bool(use_numba and _HAS_NUMBA)
    encoded_tracts: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if use_numba_kernel:
        for key, tracts in tracts_by_sample_hap.items():
            left = np.empty(len(tracts), dtype=np.int64)
            right = np.empty(len(tracts), dtype=np.int64)
            anc_codes = np.empty(len(tracts), dtype=np.int16)
            for idx, tract in enumerate(tracts):
                anc = str(tract.ancestry)
                if anc not in ancestry_to_index:
                    raise ValueError(f"Unknown ancestry label {anc!r} (not in ancestry_order)")
                left[idx] = int(tract.left_bp)
                right[idx] = int(tract.right_bp)
                anc_codes[idx] = int(ancestry_to_index[anc])
            encoded_tracts[key] = (left, right, anc_codes)

    for start in range(0, n_samples, int(chunk_n_samples)):
        end = min(n_samples, start + int(chunk_n_samples))
        for i in range(start, end):
            sid = str(sample_ids[i])
            for hap_idx in (0, 1):
                if use_numba_kernel:
                    left, right, anc_codes = encoded_tracts[(sid, hap_idx)]
                    out_sorted[i, :, hap_idx] = _fill_codes_numba(
                        pos_sorted, left, right, anc_codes
                    )
                else:
                    tracts = tracts_by_sample_hap[(sid, hap_idx)]
                    out_sorted[i, :, hap_idx] = _fill_codes_for_one_haplotype(
                        pos_sorted, tracts=tracts, ancestry_to_index=ancestry_to_index
                    )

    out = out_sorted[:, inv, :]
    if np.any(out < 0):
        bad = np.where(out < 0)
        first = (int(bad[0][0]), int(bad[1][0]), int(bad[2][0]))
        raise ValueError(
            "Local ancestry rasterization produced missing codes. "
            f"First missing sample_idx={first[0]} variant_idx={first[1]} hap_idx={first[2]}"
        )
    return out
