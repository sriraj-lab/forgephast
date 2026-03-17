"""Donor panel loading and donor HapStore cache materialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from hapstore import ingest_vcf

from .._internal.cache import canonical_hash
from .vcfio import VariantTable


@dataclass
class PanelChromData:
    variants: VariantTable
    donor_haps: np.ndarray  # shape: (2 * n_donors, n_variants)
    donor_ids: list[str]


def load_donor_haplotypes_hapstore(
    hapstore_path: str | Path,
    chrom: int | str,
    donor_ids: list[str],
) -> PanelChromData:
    donor_ids = list(donor_ids)
    from hapstore import HapStore

    hs = HapStore(hapstore_path, mode="r")
    all_ids = [str(x) for x in hs.sample_ids(chrom)]
    index_by_id = {sid: idx for idx, sid in enumerate(all_ids)}
    missing = [sid for sid in donor_ids if sid not in index_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} donor IDs not present in panel HapStore {hapstore_path}; first={missing[:5]}"
        )

    requested_idx = np.asarray([index_by_id[sid] for sid in donor_ids], dtype=np.int64)
    sample_mask = np.zeros(len(all_ids), dtype=bool)
    sample_mask[requested_idx] = True
    selected_haps = hs.haplotypes(chrom, sample_mask=sample_mask)

    selected_idx = np.flatnonzero(sample_mask).astype(np.int64, copy=False)
    reorder = np.searchsorted(selected_idx, requested_idx)
    if reorder.size != requested_idx.size or not np.array_equal(selected_idx[reorder], requested_idx):
        raise RuntimeError("Unable to reorder HapStore donor samples to requested order")

    hap_rows = np.empty(reorder.size * 2, dtype=np.int64)
    hap_rows[0::2] = 2 * reorder
    hap_rows[1::2] = (2 * reorder) + 1
    donor_haps = np.asarray(selected_haps[hap_rows, :], dtype=np.uint8)

    variant_info = hs.variant_info(chrom)
    pos = np.asarray(hs.variant_positions(chrom), dtype=np.int64)
    ref = [str(x) for x in variant_info["ref"]]
    alt = [str(x) for x in variant_info["alt"]]
    vid = [str(x) for x in variant_info["vid"]]

    keep = np.asarray(
        [
            (len(ref_i) == 1) and (len(alt_i) == 1) and ("," not in alt_i)
            for ref_i, alt_i in zip(ref, alt)
        ],
        dtype=bool,
    )
    if keep.shape[0] != pos.shape[0] or keep.shape[0] != donor_haps.shape[1]:
        raise ValueError(
            "HapStore donor panel variant axis mismatch while filtering biallelic SNPs: "
            f"pos={pos.shape[0]} keep={keep.shape[0]} haps={donor_haps.shape[1]}"
        )
    if not np.any(keep):
        raise ValueError(f"No eligible biallelic SNPs found in {hapstore_path} chrom={chrom}")

    pos_kept = pos[keep]
    donor_haps_kept = donor_haps[:, keep]
    ref_kept = [ref[i] for i, flag in enumerate(keep.tolist()) if flag]
    alt_kept = [alt[i] for i, flag in enumerate(keep.tolist()) if flag]
    vid_kept = [vid[i] for i, flag in enumerate(keep.tolist()) if flag]
    chrom_s = str(chrom)
    if chrom_s.isdigit():
        chrom_s = f"chr{chrom_s}"
    variants = VariantTable(
        chrom=chrom_s,
        pos=pos_kept,
        vid=vid_kept,
        ref=ref_kept,
        alt=alt_kept,
    )
    return PanelChromData(variants=variants, donor_haps=donor_haps_kept, donor_ids=donor_ids)


def donor_panel_cache_path(
    cache_dir: str | Path,
    source_path: str | Path,
    source_kind: str,
    chrom: int | str,
    donor_ids: list[str],
) -> Path:
    chrom_s = str(chrom)
    key = canonical_hash(
        {
            "source_kind": str(source_kind),
            "source_path": str(Path(source_path).resolve()),
            "chrom": chrom_s,
            "donor_ids": [str(sid) for sid in donor_ids],
        }
    )[:16]
    stem = Path(source_path).name
    return Path(cache_dir) / f"chr{chrom_s}" / f"{source_kind}.{stem}.{key}.hapstore.zarr"


def donor_panel_cache_metadata_path(cache_path: str | Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.parent / f"{cache_path.name}.source.json"


def _source_fingerprint(source_path: str | Path) -> dict[str, int | str]:
    source = Path(source_path)
    payload: dict[str, int | str] = {"source_path": str(source.resolve())}
    if source.exists():
        stat = source.stat()
        payload["source_size"] = int(stat.st_size)
        payload["source_mtime_ns"] = int(stat.st_mtime_ns)
    else:
        payload["source_size"] = -1
        payload["source_mtime_ns"] = -1
    return payload


def _write_cache_metadata(cache_path: Path, *, source_path: str | Path, source_kind: str) -> None:
    metadata_path = donor_panel_cache_metadata_path(cache_path)
    metadata = {"source_kind": str(source_kind), **_source_fingerprint(source_path)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        dir=str(cache_path.parent),
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(metadata_path)


def _cache_matches_source(
    cache_path: Path,
    source_path: Path,
    source_kind: str,
) -> bool:
    metadata_path = donor_panel_cache_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return False
    if str(metadata.get("source_kind")) != str(source_kind):
        return False
    if not source_path.exists():
        return True
    expected = _source_fingerprint(source_path)
    return (
        str(metadata.get("source_path")) == str(expected["source_path"])
        and int(metadata.get("source_size", -1)) == int(expected["source_size"])
        and int(metadata.get("source_mtime_ns", -1)) == int(expected["source_mtime_ns"])
    )


def ensure_donor_panel_hapstore(
    source_path: str | Path,
    source_kind: str,
    chrom: int | str,
    donor_ids: list[str],
    cache_dir: str | Path,
    *,
    force_refresh: bool = False,
    ingest_workers: int = 1,
) -> tuple[Path, bool]:
    source_file = Path(source_path)
    if source_kind == "hapstore":
        if not source_file.exists():
            raise FileNotFoundError(f"Donor HapStore not found: {source_file}")
        return source_file, True
    if source_kind != "vcf":
        raise ValueError(f"Unsupported donor source kind: {source_kind!r}")

    cache_path = donor_panel_cache_path(
        cache_dir=cache_dir,
        source_path=source_path,
        source_kind=source_kind,
        chrom=chrom,
        donor_ids=donor_ids,
    )
    if not force_refresh and _cache_matches_source(cache_path, source_file, source_kind):
        return cache_path, True

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache_dir = Path(
        tempfile.mkdtemp(prefix=f".{cache_path.name}.tmp-", dir=str(cache_path.parent))
    )
    try:
        ingest_vcf(
            vcf_path=source_file,
            output_zarr=tmp_cache_dir,
            chrom=chrom,
            sample_ids=list(donor_ids),
            workers=max(1, int(ingest_workers)),
        )
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)
        metadata_path = donor_panel_cache_metadata_path(cache_path)
        metadata_path.unlink(missing_ok=True)
        tmp_cache_dir.replace(cache_path)
        _write_cache_metadata(cache_path, source_path=source_file, source_kind=source_kind)
    except Exception:
        shutil.rmtree(tmp_cache_dir, ignore_errors=True)
        raise
    return cache_path, False


def load_donor_haplotypes_from_source(
    source_path: str | Path,
    source_kind: str,
    chrom: int | str,
    donor_ids: list[str],
    cache_dir: str | Path,
    force_refresh: bool = False,
    ingest_workers: int = 1,
) -> tuple[PanelChromData, Path, bool]:
    donor_ids = list(donor_ids)
    hapstore_path, cache_hit = ensure_donor_panel_hapstore(
        source_path=source_path,
        source_kind=source_kind,
        chrom=chrom,
        donor_ids=donor_ids,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        ingest_workers=ingest_workers,
    )
    panel = load_donor_haplotypes_hapstore(
        hapstore_path=hapstore_path,
        chrom=chrom,
        donor_ids=donor_ids,
    )
    return panel, hapstore_path, cache_hit


def subset_panel_donors(panel: PanelChromData, donor_ids: list[str]) -> PanelChromData:
    donor_ids = list(donor_ids)
    index_by_donor = {sid: i for i, sid in enumerate(panel.donor_ids)}
    missing = [sid for sid in donor_ids if sid not in index_by_donor]
    if missing:
        raise ValueError(
            f"{len(missing)} donor IDs not present in preloaded panel; first={missing[:5]}"
        )

    row_idx = np.empty(2 * len(donor_ids), dtype=np.int64)
    for i, sid in enumerate(donor_ids):
        donor_idx = index_by_donor[sid]
        row_idx[2 * i] = 2 * donor_idx
        row_idx[2 * i + 1] = 2 * donor_idx + 1

    return PanelChromData(
        variants=panel.variants,
        donor_haps=panel.donor_haps[row_idx, :],
        donor_ids=donor_ids,
    )
