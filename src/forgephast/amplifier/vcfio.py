"""VCF and PGEN I/O utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pysam


@dataclass
class VariantTable:
    chrom: str
    pos: np.ndarray
    vid: list[str]
    ref: list[str]
    alt: list[str]


@dataclass
class PanelChromData:
    variants: VariantTable
    donor_haps: np.ndarray  # shape: (2 * n_donors, n_variants)
    donor_ids: list[str]


def _run_checked_command(cmd: list[str], label: str) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )



def _iter_records(vcf: pysam.VariantFile, chrom: str):
    aliases = [chrom]
    if chrom.startswith("chr"):
        aliases.append(chrom[3:])
    else:
        aliases.append(f"chr{chrom}")
    aliases = [a for i, a in enumerate(aliases) if a and a not in aliases[:i]]

    for alias in aliases:
        try:
            found = False
            for rec in vcf.fetch(alias):
                found = True
                yield rec
            if found:
                return
        except (ValueError, OSError):
            continue

    # Fallback for unindexed VCFs or non-standard contig names.
    alias_set = set(aliases)
    for rec in vcf:
        if rec.contig in alias_set:
            yield rec



def load_donor_haplotypes(vcf_path: str | Path, chrom: int | str, donor_ids: list[str]) -> PanelChromData:
    chrom_s = str(chrom)
    donor_ids = list(donor_ids)
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        missing = [sid for sid in donor_ids if sid not in list(vcf.header.samples)]
        if missing:
            raise ValueError(
                f"{len(missing)} donor IDs not present in panel VCF {vcf_path}; first={missing[:5]}"
            )

        cols: list[np.ndarray] = []
        pos: list[int] = []
        vid: list[str] = []
        ref: list[str] = []
        alt: list[str] = []
        out_chrom: str | None = None

        n_haps = 2 * len(donor_ids)

        for rec in _iter_records(vcf, chrom_s):
            # Keep biallelic SNPs only in v1.
            if len(rec.alleles) != 2:
                continue
            if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue

            col = np.zeros(n_haps, dtype=np.uint8)
            for i, sid in enumerate(donor_ids):
                gt = rec.samples[sid].get("GT")
                a0 = 0
                a1 = 0
                if gt and len(gt) >= 2:
                    if gt[0] is not None and gt[0] > 0:
                        a0 = 1
                    if gt[1] is not None and gt[1] > 0:
                        a1 = 1
                col[2 * i] = a0
                col[2 * i + 1] = a1

            cols.append(col)
            pos.append(int(rec.pos))
            vid.append(str(rec.id) if rec.id else ".")
            ref.append(rec.ref)
            alt.append(rec.alts[0])
            if out_chrom is None:
                out_chrom = str(rec.contig)

    if not cols:
        raise ValueError(f"No eligible biallelic SNPs found in {vcf_path} chrom={chrom_s}")

    donor_haps = np.stack(cols, axis=1)
    variants = VariantTable(
        chrom=out_chrom if out_chrom is not None else chrom_s,
        pos=np.asarray(pos, dtype=np.int64),
        vid=vid,
        ref=ref,
        alt=alt,
    )
    return PanelChromData(variants=variants, donor_haps=donor_haps, donor_ids=donor_ids)


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
    source_abspath = str(Path(source_path).resolve())
    chrom_s = str(chrom)
    digest = hashlib.sha256()
    digest.update(source_kind.encode("utf-8"))
    digest.update(b"\n")
    digest.update(source_abspath.encode("utf-8"))
    digest.update(b"\n")
    digest.update(chrom_s.encode("utf-8"))
    digest.update(b"\n")
    for sid in donor_ids:
        digest.update(str(sid).encode("utf-8"))
        digest.update(b"\n")
    key = digest.hexdigest()[:16]
    stem = Path(source_path).name
    return Path(cache_dir) / f"chr{chrom_s}" / f"{source_kind}.{stem}.{key}.npz"


def save_donor_panel_cache(
    cache_path: str | Path,
    panel: PanelChromData,
    source_path: str | Path,
    source_kind: str,
) -> Path:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    source_size = -1
    source_mtime_ns = -1
    if source.exists():
        stat = source.stat()
        source_size = int(stat.st_size)
        source_mtime_ns = int(stat.st_mtime_ns)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".tmp",
        dir=str(cache_path.parent),
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        np.savez_compressed(
            handle,
            chrom=np.asarray([panel.variants.chrom], dtype=np.str_),
            pos=panel.variants.pos.astype(np.int64),
            vid=np.asarray(panel.variants.vid, dtype=np.str_),
            ref=np.asarray(panel.variants.ref, dtype=np.str_),
            alt=np.asarray(panel.variants.alt, dtype=np.str_),
            donor_ids=np.asarray(panel.donor_ids, dtype=np.str_),
            donor_haps=panel.donor_haps.astype(np.uint8),
            source_kind=np.asarray([str(source_kind)], dtype=np.str_),
            source_path=np.asarray([str(source.resolve())], dtype=np.str_),
            source_size=np.asarray([source_size], dtype=np.int64),
            source_mtime_ns=np.asarray([source_mtime_ns], dtype=np.int64),
        )
    tmp_path.replace(cache_path)
    return cache_path


def load_donor_panel_cache(cache_path: str | Path) -> PanelChromData:
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        variants = VariantTable(
            chrom=str(payload["chrom"][0]),
            pos=payload["pos"].astype(np.int64, copy=False),
            vid=[str(x) for x in payload["vid"].tolist()],
            ref=[str(x) for x in payload["ref"].tolist()],
            alt=[str(x) for x in payload["alt"].tolist()],
        )
        donor_ids = [str(x) for x in payload["donor_ids"].tolist()]
        donor_haps = payload["donor_haps"].astype(np.uint8, copy=False)
    return PanelChromData(variants=variants, donor_haps=donor_haps, donor_ids=donor_ids)


def _cache_matches_source(cache_path: Path, source_path: Path, source_kind: str) -> bool:
    if not source_path.exists():
        return True
    with np.load(cache_path, allow_pickle=False) as payload:
        cached_kind = str(payload["source_kind"][0]) if "source_kind" in payload else "vcf"
        if cached_kind != str(source_kind):
            return False
        cached_size = int(payload["source_size"][0]) if "source_size" in payload else -1
        cached_mtime_ns = int(payload["source_mtime_ns"][0]) if "source_mtime_ns" in payload else -1
    stat = source_path.stat()
    return cached_size == int(stat.st_size) and cached_mtime_ns == int(stat.st_mtime_ns)


def load_donor_haplotypes_cached(
    source_path: str | Path,
    source_kind: str,
    chrom: int | str,
    donor_ids: list[str],
    cache_dir: str | Path,
    force_refresh: bool = False,
) -> tuple[PanelChromData, Path, bool]:
    if source_kind not in {"vcf", "hapstore"}:
        raise ValueError(f"Unsupported donor source kind: {source_kind!r}")
    donor_ids = list(donor_ids)
    cache_path = donor_panel_cache_path(
        cache_dir=cache_dir,
        source_path=source_path,
        source_kind=source_kind,
        chrom=chrom,
        donor_ids=donor_ids,
    )
    source_file = Path(source_path)
    if not force_refresh and cache_path.exists():
        try:
            if _cache_matches_source(cache_path, source_file, source_kind):
                panel = load_donor_panel_cache(cache_path)
                return panel, cache_path, True
        except Exception:
            # Corrupt cache or schema mismatch: rebuild from source panel.
            pass

    if source_kind == "vcf":
        panel = load_donor_haplotypes(vcf_path=source_path, chrom=chrom, donor_ids=donor_ids)
    else:
        panel = load_donor_haplotypes_hapstore(
            hapstore_path=source_path,
            chrom=chrom,
            donor_ids=donor_ids,
        )
    save_donor_panel_cache(
        cache_path=cache_path,
        panel=panel,
        source_path=source_path,
        source_kind=source_kind,
    )
    return panel, cache_path, False


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



def write_synthetic_vcf(
    output_vcf_gz: str | Path,
    variants: VariantTable,
    sample_ids: list[str],
    haplotypes: np.ndarray,
) -> Path:
    """Write phased GT VCF and index with tabix."""
    output_vcf_gz = Path(output_vcf_gz)
    output_vcf_gz.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(sample_ids)
    if haplotypes.shape[0] != 2 * n_samples:
        raise ValueError(
            f"haplotype rows ({haplotypes.shape[0]}) must equal 2 * n_samples ({2 * n_samples})"
        )
    if haplotypes.shape[1] != len(variants.pos):
        raise ValueError("haplotype variant count does not match variant table")

    temp_vcf = output_vcf_gz.with_suffix("")  # drops .gz
    if temp_vcf.suffix != ".vcf":
        temp_vcf = output_vcf_gz.with_name(output_vcf_gz.name.replace(".vcf.gz", ".vcf"))

    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.contigs.add(variants.chrom)
    header.formats.add("GT", number=1, type="String", description="Phased genotype")
    for sid in sample_ids:
        header.add_sample(sid)

    with pysam.VariantFile(str(temp_vcf), "w", header=header) as out_vcf:
        for j, pos in enumerate(variants.pos.tolist()):
            rec = out_vcf.new_record(
                contig=variants.chrom,
                start=int(pos) - 1,
                stop=int(pos),
                id=variants.vid[j],
                alleles=(variants.ref[j], variants.alt[j]),
            )
            for i, sid in enumerate(sample_ids):
                rec.samples[sid]["GT"] = (
                    int(haplotypes[2 * i, j]),
                    int(haplotypes[2 * i + 1, j]),
                )
                rec.samples[sid].phased = True
            out_vcf.write(rec)

    pysam.tabix_compress(str(temp_vcf), str(output_vcf_gz), force=True)
    pysam.tabix_index(str(output_vcf_gz), preset="vcf", force=True)
    temp_vcf.unlink(missing_ok=True)
    return output_vcf_gz



def convert_vcf_to_pgen(
    vcf_path: str | Path,
    out_prefix: str | Path,
    plink2_path: str = "plink2",
    threads: int = 1,
) -> Path:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    cmd = [
        plink2_path,
        "--vcf",
        str(vcf_path),
        "--threads",
        str(threads),
        "--make-pgen",
        "--out",
        str(out_prefix),
    ]
    _run_checked_command(cmd, label="PLINK2 --make-pgen")
    for ext in (".pgen", ".pvar", ".psam"):
        path = Path(f"{out_prefix}{ext}")
        if not path.exists():
            raise RuntimeError(f"Missing PLINK2 output {path}")
    return out_prefix


def convert_hapstore_to_pgen(
    hapstore_path: str | Path,
    out_prefix: str | Path,
    chrom: int | str,
    variant_block_size: int = 8192,
) -> Path:
    """Convert HapStore directly to PGEN using hapstore fast path."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if int(variant_block_size) <= 0:
        raise ValueError("variant_block_size must be > 0")

    from hapstore import export_pgen

    export_pgen(
        store_path=hapstore_path,
        out_prefix=out_prefix,
        chrom=f"chr{chrom}" if str(chrom).isdigit() else str(chrom),
        variant_block_size=int(variant_block_size),
        strict_biallelic=True,
        on_partial_missing="error",
    )
    for ext in (".pgen", ".pvar", ".psam"):
        path = Path(f"{out_prefix}{ext}")
        if not path.exists():
            raise RuntimeError(f"Missing direct HapStore->PGEN output {path}")
    return out_prefix


def convert_hapstore_to_vcf(
    hapstore_path: str | Path,
    output_vcf: str | Path,
    chrom: int | str,
    *,
    variant_block_size: int = 1024,
    plink2_location: str | None = None,
    keep_intermediate_files: bool = False,
    n_shards: int = 1,
) -> Path:
    """Convert HapStore to phased VCF.

    When `plink2_location` is provided, route export through
    HapStore -> PGEN -> PLINK2 VCF export. When `None`, use default
    VCF writer behavior from hapstore.
    """
    output_vcf = Path(output_vcf)
    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    if int(variant_block_size) <= 0:
        raise ValueError("variant_block_size must be > 0")
    if int(n_shards) <= 0:
        raise ValueError("n_shards must be > 0")

    from hapstore import export_vcf

    return export_vcf(
        store_path=hapstore_path,
        output_vcf=output_vcf,
        chrom=f"chr{chrom}" if str(chrom).isdigit() else str(chrom),
        bgzip=True,
        tabix_index=True,
        strict_variant_info=True,
        variant_block_size=int(variant_block_size),
        plink2_location=plink2_location,
        keep_intermediate_files=bool(keep_intermediate_files),
        n_shards=int(n_shards),
    )


def convert_vcf_to_bed_plink19(
    vcf_path: str | Path,
    out_prefix: str | Path,
    plink1_path: str = "plink",
    threads: int = 1,
) -> Path:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    cmd = [
        plink1_path,
        "--vcf",
        str(vcf_path),
        "--double-id",
        "--allow-extra-chr",
        "--threads",
        str(threads),
        "--make-bed",
        "--out",
        str(out_prefix),
    ]
    _run_checked_command(cmd, label="PLINK1.9 --make-bed")
    for ext in (".bed", ".bim", ".fam"):
        path = Path(f"{out_prefix}{ext}")
        if not path.exists():
            raise RuntimeError(f"Missing PLINK1.9 output {path}")
    return out_prefix


def merge_shard_beds_plink19(
    shard_prefixes: list[Path],
    output_prefix: Path,
    plink1_path: str = "plink",
    threads: int = 1,
) -> Path:
    if not shard_prefixes:
        raise ValueError("merge_shard_beds_plink19 requires at least one shard prefix")
    if threads <= 0:
        raise ValueError("threads must be > 0")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for prefix in shard_prefixes:
        for ext in (".bed", ".bim", ".fam"):
            path = Path(f"{prefix}{ext}")
            if not path.exists():
                raise FileNotFoundError(f"Missing shard BED component: {path}")

    if len(shard_prefixes) == 1:
        src = shard_prefixes[0]
        for ext in (".bed", ".bim", ".fam"):
            shutil.copy2(Path(f"{src}{ext}"), Path(f"{output_prefix}{ext}"))
        return output_prefix

    first_prefix = shard_prefixes[0]
    with tempfile.NamedTemporaryFile("w", suffix=".merge.list", delete=False, encoding="utf-8") as handle:
        merge_list = Path(handle.name)
        for prefix in shard_prefixes[1:]:
            handle.write(f"{prefix}.bed {prefix}.bim {prefix}.fam\n")

    try:
        cmd = [
            plink1_path,
            "--bfile",
            str(first_prefix),
            "--merge-list",
            str(merge_list),
            "--threads",
            str(threads),
            "--make-bed",
            "--out",
            str(output_prefix),
        ]
        _run_checked_command(cmd, label="PLINK1.9 --merge-list")
    finally:
        merge_list.unlink(missing_ok=True)

    for ext in (".bed", ".bim", ".fam"):
        path = Path(f"{output_prefix}{ext}")
        if not path.exists():
            raise RuntimeError(f"Missing PLINK1.9 merged output {path}")
    return output_prefix


def merge_shard_pgens_plink2(
    shard_prefixes: list[Path],
    output_prefix: Path,
    plink2_path: str = "plink2",
    threads: int = 1,
) -> Path:
    if not shard_prefixes:
        raise ValueError("merge_shard_pgens_plink2 requires at least one shard prefix")
    if threads <= 0:
        raise ValueError("threads must be > 0")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for prefix in shard_prefixes:
        for ext in (".pgen", ".pvar", ".psam"):
            path = Path(f"{prefix}{ext}")
            if not path.exists():
                raise FileNotFoundError(f"Missing shard PGEN component: {path}")

    with tempfile.NamedTemporaryFile("w", suffix=".pmerge.list", delete=False, encoding="utf-8") as handle:
        merge_list = Path(handle.name)
        for prefix in shard_prefixes:
            handle.write(f"{prefix}\n")

    try:
        cmd = [
            plink2_path,
            "--threads",
            str(threads),
            "--pmerge-list",
            str(merge_list),
            "pfile",
            "--make-pgen",
            "--out",
            str(output_prefix),
        ]
        _run_checked_command(cmd, label="PLINK2 --pmerge-list")
    finally:
        merge_list.unlink(missing_ok=True)

    for ext in (".pgen", ".pvar", ".psam"):
        path = Path(f"{output_prefix}{ext}")
        if not path.exists():
            raise RuntimeError(f"Missing PLINK2 merged output {path}")
    return output_prefix



def merge_shard_vcfs(shard_vcfs: list[Path], output_vcf_gz: Path) -> Path:
    if not shard_vcfs:
        raise ValueError("merge_shard_vcfs requires at least one shard VCF")

    output_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    readers = [pysam.VariantFile(str(path), "r") for path in shard_vcfs]

    try:
        header = pysam.VariantHeader()
        header.add_meta("fileformat", "VCFv4.2")

        first = readers[0]
        for contig_name, contig in first.header.contigs.items():
            if contig.length is not None:
                header.contigs.add(contig_name, length=contig.length)
            else:
                header.contigs.add(contig_name)

        # GT is all we need for this package.
        header.formats.add("GT", number=1, type="String", description="Phased genotype")

        for reader in readers:
            for sid in list(reader.header.samples):
                header.add_sample(sid)

        temp_vcf = output_vcf_gz.with_name(output_vcf_gz.name.replace(".vcf.gz", ".vcf"))
        with pysam.VariantFile(str(temp_vcf), "w", header=header) as writer:
            iterators = [iter(r) for r in readers]
            while True:
                chunk = []
                try:
                    for it in iterators:
                        chunk.append(next(it))
                except StopIteration:
                    break

                ref_rec = chunk[0]
                key = (ref_rec.contig, ref_rec.pos, ref_rec.ref, tuple(ref_rec.alts or []))
                for rec in chunk[1:]:
                    other_key = (rec.contig, rec.pos, rec.ref, tuple(rec.alts or []))
                    if other_key != key:
                        raise ValueError("Shard VCF variants are misaligned; cannot merge")

                out_rec = writer.new_record(
                    contig=ref_rec.contig,
                    start=ref_rec.start,
                    stop=ref_rec.stop,
                    id=ref_rec.id if ref_rec.id else ".",
                    alleles=ref_rec.alleles,
                )
                for rec in chunk:
                    for sid in list(rec.samples):
                        out_rec.samples[sid]["GT"] = rec.samples[sid].get("GT")
                        out_rec.samples[sid].phased = True
                writer.write(out_rec)

        pysam.tabix_compress(str(temp_vcf), str(output_vcf_gz), force=True)
        pysam.tabix_index(str(output_vcf_gz), preset="vcf", force=True)
        temp_vcf.unlink(missing_ok=True)
        return output_vcf_gz
    finally:
        for r in readers:
            r.close()


def merge_shard_vcfs_bcftools(
    shard_vcfs: list[Path],
    output_vcf_gz: Path,
    bcftools_path: str = "bcftools",
    threads: int = 1,
    no_index: bool = False,
) -> Path:
    if not shard_vcfs:
        raise ValueError("merge_shard_vcfs_bcftools requires at least one shard VCF")
    if threads <= 0:
        raise ValueError("threads must be > 0")

    output_vcf_gz.parent.mkdir(parents=True, exist_ok=True)
    output_vcf_gz.unlink(missing_ok=True)
    Path(f"{output_vcf_gz}.tbi").unlink(missing_ok=True)

    cmd = [bcftools_path, "merge"]
    if no_index:
        cmd.append("--no-index")
    if threads > 1:
        cmd.extend(["--threads", str(threads)])
    cmd.extend(["-Oz", "-o", str(output_vcf_gz)])
    cmd.extend(str(path) for path in shard_vcfs)
    _run_checked_command(cmd, label="bcftools merge")

    _run_checked_command(
        [bcftools_path, "index", "-t", str(output_vcf_gz)],
        label="bcftools index",
    )
    return output_vcf_gz



def sample_manifest_from_vcf(vcf_path: str | Path) -> list[str]:
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        return list(vcf.header.samples)
