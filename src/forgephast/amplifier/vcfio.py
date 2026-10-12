"""VCF and PGEN I/O utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
import pysam


@dataclass
class VariantTable:
    """Aligned, variant-major metadata for one chromosome panel.

    All fields preserve a shared source-variant order. ``pos`` stores integer
    base-pair positions; ``vid``, ``ref``, and ``alt`` have one entry per
    position.
    """

    chrom: str
    pos: np.ndarray
    vid: list[str]
    ref: list[str]
    alt: list[str]


def _run_checked_command(cmd: list[str], label: str) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )



def convert_hapstore_to_pgen(
    hapstore_path: str | Path,
    out_prefix: str | Path,
    chrom: int | str,
    variant_block_size: int = 8192,
) -> Path:
    """Export one HapStore chromosome directly to a validated PGEN triplet.

    The bridge requires biallelic variants and fails on partially missing
    genotype pairs. It verifies that ``.pgen``, ``.pvar``, and ``.psam`` were
    written before returning the output prefix.
    """
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


def sample_manifest_from_vcf(vcf_path: str | Path) -> list[str]:
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        return list(vcf.header.samples)
