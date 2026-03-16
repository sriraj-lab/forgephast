"""Chromosome naming helpers."""

from __future__ import annotations


def normalize_chrom(chrom: str | int, prefix: bool = True) -> str:
    """Normalize chromosome naming to `chrN` or `N` style."""
    base = str(chrom)
    if base.startswith("chr"):
        base = base[3:]

    if prefix:
        return f"chr{base}"
    return base


def candidate_contigs(chrom: str | int) -> list[str]:
    """Return possible VCF contig aliases for a chromosome."""
    plain = normalize_chrom(chrom, prefix=False)
    prefixed = f"chr{plain}"
    if prefixed == plain:
        return [plain]
    return [prefixed, plain]


def uses_chr_prefix(contigs: list[str]) -> bool:
    """Infer whether contig names use `chr` prefix from a contig list."""
    if not contigs:
        return True
    return contigs[0].startswith("chr")
