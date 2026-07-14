"""Shared PLINK/PGEN artifact helpers for phenotype workflows."""

from __future__ import annotations

from pathlib import Path

from ..config import RunConfig


def pgen_triplet_exists(prefix: str | Path) -> bool:
    base = Path(prefix)
    return all(Path(f"{base}{extension}").exists() for extension in (".pgen", ".pvar", ".psam"))


def plink_memory_args(run_config: RunConfig) -> list[str]:
    memory_mb = getattr(run_config.gwas.plink2, "memory_mb", None)
    if memory_mb is None:
        return []
    return ["--memory", str(int(memory_mb))]
