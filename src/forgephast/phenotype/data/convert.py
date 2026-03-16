"""Genotype conversion helpers to canonical PGEN format."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..config import GenotypeInputConfig

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    """Run subprocess command with checked error handling."""
    logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def ensure_pgen_input(geno_cfg: GenotypeInputConfig, run_tmp_dir: Path) -> str | None:
    """Return a PGEN prefix, converting genotype input if required."""
    if geno_cfg.format == "hapstore":
        return None

    if geno_cfg.format == "pgen":
        return geno_cfg.path_prefix

    if not geno_cfg.conversion_enabled:
        raise ValueError("conversion_enabled is false but non-pgen input was provided")

    run_tmp_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = run_tmp_dir / "converted_input"

    if geno_cfg.format == "vcf":
        cmd = [
            geno_cfg.plink2_path,
            "--vcf",
            geno_cfg.path_prefix,
            "dosage=DS-force",
            "--make-pgen",
            "--out",
            str(out_prefix),
        ]
    elif geno_cfg.format == "bfile":
        cmd = [
            geno_cfg.plink2_path,
            "--bfile",
            geno_cfg.path_prefix,
            "--make-pgen",
            "--out",
            str(out_prefix),
        ]
    else:
        raise ValueError(f"Unsupported genotype format: {geno_cfg.format}")

    _run(cmd)
    for ext in (".pgen", ".pvar", ".psam"):
        if not (Path(f"{out_prefix}{ext}")).exists():
            raise RuntimeError(f"Missing converted PGEN artifact: {out_prefix}{ext}")

    logger.info("Converted genotype input to PGEN: %s", out_prefix)
    return str(out_prefix)
