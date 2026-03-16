"""PLINK2 GWAS engine implementation."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict

from .base import GwasEngine, GwasTask, merge_covariates_for_gwas
from ..config import Plink2EngineConfig

logger = logging.getLogger(__name__)


class Plink2Engine(GwasEngine):
    """Run quantitative GWAS with PLINK2 --glm."""

    def __init__(self, cfg: Plink2EngineConfig):
        self.cfg = cfg

    def name(self) -> str:
        """Engine key."""
        return "plink2"

    def run(self, task: GwasTask, run_dir: Path) -> Dict[str, str]:
        """Execute one PLINK2 GWAS task."""
        out_dir = run_dir / "plink2"
        out_dir.mkdir(parents=True, exist_ok=True)

        covar_path = out_dir / "merged_covariates.tsv"
        covar_cols = merge_covariates_for_gwas(
            covariates_file=task.covariates_file,
            pcs_file=task.global_pcs_file,
            out_path=covar_path,
        )

        out_prefix = out_dir / "assoc"
        cmd = [
            self.cfg.path,
            "--pfile",
            task.pgen_prefix,
            "--keep",
            task.keep_file,
            "--pheno",
            task.phenotype_file,
            "--pheno-name",
            "PHENO",
        ]
        if task.chromosomes:
            cmd.extend(["--chr", ",".join(str(chrom) for chrom in task.chromosomes)])
        cmd.extend([
            "--glm",
            "hide-covar",
            "--covar",
            str(covar_path),
            "--covar-name",
            ",".join(covar_cols),
            "--threads",
            str(self.cfg.threads),
            "--out",
            str(out_prefix),
        ])
        cmd.extend(list(self.cfg.extra_args))

        logger.info("Running PLINK2 GWAS: %s", " ".join(cmd))
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "PLINK2 GWAS failed\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        candidates = sorted(out_dir.glob(f"{out_prefix.name}.*.glm.*"))
        if not candidates:
            # Backward-compatible fallback for unexpected PLINK2 naming.
            candidates = sorted(out_dir.glob("assoc*.glm.*"))
        if not candidates:
            raise RuntimeError(f"PLINK2 GWAS produced no .glm output under {out_dir}")

        return {
            "engine": self.name(),
            "output_dir": str(out_dir),
            "primary_output": str(candidates[0]),
        }
