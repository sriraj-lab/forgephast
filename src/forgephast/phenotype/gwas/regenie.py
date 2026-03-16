"""REGENIE GWAS engine implementation."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict

from .base import GwasEngine, GwasTask, merge_covariates_for_gwas
from ..config import RegenieEngineConfig

logger = logging.getLogger(__name__)


class RegenieEngine(GwasEngine):
    """Run quantitative trait GWAS using REGENIE step 1 + step 2."""

    def __init__(self, cfg: RegenieEngineConfig):
        self.cfg = cfg

    def name(self) -> str:
        """Engine key."""
        return "regenie"

    def run(self, task: GwasTask, run_dir: Path) -> Dict[str, str]:
        """Execute one REGENIE GWAS task."""
        out_dir = run_dir / "regenie"
        out_dir.mkdir(parents=True, exist_ok=True)

        covar_path = out_dir / "merged_covariates.tsv"
        covar_cols = merge_covariates_for_gwas(
            covariates_file=task.covariates_file,
            pcs_file=task.global_pcs_file,
            out_path=covar_path,
        )
        covar_col_list = ",".join(covar_cols)
        chr_list = ",".join(str(chrom) for chrom in task.chromosomes) if task.chromosomes else ""

        step1_prefix = out_dir / "step1"
        cmd1 = [
            self.cfg.path,
            "--step",
            "1",
            "--pgen",
            task.pgen_prefix,
            "--keep",
            task.keep_file,
            "--phenoFile",
            task.phenotype_file,
            "--phenoCol",
            "PHENO",
            "--covarFile",
            str(covar_path),
            "--covarColList",
            covar_col_list,
            "--qt",
        ]
        if chr_list:
            cmd1.extend(["--chrList", chr_list])
        cmd1.extend([
            "--bsize",
            str(self.cfg.bsize_step1),
            "--threads",
            str(self.cfg.threads),
            "--out",
            str(step1_prefix),
        ])
        cmd1.extend(list(self.cfg.extra_args_step1))

        logger.info("Running REGENIE step 1: %s", " ".join(cmd1))
        step1 = subprocess.run(cmd1, capture_output=True, text=True)
        if step1.returncode != 0:
            raise RuntimeError(
                "REGENIE step 1 failed\n"
                f"CMD: {' '.join(cmd1)}\n"
                f"STDOUT:\n{step1.stdout}\n"
                f"STDERR:\n{step1.stderr}"
            )

        pred_list = Path(f"{step1_prefix}_pred.list")
        if not pred_list.exists():
            raise RuntimeError(f"REGENIE step 1 missing prediction list: {pred_list}")

        step2_prefix = out_dir / "step2"
        cmd2 = [
            self.cfg.path,
            "--step",
            "2",
            "--pgen",
            task.pgen_prefix,
            "--keep",
            task.keep_file,
            "--phenoFile",
            task.phenotype_file,
            "--phenoCol",
            "PHENO",
            "--covarFile",
            str(covar_path),
            "--covarColList",
            covar_col_list,
            "--pred",
            str(pred_list),
            "--qt",
        ]
        if chr_list:
            cmd2.extend(["--chrList", chr_list])
        cmd2.extend([
            "--bsize",
            str(self.cfg.bsize_step2),
            "--threads",
            str(self.cfg.threads),
            "--out",
            str(step2_prefix),
        ])
        cmd2.extend(list(self.cfg.extra_args_step2))

        logger.info("Running REGENIE step 2: %s", " ".join(cmd2))
        step2 = subprocess.run(cmd2, capture_output=True, text=True)
        if step2.returncode != 0:
            raise RuntimeError(
                "REGENIE step 2 failed\n"
                f"CMD: {' '.join(cmd2)}\n"
                f"STDOUT:\n{step2.stdout}\n"
                f"STDERR:\n{step2.stderr}"
            )

        # REGENIE writes per-trait files like step2_<trait>.regenie.
        candidates = sorted(out_dir.glob("step2*.regenie*"))
        primary = next((p for p in candidates if p.stat().st_size > 0), None)
        if primary is None:
            raise RuntimeError(
                f"REGENIE step 2 output missing/empty under {out_dir} (expected step2*.regenie*)"
            )

        return {
            "engine": self.name(),
            "output_dir": str(out_dir),
            "primary_output": str(primary),
        }
