"""Built-in GWAS engine implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..._internal.command import run_checked_command
from .base import GwasEngine, GwasResult, GwasTask, merge_covariates_for_gwas

if TYPE_CHECKING:
    from ..config_schema import Plink2EngineConfig, RegenieEngineConfig

logger = logging.getLogger(__name__)


class Plink2Engine(GwasEngine):
    """Run quantitative GWAS with PLINK2 --glm."""

    def __init__(self, cfg: "Plink2EngineConfig"):
        self.cfg = cfg

    def name(self) -> str:
        return "plink2"

    def run(self, task: GwasTask, run_dir: Path, prepared_state: object | None = None) -> GwasResult:
        _ = prepared_state
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
        glm_modifiers = ["hide-covar"]
        if not covar_cols:
            glm_modifiers.append("allow-no-covars")
        cmd.extend(["--glm", *glm_modifiers])
        if covar_cols:
            cmd.extend(
                [
                    "--covar",
                    str(covar_path),
                    "--covar-name",
                    ",".join(covar_cols),
                    "--covar-variance-standardize",
                ]
            )
        cmd.extend(
            [
                "--threads",
                str(self.cfg.threads),
                "--out",
                str(out_prefix),
            ]
        )
        cmd.extend(list(self.cfg.extra_args))

        run_checked_command(cmd, label="PLINK2 GWAS", logger=logger)

        candidates = sorted(out_dir.glob(f"{out_prefix.name}.*.glm.*"))
        if not candidates:
            candidates = sorted(out_dir.glob("assoc*.glm.*"))
        if not candidates:
            raise RuntimeError(f"PLINK2 GWAS produced no .glm output under {out_dir}")

        return GwasResult(
            engine=self.name(),
            output_dir=str(out_dir),
            primary_output=str(candidates[0]),
        )


class RegenieEngine(GwasEngine):
    """Run quantitative trait GWAS using REGENIE step 1 + step 2."""

    def __init__(self, cfg: "RegenieEngineConfig"):
        self.cfg = cfg

    def name(self) -> str:
        return "regenie"

    def run(self, task: GwasTask, run_dir: Path, prepared_state: object | None = None) -> GwasResult:
        _ = prepared_state
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
            "--qt",
        ]
        if covar_cols:
            cmd1.extend(
                [
                    "--covarFile",
                    str(covar_path),
                    "--covarColList",
                    covar_col_list,
                ]
            )
        if chr_list:
            cmd1.extend(["--chrList", chr_list])
        cmd1.extend(
            [
                "--bsize",
                str(self.cfg.bsize_step1),
                "--threads",
                str(self.cfg.threads),
                "--out",
                str(step1_prefix),
            ]
        )
        cmd1.extend(list(self.cfg.extra_args_step1))
        run_checked_command(cmd1, label="REGENIE step 1", logger=logger)

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
            "--pred",
            str(pred_list),
            "--qt",
        ]
        if covar_cols:
            cmd2.extend(
                [
                    "--covarFile",
                    str(covar_path),
                    "--covarColList",
                    covar_col_list,
                ]
            )
        if chr_list:
            cmd2.extend(["--chrList", chr_list])
        cmd2.extend(
            [
                "--bsize",
                str(self.cfg.bsize_step2),
                "--threads",
                str(self.cfg.threads),
                "--out",
                str(step2_prefix),
            ]
        )
        cmd2.extend(list(self.cfg.extra_args_step2))
        run_checked_command(cmd2, label="REGENIE step 2", logger=logger)

        candidates = sorted(out_dir.glob("step2*.regenie*"))
        primary = next((p for p in candidates if p.stat().st_size > 0), None)
        if primary is None:
            raise RuntimeError(
                f"REGENIE step 2 output missing/empty under {out_dir} (expected step2*.regenie*)"
            )

        return GwasResult(
            engine=self.name(),
            output_dir=str(out_dir),
            primary_output=str(primary),
        )


__all__ = [
    "Plink2Engine",
    "RegenieEngine",
]
