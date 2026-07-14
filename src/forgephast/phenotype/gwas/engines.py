"""Built-in GWAS engine implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..._internal.command import run_checked_command
from .base import (
    GwasEngine,
    GwasResult,
    GwasTask,
    merge_covariates_for_gwas,
    read_prepared_covariate_columns,
)
from .regenie_step1 import prepare_regenie_step1_variant_list

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

        if task.merged_covariates_file:
            covar_path = Path(task.merged_covariates_file)
            covar_cols = read_prepared_covariate_columns(str(covar_path))
        else:
            covar_path = out_dir / "merged_covariates.tsv"
            covar_cols = merge_covariates_for_gwas(
                covariates_file=task.covariates_file,
                pcs_file=task.global_pcs_file,
                out_path=covar_path,
            )

        out_prefix = out_dir / "assoc"
        expected_glob = str(out_dir / "assoc*.glm.*")
        expected_log = str(out_dir / "assoc.log")
        logger.info(
            "[gwas-monitor] engine=plink2 rep=%d cohort=%s out_dir=%s primary_glob=%s log=%s",
            int(task.replicate),
            task.cohort,
            out_dir,
            expected_glob,
            expected_log,
        )
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
        if str(task.trait_type) == "binary":
            cmd.append("--1")
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
            ]
        )
        if self.cfg.memory_mb is not None:
            cmd.extend(["--memory", str(int(self.cfg.memory_mb))])
        cmd.extend(
            [
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

        logger.info(
            "[gwas-monitor] engine=plink2 rep=%d cohort=%s completed primary_output=%s",
            int(task.replicate),
            task.cohort,
            candidates[0],
        )

        return GwasResult(
            engine=self.name(),
            output_dir=str(out_dir),
            primary_output=str(candidates[0]),
            metadata={
                "monitor_log": expected_log,
                "monitor_primary_glob": expected_glob,
            },
        )


class RegenieEngine(GwasEngine):
    """Run quantitative trait GWAS using REGENIE step 1 + step 2."""

    def __init__(
        self,
        cfg: "RegenieEngineConfig",
        plink2_cfg: "Plink2EngineConfig | None" = None,
    ):
        self.cfg = cfg
        self.plink2_cfg = plink2_cfg

    def name(self) -> str:
        return "regenie"

    @staticmethod
    def _choose_primary_output(out_dir: Path) -> Path | None:
        candidates = [
            path
            for path in sorted(out_dir.glob("step2*.regenie*"))
            if path.exists() and path.stat().st_size > 0
        ]
        if not candidates:
            return None

        assoc_candidates = [
            path
            for path in candidates
            if path.name.endswith(".regenie") or path.name.endswith(".regenie.gz")
        ]
        if assoc_candidates:
            return assoc_candidates[0]

        non_ydict = [path for path in candidates if not path.name.endswith(".Ydict")]
        if non_ydict:
            return non_ydict[0]
        return candidates[0]

    def run(self, task: GwasTask, run_dir: Path, prepared_state: object | None = None) -> GwasResult:
        _ = prepared_state
        out_dir = run_dir / "regenie"
        out_dir.mkdir(parents=True, exist_ok=True)

        if task.merged_covariates_file:
            covar_path = Path(task.merged_covariates_file)
            covar_cols = read_prepared_covariate_columns(str(covar_path))
        else:
            covar_path = out_dir / "merged_covariates.tsv"
            covar_cols = merge_covariates_for_gwas(
                covariates_file=task.covariates_file,
                pcs_file=task.global_pcs_file,
                out_path=covar_path,
            )
        covar_col_list = ",".join(covar_cols)
        chr_list = ",".join(str(chrom) for chrom in task.chromosomes) if task.chromosomes else ""

        step1_prefix = out_dir / "step1"
        step1_log = str(Path(f"{step1_prefix}.log"))
        step1_pred = str(Path(f"{step1_prefix}_pred.list"))
        step2_prefix = out_dir / "step2"
        step2_glob = str(out_dir / "step2*.regenie*")
        logger.info(
            "[gwas-monitor] engine=regenie rep=%d cohort=%s out_dir=%s step1_log=%s step1_pred=%s step2_glob=%s",
            int(task.replicate),
            task.cohort,
            out_dir,
            step1_log,
            step1_pred,
            step2_glob,
        )
        trait_flag = "--bt" if str(task.trait_type) == "binary" else "--qt"
        step1_variant_list = prepare_regenie_step1_variant_list(
            task=task,
            out_dir=out_dir,
            regenie_cfg=self.cfg,
            plink2_cfg=self.plink2_cfg,
        )
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
            trait_flag,
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
        if step1_variant_list is not None:
            cmd1.extend(["--extract", str(step1_variant_list.path)])
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
            trait_flag,
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

        primary = self._choose_primary_output(out_dir)
        if primary is None:
            raise RuntimeError(
                f"REGENIE step 2 output missing/empty under {out_dir} (expected step2*.regenie*)"
            )

        logger.info(
            "[gwas-monitor] engine=regenie rep=%d cohort=%s completed primary_output=%s",
            int(task.replicate),
            task.cohort,
            primary,
        )

        return GwasResult(
            engine=self.name(),
            output_dir=str(out_dir),
            primary_output=str(primary),
            metadata={
                "monitor_step1_log": step1_log,
                "monitor_step1_pred_list": step1_pred,
                "monitor_step2_glob": step2_glob,
                "step1_variant_mode": str(self.cfg.step1_variant_mode),
                "step1_variant_list": (
                    str(step1_variant_list.path) if step1_variant_list is not None else ""
                ),
                "step1_variant_count": (
                    int(step1_variant_list.variant_count) if step1_variant_list is not None else ""
                ),
                "step1_variant_warning_threshold": (
                    self.cfg.step1_variant_warning_threshold
                    if step1_variant_list is not None
                    else ""
                ),
                "step1_variant_warning_exceeded": (
                    bool(step1_variant_list.warning_threshold_exceeded)
                    if step1_variant_list is not None
                    else ""
                ),
                "step1_variant_cache_hit": (
                    bool(step1_variant_list.cache_hit) if step1_variant_list is not None else ""
                ),
            },
        )


__all__ = [
    "Plink2Engine",
    "RegenieEngine",
]
