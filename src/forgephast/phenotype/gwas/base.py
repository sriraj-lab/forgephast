"""GWAS engine abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class GwasTask:
    """One GWAS run task from manifest row."""

    run_name: str
    replicate: int
    architecture: str
    cohort: str
    pgen_prefix: str
    keep_file: str
    phenotype_file: str
    covariates_file: str
    global_pcs_file: str
    output_dir: str
    chromosomes: List[int] | None = None


class GwasEngine(ABC):
    """Abstract GWAS engine interface."""

    @abstractmethod
    def name(self) -> str:
        """Engine name key."""

    @abstractmethod
    def run(self, task: GwasTask, run_dir: Path) -> Dict[str, str]:
        """Run GWAS task and return output metadata."""


def merge_covariates_for_gwas(
    covariates_file: str,
    pcs_file: str,
    out_path: Path,
) -> List[str]:
    """Merge non-genetic covariates + global PCs into one covariate file.

    Returns ordered covariate column names.
    """
    import pandas as pd

    cov_df = pd.read_csv(covariates_file, sep="\t")
    pcs_df = pd.read_csv(pcs_file, sep="\t")

    for df_name, df in (("covariates", cov_df), ("global_pcs", pcs_df)):
        missing = [col for col in ("FID", "IID") if col not in df.columns]
        if missing:
            raise ValueError(
                f"{df_name} file is missing required ID columns {missing}: "
                f"{covariates_file if df_name == 'covariates' else pcs_file}"
            )
        for id_col in ("FID", "IID"):
            # Keep merge semantics stable even if one side was parsed as numeric.
            df[id_col] = df[id_col].astype("string").fillna("").str.strip()

    merged = cov_df.merge(pcs_df, on=["FID", "IID"], how="inner", suffixes=("", "_pcs"))
    if merged.empty:
        # Some backends emit PCA tables with FID=IID even when phenotypes use FID=0.
        # Fall back to IID-only alignment and retain FID from covariates.
        pcs_iid = pcs_df.drop(columns=["FID"], errors="ignore").drop_duplicates(subset=["IID"])
        merged = cov_df.merge(pcs_iid, on=["IID"], how="inner", suffixes=("", "_pcs"))
    if merged.empty:
        raise ValueError("covariate/PC merge is empty")

    covar_cols = [c for c in merged.columns if c not in {"FID", "IID"}]
    merged.to_csv(out_path, sep="\t", index=False)
    return covar_cols
