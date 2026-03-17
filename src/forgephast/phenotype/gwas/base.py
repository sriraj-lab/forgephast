"""GWAS engine contracts and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)


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


@dataclass
class GwasResult:
    """Normalized GWAS engine output summary."""

    engine: str
    output_dir: str
    primary_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class GwasEngine(ABC):
    """Base contract for phenotype GWAS engines."""

    @abstractmethod
    def name(self) -> str:
        """Return the stable engine key."""

    def prepare_batch(self, tasks: list[GwasTask], batch_dir: Path) -> Any | None:
        """Optionally prepare shared artifacts for a batch of tasks.

        Engines like SAIGE often need cohort- or run-level state such as null
        models or sparse GRMs that should be built once and then reused across
        task executions. Engines that do not need this can keep the default
        no-op implementation.
        """

        _ = tasks
        _ = batch_dir
        return None

    @abstractmethod
    def run(self, task: GwasTask, run_dir: Path, prepared_state: Any | None = None) -> GwasResult:
        """Execute one GWAS task."""


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
            df[id_col] = df[id_col].astype("string").fillna("").str.strip()

    merged = cov_df.merge(pcs_df, on=["FID", "IID"], how="inner", suffixes=("", "_pcs"))
    if merged.empty:
        pcs_iid = pcs_df.drop(columns=["FID"], errors="ignore").drop_duplicates(subset=["IID"])
        merged = cov_df.merge(pcs_iid, on=["IID"], how="inner", suffixes=("", "_pcs"))
    if merged.empty:
        raise ValueError("covariate/PC merge is empty")

    covar_cols = [c for c in merged.columns if c not in {"FID", "IID"}]
    merged.to_csv(out_path, sep="\t", index=False)
    return covar_cols


__all__ = [
    "GwasEngine",
    "GwasResult",
    "GwasTask",
    "merge_covariates_for_gwas",
]
