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
    merged_covariates_file: str = ""
    trait_type: str = "quantitative"
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

    if "age" in merged.columns and "age_centered" in merged.columns:
        logger.info("Dropping raw age from GWAS covariates because age_centered is present")
        merged = merged.drop(columns=["age"])

    covar_cols = [c for c in merged.columns if c not in {"FID", "IID"}]

    constant_cols = [col for col in covar_cols if merged[col].nunique(dropna=False) <= 1]
    if constant_cols:
        logger.info("Dropping constant GWAS covariates: %s", ",".join(constant_cols))
        merged = merged.drop(columns=constant_cols)
        covar_cols = [c for c in covar_cols if c not in set(constant_cols)]

    pc_cols = [col for col in covar_cols if col.startswith("PC")]
    if pc_cols and "population_code" in covar_cols:
        logger.info("Dropping population_code from GWAS covariates because PCs are present")
        merged = merged.drop(columns=["population_code"])
        covar_cols = [c for c in covar_cols if c != "population_code"]

    merged.to_csv(out_path, sep="\t", index=False)
    return covar_cols


def read_prepared_covariate_columns(merged_covariates_file: str) -> List[str]:
    """Read covariate column names from an already-prepared GWAS covariate table."""
    import pandas as pd

    merged = pd.read_csv(merged_covariates_file, sep="\t", nrows=1)
    return [c for c in merged.columns if c not in {"FID", "IID"}]


def prepare_manifest_covariates_for_gwas(manifest_df):  # noqa: ANN001
    """Prepare one merged covariate table per manifest row for engine reuse."""
    import pandas as pd

    prepared = manifest_df.copy()
    if "merged_covariates_file" not in prepared.columns:
        prepared["merged_covariates_file"] = ""

    for idx, row in prepared.iterrows():
        output_dir = Path(str(row["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        covar_path = output_dir / "merged_covariates.tsv"
        merge_covariates_for_gwas(
            covariates_file=str(row["covariates_file"]),
            pcs_file=str(row["global_pcs_file"]),
            out_path=covar_path,
        )
        prepared.loc[idx, "merged_covariates_file"] = str(covar_path)

    if not isinstance(prepared, pd.DataFrame):
        raise TypeError("prepared manifest must remain a DataFrame")
    return prepared


__all__ = [
    "GwasEngine",
    "GwasResult",
    "GwasTask",
    "merge_covariates_for_gwas",
    "prepare_manifest_covariates_for_gwas",
    "read_prepared_covariate_columns",
]
