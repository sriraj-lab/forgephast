"""Round-trip GWAS validation against simulation truth tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids, read_sample_ids

from ..config import CohortConfig, RunConfig
from ..local_ancestry.tracts import load_tracts_tsv_gz

logger = logging.getLogger(__name__)


def _choose_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _read_assoc_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", compression="infer")
    id_col = _choose_column(list(df.columns), ["ID", "SNP", "VARIANT"])
    if id_col is None:
        chrom_col = _choose_column(list(df.columns), ["#CHROM", "CHROM"])
        pos_col = _choose_column(list(df.columns), ["POS", "BP"])
        if chrom_col is None or pos_col is None:
            raise ValueError(f"Unable to infer variant identifier columns from {path}")
        df["variant_id"] = (
            "chr"
            + df[chrom_col].astype(str).str.replace("^chr", "", regex=True)
            + ":"
            + df[pos_col].astype(str)
        )
    else:
        df["variant_id"] = df[id_col].astype(str)

    beta_col = _choose_column(list(df.columns), ["BETA", "Effect"])
    if beta_col is None:
        raise ValueError(f"Unable to infer beta column from {path}")
    df["beta_hat"] = pd.to_numeric(df[beta_col], errors="coerce")

    p_col = _choose_column(list(df.columns), ["P", "PVAL"])
    if p_col is not None:
        df["p_value"] = pd.to_numeric(df[p_col], errors="coerce")
    else:
        log10p_col = _choose_column(list(df.columns), ["LOG10P", "LOG10_P"])
        if log10p_col is None:
            raise ValueError(f"Unable to infer p-value/log10p column from {path}")
        df["p_value"] = 10.0 ** (-pd.to_numeric(df[log10p_col], errors="coerce"))

    allele_col = _choose_column(list(df.columns), ["A1", "ALLELE1", "ALT"])
    df["gwas_effect_allele"] = "" if allele_col is None else df[allele_col].astype(str)
    return df[["variant_id", "beta_hat", "p_value", "gwas_effect_allele"]].copy()


def _admix_population_mass(
    *,
    cohort: CohortConfig,
    admix_sample_ids: list[str],
    run_cfg: RunConfig,
) -> Dict[str, float]:
    if not admix_sample_ids:
        return {}
    if not cohort.local_ancestry_tracts_pattern:
        return {}

    ancestry_bp: Dict[str, float] = {}
    for chrom in run_cfg.chromosomes:
        pattern = str(cohort.local_ancestry_tracts_pattern)
        tracts_path = Path(pattern.format(chrom=int(chrom))) if "{chrom}" in pattern else Path(pattern)
        tracts = load_tracts_tsv_gz(
            tracts_path,
            sample_ids=admix_sample_ids,
            label_map=dict(run_cfg.local_ancestry.label_map),
        )
        for tract_list in tracts.values():
            for tract in tract_list:
                span = max(0, int(tract.right_bp) - int(tract.left_bp))
                ancestry_bp[str(tract.ancestry)] = ancestry_bp.get(str(tract.ancestry), 0.0) + float(span)

    total_bp = float(sum(ancestry_bp.values()))
    if total_bp <= 0.0:
        return {}
    admix_mass = float(len(admix_sample_ids))
    return {pop: admix_mass * (bp / total_bp) for pop, bp in ancestry_bp.items()}


def _population_weights(cohort: CohortConfig, sample_ids: list[str], run_cfg: RunConfig) -> Dict[str, float]:
    if not cohort.population_map_file:
        return {}
    pop_df = pd.read_csv(cohort.population_map_file, sep="\t")
    id_col = "IID" if "IID" in pop_df.columns else "#IID" if "#IID" in pop_df.columns else None
    if id_col is None:
        raise ValueError(
            f"cohort {cohort.name}: population_map_file must contain IID or #IID column"
        )
    pop_col = "population" if "population" in pop_df.columns else None
    if pop_col is None:
        other = [c for c in pop_df.columns if c != id_col]
        if not other:
            raise ValueError(f"cohort {cohort.name}: population map file has no population column")
        pop_col = other[0]
    aligned = align_by_ids(
        pop_df[[id_col, pop_col]].copy(),
        id_column=id_col,
        ids=sample_ids,
        context=f"cohort {cohort.name} population map alignment",
    )
    pop_series = aligned[pop_col].astype(str)
    counts = pop_series.value_counts(dropna=False).astype(float).to_dict()
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {}

    admix_ids = aligned.loc[pop_series.str.upper() == "ADMIX", "IID"].astype(str).tolist()
    if run_cfg.local_ancestry.enabled and admix_ids:
        counts.pop("ADMIX", None)
        counts.pop("admix", None)
        for pop, mass in _admix_population_mass(
            cohort=cohort,
            admix_sample_ids=admix_ids,
            run_cfg=run_cfg,
        ).items():
            counts[str(pop)] = float(counts.get(str(pop), 0.0) + float(mass))

    return {str(pop): float(count / total) for pop, count in counts.items()}


def _weighted_truth_effects(
    *,
    truth_df: pd.DataFrame,
    replicate: int,
    architecture: str,
    cohort: str,
    pop_weights: Dict[str, float],
    default_population: str,
) -> pd.DataFrame:
    rows = truth_df[
        (truth_df["replicate"] == int(replicate))
        & (truth_df["architecture"] == str(architecture))
        & (truth_df["cohort"] == str(cohort))
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["variant_id", "beta_true", "dosage_effect_allele"])

    pivot = rows.pivot_table(
        index="variant_id",
        columns="population",
        values="beta",
        aggfunc="first",
    )
    if not pop_weights:
        pop_weights = {str(default_population): 1.0}

    for pop in pop_weights:
        if pop not in pivot.columns:
            pivot[pop] = 0.0
    weight_vec = np.asarray([float(pop_weights[pop]) for pop in pop_weights], dtype=float)
    beta_mat = pivot[list(pop_weights.keys())].to_numpy(dtype=float)
    beta_true = beta_mat @ weight_vec

    allele_map = (
        rows.groupby("variant_id")["dosage_effect_allele"]
        .agg(lambda s: s.dropna().astype(str).iloc[0] if not s.dropna().empty else "")
        .to_dict()
    )
    out = pd.DataFrame(
        {
            "variant_id": pivot.index.astype(str).tolist(),
            "beta_true": beta_true,
            "dosage_effect_allele": [allele_map.get(str(v), "") for v in pivot.index.tolist()],
        }
    )
    return out


def _linear_metrics(merged: pd.DataFrame) -> Tuple[float, float, float, float]:
    if merged.shape[0] < 2:
        return np.nan, np.nan, np.nan, np.nan
    x = merged["beta_true"].to_numpy(dtype=float)
    y = merged["beta_hat_harmonized"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    corr = float(np.corrcoef(x, y)[0, 1])
    nz = (np.abs(x) > 0) & (np.abs(y) > 0)
    sign = float(np.mean(np.sign(x[nz]) == np.sign(y[nz]))) if np.any(nz) else np.nan
    return float(slope), float(intercept), corr, sign


def build_roundtrip_report(
    *,
    run_cfg: RunConfig,
    run_dir: Path,
    gwas_manifest_path: Path,
) -> Path:
    """Build GWAS-vs-truth validation report and enforce configured gates."""
    gwas_manifest = pd.read_csv(gwas_manifest_path, sep="\t")
    gwas_manifest = gwas_manifest[gwas_manifest["status"] == "success"].copy()
    if gwas_manifest.empty:
        raise ValueError("No successful GWAS rows found for round-trip validation")

    truth_path = run_dir / "truth" / "effect_sizes.tsv"
    membership_path = run_dir / "truth" / "causal_membership.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth effect file: {truth_path}")
    if not membership_path.exists():
        raise FileNotFoundError(f"Missing causal membership file: {membership_path}")

    truth_df = pd.read_csv(truth_path, sep="\t")
    membership_df = pd.read_csv(membership_path, sep="\t")

    cohort_cfg = {cohort.name: cohort for cohort in run_cfg.cohorts}
    default_pop = run_cfg.effect_model.population_order[0]
    pop_weights_cache: Dict[str, Dict[str, float]] = {}

    rows = []
    failures = []
    for _, row in gwas_manifest.iterrows():
        replicate = int(row["replicate"])
        architecture = str(row["architecture"])
        cohort_name = str(row["cohort"])
        engine = str(row["engine"])
        assoc_path = Path(str(row["primary_output"]))
        if not assoc_path.exists():
            raise FileNotFoundError(f"GWAS primary output missing: {assoc_path}")

        cohort = cohort_cfg.get(cohort_name)
        if cohort is None:
            raise ValueError(f"Cohort {cohort_name!r} not found in config")
        if cohort_name not in pop_weights_cache:
            sample_ids = read_sample_ids(cohort.sample_ids_file)
            pop_weights_cache[cohort_name] = _population_weights(cohort, sample_ids, run_cfg)

        assoc = _read_assoc_table(assoc_path)
        truth_weighted = _weighted_truth_effects(
            truth_df=truth_df,
            replicate=replicate,
            architecture=architecture,
            cohort=cohort_name,
            pop_weights=pop_weights_cache[cohort_name],
            default_population=default_pop,
        )
        merged = assoc.merge(truth_weighted, on="variant_id", how="inner")
        if merged.empty:
            rows.append(
                {
                    "run_name": str(row["run_name"]),
                    "replicate": replicate,
                    "architecture": architecture,
                    "cohort": cohort_name,
                    "engine": engine,
                    "n_assoc_variants": int(assoc.shape[0]),
                    "n_truth_variants": int(truth_weighted.shape[0]),
                    "n_overlap": 0,
                    "beta_slope": np.nan,
                    "beta_intercept": np.nan,
                    "beta_corr": np.nan,
                    "sign_concordance": np.nan,
                    "top_k": int(run_cfg.roundtrip_validation.top_k),
                    "top_k_causal_count": 0,
                    "top_k_causal_fraction": 0.0,
                }
            )
            continue

        merged["beta_hat_harmonized"] = merged["beta_hat"].astype(float)
        allele_present = (
            merged["gwas_effect_allele"].astype(str) != ""
        ) & (merged["dosage_effect_allele"].astype(str) != "")
        flip_mask = allele_present & (
            merged["gwas_effect_allele"].astype(str).str.upper()
            != merged["dosage_effect_allele"].astype(str).str.upper()
        )
        merged.loc[flip_mask, "beta_hat_harmonized"] *= -1.0

        slope, intercept, corr, sign = _linear_metrics(merged)
        causal_set = set(
            membership_df[
                (membership_df["replicate"] == replicate)
                & (membership_df["architecture"] == architecture)
                & (membership_df["cohort"] == cohort_name)
            ][run_cfg.causal_selection.variant_id_column]
            .astype(str)
            .tolist()
        )
        top_k = int(run_cfg.roundtrip_validation.top_k)
        top = assoc.sort_values("p_value", ascending=True).head(top_k).copy()
        top["is_causal"] = top["variant_id"].astype(str).isin(causal_set)
        top_causal = int(top["is_causal"].sum())
        top_frac = float(top_causal / max(1, top.shape[0]))

        metrics = {
            "run_name": str(row["run_name"]),
            "replicate": replicate,
            "architecture": architecture,
            "cohort": cohort_name,
            "engine": engine,
            "n_assoc_variants": int(assoc.shape[0]),
            "n_truth_variants": int(truth_weighted.shape[0]),
            "n_overlap": int(merged.shape[0]),
            "beta_slope": slope,
            "beta_intercept": intercept,
            "beta_corr": corr,
            "sign_concordance": sign,
            "top_k": top_k,
            "top_k_causal_count": top_causal,
            "top_k_causal_fraction": top_frac,
        }
        rows.append(metrics)

        if (
            run_cfg.roundtrip_validation.require_positive_slope
            and np.isfinite(metrics["beta_slope"])
            and float(metrics["beta_slope"]) <= 0.0
        ):
            failures.append(
                f"Negative/non-positive slope for engine={engine} rep={replicate} "
                f"arch={architecture} cohort={cohort_name}: slope={metrics['beta_slope']:.6f}"
            )

    report_df = pd.DataFrame(rows)
    report_path = (
        Path(run_cfg.roundtrip_validation.report_path)
        if run_cfg.roundtrip_validation.report_path
        else run_dir / "roundtrip_validation.tsv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, sep="\t", index=False)
    logger.info("[validate] wrote round-trip report: %s", report_path)

    if failures:
        raise RuntimeError(
            "Round-trip validation failed:\n" + "\n".join(failures)
        )
    return report_path
