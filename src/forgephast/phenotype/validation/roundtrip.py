"""Round-trip GWAS validation against simulation truth tables."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids, read_sample_ids

from ..config import CohortConfig, RunConfig
from ..local_ancestry.tracts import load_tracts_tsv_gz

logger = logging.getLogger(__name__)


@dataclass
class RoundTripComparison:
    """Per-engine round-trip comparison payload for reports and plots."""

    run_name: str
    replicate: int
    architecture: str
    cohort: str
    engine: str
    assoc_path: Path
    assoc_df: pd.DataFrame
    truth_df: pd.DataFrame
    merged_df: pd.DataFrame
    top_df: pd.DataFrame
    metrics: dict[str, Any]


def _choose_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        needle = candidate.lower()
        found = lookup.get(needle)
        if found is not None:
            return found
        for lowered, original in lookup.items():
            if lowered.startswith(f"{needle}.") or lowered.startswith(f"{needle}_"):
                return original
    return None


def _read_assoc_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", compression="infer")
    chrom_col = _choose_column(list(df.columns), ["#CHROM", "CHROM"])
    pos_col = _choose_column(list(df.columns), ["POS", "BP", "GENPOS"])
    chrom_values = pd.Series(pd.NA, index=df.index, dtype="object")
    pos_values = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if chrom_col is not None and pos_col is not None:
        chrom_values = (
            df[chrom_col].astype(str).str.replace("^chr", "", regex=True)
        )
        pos_values = pd.to_numeric(df[pos_col], errors="coerce").astype("Int64")
        df["variant_id"] = (
            "chr"
            + chrom_values.astype(str)
            + ":"
            + pos_values.astype(str)
        )
    else:
        id_col = _choose_column(list(df.columns), ["ID", "SNP", "VARIANT"])
        if id_col is None:
            raise ValueError(f"Unable to infer variant identifier columns from {path}")
        df["variant_id"] = df[id_col].astype(str)
        parsed = df["variant_id"].str.extract(r"^chr(?P<chrom>[^:]+):(?P<pos>\d+)$")
        if not parsed.empty:
            chrom_values = parsed["chrom"].astype("object")
            pos_values = pd.to_numeric(parsed["pos"], errors="coerce").astype("Int64")

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
    df["chrom"] = chrom_values
    df["pos"] = pos_values
    return df[["variant_id", "chrom", "pos", "beta_hat", "p_value", "gwas_effect_allele"]].copy()


def _resolve_assoc_output_path(engine: str, primary_output: Path) -> Path:
    if engine.lower() != "regenie":
        return primary_output
    if primary_output.name.endswith(".regenie") or primary_output.name.endswith(".regenie.gz"):
        return primary_output

    out_dir = primary_output.parent
    candidates = [
        path
        for path in sorted(out_dir.glob("step2*.regenie*"))
        if path.exists() and path.stat().st_size > 0
    ]
    assoc_candidates = [
        path
        for path in candidates
        if path.name.endswith(".regenie") or path.name.endswith(".regenie.gz")
    ]
    if assoc_candidates:
        return assoc_candidates[0]
    return primary_output


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


def _membership_subset(
    *,
    membership_df: pd.DataFrame,
    replicate: int,
    architecture: str,
    cohort: str,
    variant_id_column: str,
) -> pd.DataFrame:
    rows = membership_df[
        (membership_df["replicate"] == int(replicate))
        & (membership_df["architecture"] == str(architecture))
        & (membership_df["cohort"] == str(cohort))
    ].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["variant_id", "sampling_weight", "annotation_hit", "annotation_labels"]
        )
    keep_cols = [variant_id_column]
    for optional in ("sampling_weight", "annotation_hit", "annotation_labels"):
        if optional in rows.columns:
            keep_cols.append(optional)
    out = rows[keep_cols].rename(columns={variant_id_column: "variant_id"}).copy()
    if "sampling_weight" not in out.columns:
        out["sampling_weight"] = 1.0
    else:
        out["sampling_weight"] = pd.to_numeric(out["sampling_weight"], errors="coerce").fillna(1.0)
    if "annotation_hit" not in out.columns:
        out["annotation_hit"] = False
    else:
        out["annotation_hit"] = out["annotation_hit"].fillna(False).astype(bool)
    if "annotation_labels" not in out.columns:
        out["annotation_labels"] = ""
    else:
        out["annotation_labels"] = out["annotation_labels"].fillna("").astype(str)
    out = out.drop_duplicates(subset=["variant_id"], keep="first").reset_index(drop=True)
    return out


def merge_assoc_truth_effects(
    assoc_df: pd.DataFrame,
    truth_weighted: pd.DataFrame,
) -> pd.DataFrame:
    """Merge association and truth tables with allele-aware beta harmonization."""
    merged = assoc_df.merge(truth_weighted, on="variant_id", how="inner")
    if merged.empty:
        merged["beta_hat_harmonized"] = pd.Series(dtype=float)
        return merged
    merged["beta_hat_harmonized"] = merged["beta_hat"].astype(float)
    allele_present = (
        merged["gwas_effect_allele"].astype(str) != ""
    ) & (merged["dosage_effect_allele"].astype(str) != "")
    flip_mask = allele_present & (
        merged["gwas_effect_allele"].astype(str).str.upper()
        != merged["dosage_effect_allele"].astype(str).str.upper()
    )
    merged.loc[flip_mask, "beta_hat_harmonized"] *= -1.0
    return merged


def _linear_metrics(merged: pd.DataFrame) -> Tuple[float, float, float, float]:
    if merged.shape[0] < 2:
        return np.nan, np.nan, np.nan, np.nan
    x = merged["beta_true"].to_numpy(dtype=float)
    y = merged["beta_hat_harmonized"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan, np.nan, np.nan, np.nan
    x = x[finite]
    y = y[finite]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan, np.nan, np.nan
    try:
        slope, intercept = np.polyfit(x, y, deg=1)
        corr = float(np.corrcoef(x, y)[0, 1])
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        logger.warning(
            "Round-trip linear metrics are numerically degenerate; returning NaN metrics"
        )
        return np.nan, np.nan, np.nan, np.nan
    nz = (np.abs(x) > 0) & (np.abs(y) > 0)
    sign = float(np.mean(np.sign(x[nz]) == np.sign(y[nz]))) if np.any(nz) else np.nan
    return float(slope), float(intercept), corr, sign


def _component_summary_lookup(truth_dir: Path) -> Dict[tuple[int, str, str], pd.Series]:
    """Load component_summary rows keyed by replicate x architecture x cohort."""
    path = truth_dir / "component_summary.tsv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t")
    required = {"replicate", "architecture", "cohort"}
    if not required.issubset(frame.columns):
        return {}
    out: Dict[tuple[int, str, str], pd.Series] = {}
    for _, row in frame.iterrows():
        key = (int(row["replicate"]), str(row["architecture"]), str(row["cohort"]))
        out[key] = row
    return out


def _float_from_row(row: pd.Series | None, column: str, fallback: float) -> float:
    if row is None or column not in row:
        return float(fallback)
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return float(fallback)
    return float(value)


def _binary_truth_scale_metadata(
    *,
    component_row: pd.Series | None,
    run_cfg: RunConfig,
) -> Dict[str, float | str]:
    """Resolve liability-beta -> expected log-odds scale for binary validation."""
    gamma = _float_from_row(
        component_row,
        "binary_liability_logit_scale",
        float(run_cfg.phenotype.binary.liability_logit_scale),
    )
    liability_var = _float_from_row(component_row, "realized_var_liability", 1.0)
    liability_sd = float(np.sqrt(liability_var)) if liability_var > 0.0 else float("nan")
    scale = float(gamma / liability_sd) if np.isfinite(liability_sd) and liability_sd > 0.0 else float("nan")
    return {
        "binary_validation_scale": scale,
        "binary_liability_logit_scale": float(gamma),
        "binary_liability_sd": liability_sd,
        "binary_validation_target": "log_odds_approx",
    }


def _roundtrip_details_path(report_path: Path) -> Path:
    """Resolve the per-variant detail path next to the summary report."""
    if report_path.name.endswith(".tsv"):
        return report_path.with_name(f"{report_path.stem}_details.tsv")
    return report_path.with_name(f"{report_path.name}_details.tsv")


def _write_roundtrip_details(
    *,
    comparisons: list[RoundTripComparison],
    details_path: Path,
) -> None:
    """Write per-variant observed-vs-expected beta details for each comparison."""
    frames: list[pd.DataFrame] = []
    for comparison in comparisons:
        detail = comparison.merged_df.copy()
        if detail.empty:
            continue
        metrics = dict(comparison.metrics)
        detail.insert(0, "run_name", comparison.run_name)
        detail.insert(1, "replicate", int(comparison.replicate))
        detail.insert(2, "architecture", comparison.architecture)
        detail.insert(3, "cohort", comparison.cohort)
        detail.insert(4, "engine", comparison.engine)
        detail.insert(5, "trait_type", str(metrics.get("trait_type", "")))
        detail.insert(6, "validation_status", str(metrics.get("validation_status", "")))
        detail["assoc_path"] = str(comparison.assoc_path)
        if str(metrics.get("trait_type", "")) == "binary":
            if "expected_log_odds_beta" not in detail.columns:
                detail["expected_log_odds_beta"] = detail["beta_true"]
            if "binary_validation_scale" not in detail.columns:
                detail["binary_validation_scale"] = metrics.get("binary_validation_scale", np.nan)
        else:
            if "expected_liability_beta" not in detail.columns:
                detail["expected_liability_beta"] = detail["beta_true"]
        frames.append(detail)

    details_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            details_path,
            sep="\t",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "run_name",
                "replicate",
                "architecture",
                "cohort",
                "engine",
                "trait_type",
                "validation_status",
                "variant_id",
                "beta_hat",
                "beta_hat_harmonized",
                "beta_true",
                "expected_log_odds_beta",
                "expected_liability_beta",
            ]
        ).to_csv(details_path, sep="\t", index=False)
    logger.info("[validate] wrote round-trip details: %s", details_path)


def _resolve_truth_dir(run_dir: Path) -> Path:
    truth_dir = run_dir / "truth"
    if truth_dir.exists():
        return truth_dir
    if run_dir.name.endswith("__ids_only"):
        base_truth_dir = run_dir.parent / run_dir.name.removesuffix("__ids_only") / "truth"
        if base_truth_dir.exists():
            return base_truth_dir
    return truth_dir


def collect_roundtrip_details(
    *,
    run_cfg: RunConfig,
    run_dir: Path,
    gwas_manifest_path: Path,
) -> tuple[pd.DataFrame, list[RoundTripComparison], list[str]]:
    """Collect per-comparison round-trip detail tables for reports and plots."""
    gwas_manifest = pd.read_csv(gwas_manifest_path, sep="\t")
    gwas_manifest = gwas_manifest[gwas_manifest["status"] == "success"].copy()
    if gwas_manifest.empty:
        raise ValueError("No successful GWAS rows found for round-trip validation")

    truth_dir = _resolve_truth_dir(run_dir)
    truth_path = truth_dir / "effect_sizes.tsv"
    membership_path = truth_dir / "causal_membership.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth effect file: {truth_path}")
    if not membership_path.exists():
        raise FileNotFoundError(f"Missing causal membership file: {membership_path}")

    truth_df = pd.read_csv(truth_path, sep="\t")
    membership_df = pd.read_csv(membership_path, sep="\t")
    component_lookup = _component_summary_lookup(truth_dir)

    cohort_cfg = {cohort.name: cohort for cohort in run_cfg.cohorts}
    default_pop = run_cfg.effect_model.population_order[0]
    pop_weights_cache: Dict[str, Dict[str, float]] = {}

    rows = []
    failures: list[str] = []
    comparisons: list[RoundTripComparison] = []
    for _, row in gwas_manifest.iterrows():
        replicate = int(row["replicate"])
        architecture = str(row["architecture"])
        cohort_name = str(row["cohort"])
        engine = str(row["engine"])
        trait_type = str(row.get("trait_type", "quantitative"))
        assoc_path = _resolve_assoc_output_path(engine, Path(str(row["primary_output"])))
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
        validation_status = "validated"
        scale_metadata: Dict[str, Any] = {}
        if trait_type == "binary":
            validation_status = "validated_binary_log_odds"
            scale_metadata = _binary_truth_scale_metadata(
                component_row=component_lookup.get((replicate, architecture, cohort_name)),
                run_cfg=run_cfg,
            )
            truth_weighted = truth_weighted.copy()
            truth_weighted["beta_true_liability"] = truth_weighted["beta_true"]
            truth_weighted["expected_log_odds_beta"] = (
                truth_weighted["beta_true"].astype(float)
                * float(scale_metadata["binary_validation_scale"])
            )
            truth_weighted["beta_true"] = truth_weighted["expected_log_odds_beta"]
            truth_weighted["binary_validation_scale"] = float(
                scale_metadata["binary_validation_scale"]
            )
        membership_subset = _membership_subset(
            membership_df=membership_df,
            replicate=replicate,
            architecture=architecture,
            cohort=cohort_name,
            variant_id_column=run_cfg.causal_selection.variant_id_column,
        )
        assoc = assoc.merge(membership_subset, on="variant_id", how="left")
        assoc["sampling_weight"] = pd.to_numeric(
            assoc.get("sampling_weight", 1.0), errors="coerce"
        ).fillna(1.0)
        annotation_hit = assoc.get("annotation_hit")
        if annotation_hit is None:
            assoc["annotation_hit"] = False
        else:
            assoc["annotation_hit"] = (
                annotation_hit.astype("boolean").fillna(False).astype(bool)
            )
        annotation_labels = assoc.get("annotation_labels")
        if annotation_labels is None:
            assoc["annotation_labels"] = ""
        else:
            assoc["annotation_labels"] = annotation_labels.fillna("").astype(str)
        causal_set = set(membership_subset["variant_id"].astype(str).tolist())
        assoc["is_causal"] = assoc["variant_id"].astype(str).isin(causal_set)
        merged = merge_assoc_truth_effects(assoc, truth_weighted)
        if merged.empty:
            metrics = {
                "run_name": str(row["run_name"]),
                "replicate": replicate,
                "architecture": architecture,
                "cohort": cohort_name,
                "engine": engine,
                "trait_type": trait_type,
                "validation_status": f"{validation_status}_no_overlap",
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
            metrics.update(scale_metadata)
            rows.append(metrics)
            comparisons.append(
                RoundTripComparison(
                    run_name=str(row["run_name"]),
                    replicate=replicate,
                    architecture=architecture,
                    cohort=cohort_name,
                    engine=engine,
                    assoc_path=assoc_path,
                    assoc_df=assoc,
                    truth_df=truth_weighted,
                    merged_df=merged,
                    top_df=assoc.sort_values("p_value", ascending=True).head(
                        int(run_cfg.roundtrip_validation.top_k)
                    ).copy(),
                    metrics=metrics,
                )
            )
            continue

        slope, intercept, corr, sign = _linear_metrics(merged)
        top_k = int(run_cfg.roundtrip_validation.top_k)
        top = assoc.sort_values("p_value", ascending=True).head(top_k).copy()
        top_causal = int(top["is_causal"].sum())
        top_frac = float(top_causal / max(1, top.shape[0]))

        metrics = {
            "run_name": str(row["run_name"]),
            "replicate": replicate,
            "architecture": architecture,
            "cohort": cohort_name,
            "engine": engine,
            "trait_type": trait_type,
            "validation_status": validation_status,
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
        metrics.update(scale_metadata)
        rows.append(metrics)
        comparisons.append(
            RoundTripComparison(
                run_name=str(row["run_name"]),
                replicate=replicate,
                architecture=architecture,
                cohort=cohort_name,
                engine=engine,
                assoc_path=assoc_path,
                assoc_df=assoc,
                truth_df=truth_weighted,
                merged_df=merged,
                top_df=top,
                metrics=metrics,
            )
        )

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
    return report_df, comparisons, failures


def build_roundtrip_report(
    *,
    run_cfg: RunConfig,
    run_dir: Path,
    gwas_manifest_path: Path,
) -> Path:
    """Build GWAS-vs-truth validation report and enforce configured gates."""
    report_df, comparisons, failures = collect_roundtrip_details(
        run_cfg=run_cfg,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )
    report_path = (
        Path(run_cfg.roundtrip_validation.report_path)
        if run_cfg.roundtrip_validation.report_path
        else run_dir / "roundtrip_validation.tsv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, sep="\t", index=False)
    logger.info("[validate] wrote round-trip report: %s", report_path)
    _write_roundtrip_details(
        comparisons=comparisons,
        details_path=_roundtrip_details_path(report_path),
    )

    if failures:
        raise RuntimeError(
            "Round-trip validation failed:\n" + "\n".join(failures)
        )
    return report_path
