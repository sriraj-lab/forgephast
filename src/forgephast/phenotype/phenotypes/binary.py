"""Binary phenotype generation from continuous cohort liability."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable-enough sigmoid for phenotype probabilities."""
    clipped = np.clip(np.asarray(values, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(probability: float) -> float:
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return float(np.log(p) - np.log1p(-p))


def _standardize_liability(liability: np.ndarray) -> np.ndarray:
    values = np.asarray(liability, dtype=float)
    sd = float(np.std(values))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / sd


def _calibrate_logistic_intercept(
    standardized_liability: np.ndarray,
    prevalence: float,
    liability_logit_scale: float,
) -> float:
    """Solve alpha so mean(sigmoid(alpha + gamma * z_liability)) matches prevalence."""
    z = np.asarray(standardized_liability, dtype=float)
    if z.size == 0:
        raise ValueError("standardized_liability cannot be empty")
    target = float(prevalence)
    gamma = float(liability_logit_scale)
    if not 0.0 < target < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("liability_logit_scale must be finite and >= 0")
    if gamma == 0.0:
        return _logit(target)

    center = _logit(target)
    max_abs_z = float(np.max(np.abs(z))) if z.size else 0.0
    lo = min(center - (abs(gamma) * max_abs_z) - 10.0, -80.0)
    hi = max(center + (abs(gamma) * max_abs_z) + 10.0, 80.0)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        mean_probability = float(np.mean(_sigmoid(mid + gamma * z)))
        if mean_probability < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def _binary_phenotype_from_liability(
    *,
    liability: np.ndarray,
    sample_populations: List[str],
    prevalence: float,
    liability_logit_scale: float,
    prevalence_by_population: Dict[str, float],
    alpha_by_population: Dict[str, float] | None = None,
    admix_alpha_weights_by_row: Dict[int, Dict[str, float]] | None = None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Draw binary labels from calibrated logistic probabilities over standardized liability."""
    values = np.asarray(liability, dtype=float)
    z = _standardize_liability(values)
    populations = np.asarray(sample_populations, dtype=object)
    if populations.shape[0] != values.shape[0]:
        raise ValueError("sample_populations length must match liability length")

    labels = np.zeros(values.shape[0], dtype=np.int8)
    probabilities = np.zeros(values.shape[0], dtype=float)
    alpha_values = np.full(values.shape[0], np.nan, dtype=float)
    provided_alpha_by_population = dict(alpha_by_population or {})
    resolved_alpha_by_population: Dict[str, float] = {}
    prevalence_target_by_population: Dict[str, float] = {}
    source_alpha_by_population: Dict[str, float] = {}
    admix_mode = "none"
    alpha_mode = "shared_cohort"

    pop_labels = populations.astype(str)
    observed_groups = {
        str(pop): np.where(pop_labels == str(pop))[0]
        for pop in sorted({str(pop) for pop in sample_populations})
    }
    admix_rows = np.where(np.char.upper(pop_labels) == "ADMIX")[0]

    if prevalence_by_population and provided_alpha_by_population:
        raise ValueError(
            "prevalence_by_population and alpha_by_population are alternative source-alpha inputs"
        )
    if provided_alpha_by_population:
        for pop, alpha in provided_alpha_by_population.items():
            if str(pop).upper() == "ADMIX":
                continue
            source_alpha_by_population[str(pop)] = float(alpha)
            resolved_alpha_by_population[str(pop)] = float(alpha)

    if not prevalence_by_population and not provided_alpha_by_population:
        row_idx = np.arange(values.shape[0], dtype=np.int64)
        alpha = _calibrate_logistic_intercept(
            standardized_liability=z[row_idx],
            prevalence=float(prevalence),
            liability_logit_scale=liability_logit_scale,
        )
        alpha_values[row_idx] = float(alpha)
        resolved_alpha_by_population["__cohort__"] = float(alpha)
        prevalence_target_by_population["__cohort__"] = float(prevalence)
        if admix_rows.size:
            admix_mode = "shared_cohort_alpha"
    else:
        alpha_mode = "population_specific"
        for group, row_idx in observed_groups.items():
            if row_idx.size == 0 or str(group).upper() == "ADMIX":
                continue
            if provided_alpha_by_population:
                if group not in provided_alpha_by_population:
                    raise ValueError(
                        "Missing phenotype.binary.alpha_by_population value "
                        f"for observed population={group}"
                    )
                alpha = float(provided_alpha_by_population[group])
            else:
                target = float(prevalence_by_population.get(group, prevalence))
                alpha = _calibrate_logistic_intercept(
                    standardized_liability=z[row_idx],
                    prevalence=target,
                    liability_logit_scale=liability_logit_scale,
                )
                prevalence_target_by_population[group] = float(target)
            alpha_values[row_idx] = float(alpha)
            resolved_alpha_by_population[group] = float(alpha)
            source_alpha_by_population[group] = float(alpha)

        if admix_rows.size:
            explicit_admix_alpha = provided_alpha_by_population.get("ADMIX")
            if explicit_admix_alpha is not None:
                alpha_values[admix_rows] = float(explicit_admix_alpha)
                resolved_alpha_by_population["ADMIX"] = float(
                    explicit_admix_alpha
                )
                admix_mode = "explicit_admix_alpha"
            elif admix_alpha_weights_by_row:
                missing_rows: list[int] = []
                for row in admix_rows.astype(int).tolist():
                    weights = {
                        str(pop): float(weight)
                        for pop, weight in admix_alpha_weights_by_row.get(
                            row, {}
                        ).items()
                        if float(weight) > 0.0
                    }
                    if not weights:
                        missing_rows.append(row)
                        continue
                    missing_alpha = [
                        pop
                        for pop in weights
                        if pop not in source_alpha_by_population
                    ]
                    if missing_alpha:
                        raise ValueError(
                            "ADMIX alpha weights reference source populations "
                            "without solved/provided alphas: "
                            f"{missing_alpha}"
                        )
                    total_weight = float(sum(weights.values()))
                    if not np.isfinite(total_weight) or total_weight <= 0.0:
                        missing_rows.append(row)
                        continue
                    alpha_values[row] = float(
                        sum(
                            (weight / total_weight)
                            * source_alpha_by_population[pop]
                            for pop, weight in weights.items()
                        )
                    )
                if missing_rows:
                    raise ValueError(
                        "ADMIX samples with population-specific binary alpha require "
                        "architecture weights for every ADMIX row; "
                        f"missing_rows={missing_rows[:5]}"
                    )
                admix_mode = "causal_architecture_weighted_source_alpha"
            else:
                raise ValueError(
                    "ADMIX samples with population-specific binary alpha require "
                    "local-ancestry architecture weights or an explicit "
                    "phenotype.binary.alpha_by_population.ADMIX"
                )

    if np.any(~np.isfinite(alpha_values)):
        missing = np.where(~np.isfinite(alpha_values))[0].astype(int).tolist()
        raise ValueError(
            f"Binary alpha could not be resolved for sample rows: {missing[:5]}"
        )

    probabilities[:] = _sigmoid(
        alpha_values + float(liability_logit_scale) * z
    )
    labels[:] = rng.binomial(
        n=1,
        p=probabilities,
        size=probabilities.shape[0],
    ).astype(np.int8)
    n_cases = int(labels.sum())
    n_total = int(labels.shape[0])
    if admix_rows.size:
        admix_labels = labels[admix_rows]
        admix_probabilities = probabilities[admix_rows]
        admix_alphas = alpha_values[admix_rows]
        admix_realized = (
            float(np.mean(admix_labels))
            if admix_labels.size
            else float("nan")
        )
        admix_mean_probability = (
            float(np.mean(admix_probabilities))
            if admix_probabilities.size
            else float("nan")
        )
        admix_alpha_mean = (
            float(np.mean(admix_alphas))
            if admix_alphas.size
            else float("nan")
        )
        admix_alpha_min = (
            float(np.min(admix_alphas))
            if admix_alphas.size
            else float("nan")
        )
        admix_alpha_max = (
            float(np.max(admix_alphas))
            if admix_alphas.size
            else float("nan")
        )
    else:
        admix_realized = float("nan")
        admix_mean_probability = float("nan")
        admix_alpha_mean = float("nan")
        admix_alpha_min = float("nan")
        admix_alpha_max = float("nan")
    summary: Dict[str, Any] = {
        "binary_prevalence_target": float(prevalence),
        "binary_liability_logit_scale": float(liability_logit_scale),
        "binary_alpha": float(
            resolved_alpha_by_population.get("__cohort__", float("nan"))
        ),
        "binary_alpha_by_population": json.dumps(
            resolved_alpha_by_population,
            sort_keys=True,
        ),
        "binary_prevalence_target_by_population": json.dumps(
            prevalence_target_by_population,
            sort_keys=True,
        ),
        "binary_case_count": n_cases,
        "binary_control_count": n_total - n_cases,
        "binary_realized_prevalence": (
            float(n_cases / n_total) if n_total else float("nan")
        ),
        "binary_mean_predicted_probability": (
            float(np.mean(probabilities)) if n_total else float("nan")
        ),
        "binary_alpha_mode": alpha_mode,
        "binary_admix_alpha_mode": admix_mode,
        "binary_admix_sample_count": int(admix_rows.size),
        "binary_admix_realized_prevalence": admix_realized,
        "binary_admix_mean_predicted_probability": admix_mean_probability,
        "binary_admix_alpha_mean": admix_alpha_mean,
        "binary_admix_alpha_min": admix_alpha_min,
        "binary_admix_alpha_max": admix_alpha_max,
    }
    return labels, probabilities, alpha_values, summary


def _admix_alpha_weights_from_local_ancestry(
    *,
    admix_row_indices: List[int],
    admix_local_ancestry_codes: np.ndarray | None,
    ancestry_order: List[str],
    effects_detail: pd.DataFrame,
    variant_ids: List[str],
    liability: np.ndarray,
    liability_logit_scale: float,
) -> Dict[int, Dict[str, float]]:
    """Compute non-genotype ADMIX alpha weights from local ancestry at causal loci."""
    if not admix_row_indices or admix_local_ancestry_codes is None:
        return {}
    codes = np.asarray(admix_local_ancestry_codes, dtype=np.int64)
    if codes.ndim != 3 or codes.shape[2] != 2:
        raise ValueError(
            "admix_local_ancestry_codes must have shape "
            "(n_admix, n_variants, 2)"
        )
    if codes.shape[0] != len(admix_row_indices):
        raise ValueError(
            "admix_row_indices length must match "
            "admix_local_ancestry_codes sample axis"
        )
    if codes.shape[1] != len(variant_ids):
        raise ValueError(
            "admix_local_ancestry_codes variant axis must match "
            "selected causal variants"
        )

    beta_pivot = effects_detail.pivot(
        index="variant_id",
        columns="population",
        values="beta",
    )
    missing_variants = [
        variant_id
        for variant_id in variant_ids
        if variant_id not in beta_pivot.index
    ]
    if missing_variants:
        raise ValueError(
            f"Missing effects_detail rows for causal variants: {missing_variants[:5]}"
        )
    missing_pops = [
        pop for pop in ancestry_order if pop not in beta_pivot.columns
    ]
    if missing_pops:
        raise ValueError(
            "Missing effects_detail beta columns for ancestry populations: "
            f"{missing_pops}"
        )

    liability_sd = float(np.std(np.asarray(liability, dtype=float)))
    scale = (
        float(liability_logit_scale) / liability_sd
        if np.isfinite(liability_sd) and liability_sd > 0.0
        else 1.0
    )
    beta_log_odds = (
        beta_pivot.loc[variant_ids, ancestry_order].to_numpy(dtype=float)
        * scale
    )
    beta_weight = np.square(beta_log_odds)
    n_anc = len(ancestry_order)
    if np.min(codes) < 0:
        raise ValueError(
            "admix_local_ancestry_codes contains negative ancestry codes"
        )
    if np.max(codes) >= n_anc:
        raise ValueError(
            "admix_local_ancestry_codes contains ancestry code >= n_anc "
            f"(max={int(np.max(codes))}, n_anc={n_anc})"
        )

    weights_by_row: Dict[int, Dict[str, float]] = {}
    for local_idx, global_row in enumerate(admix_row_indices):
        totals = np.zeros(n_anc, dtype=float)
        count_totals = np.zeros(n_anc, dtype=float)
        for hap_idx in (0, 1):
            hap_codes = codes[local_idx, :, hap_idx]
            for anc_idx in range(n_anc):
                mask = hap_codes == anc_idx
                count_totals[anc_idx] += float(np.sum(mask))
                if np.any(mask):
                    totals[anc_idx] += float(
                        np.sum(beta_weight[mask, anc_idx])
                    )
        if float(np.sum(totals)) <= 0.0:
            totals = count_totals
        total = float(np.sum(totals))
        if total <= 0.0:
            totals = np.ones(n_anc, dtype=float)
            total = float(n_anc)
        weights_by_row[int(global_row)] = {
            str(pop): float(totals[idx] / total)
            for idx, pop in enumerate(ancestry_order)
            if totals[idx] > 0.0
        }
    return weights_by_row
