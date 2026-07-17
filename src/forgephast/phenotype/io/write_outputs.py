"""Output writers for simulation and manifest artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import pandas as pd


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(tmp_path, sep="\t", index=False)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_keep_file(phenotype_df: pd.DataFrame, out_path: Path) -> None:
    """Write normalized FID IID keep file."""
    keep = phenotype_df[["FID", "IID"]].copy()
    _atomic_to_csv(keep, out_path)


def write_simulation_outputs(
    run_name: str,
    output_root: Path,
    records: List[Dict[str, object]],
    run_identity: str = "",
) -> pd.DataFrame:
    """Persist simulation record tables and return run manifest dataframe."""
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_path = run_dir / "SIMULATE_COMPLETE.json"
    completion_path.unlink(missing_ok=True)

    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    membership_frames = []
    effect_frames = []
    component_rows = []
    effect_geometry_rows = []
    effect_geometry_seen: set[tuple[int, str]] = set()

    for rec in records:
        replicate = int(rec["replicate"])
        architecture = str(rec["architecture"])
        cohort = str(rec["cohort"])

        out_dir = run_dir / f"replicate_{replicate}" / f"architecture_{architecture}" / f"cohort_{cohort}"
        out_dir.mkdir(parents=True, exist_ok=True)

        phenotype_path = out_dir / "phenotype.tsv"
        covariate_path = out_dir / "covariates.tsv"
        pcs_path = out_dir / "global_pcs.tsv"
        keep_path = out_dir / "keep_ids.tsv"
        coeff_path = out_dir / "covariate_coefficients.tsv"

        phenotype_df = rec["phenotype_df"]
        covariate_df = rec["covariate_df"]
        pcs_df = rec["pcs_df"]
        coeff_df = rec["coefficients_df"]

        _atomic_to_csv(phenotype_df, phenotype_path)
        _atomic_to_csv(covariate_df, covariate_path)
        _atomic_to_csv(pcs_df, pcs_path)
        _atomic_to_csv(coeff_df, coeff_path)
        _write_keep_file(phenotype_df, keep_path)

        membership_frames.append(rec["membership_df"])
        effect_frames.append(rec["effects_df"])

        targets = rec["component_targets"]
        realized = rec.get("component_realized", {})
        effect_geometry = rec.get("effect_geometry", {})
        realized_fixed_variance_by_population = (
            rec.get("realized_fixed_variance_by_population", {}) or {}
        )
        tightness_summary = rec.get("tightness_summary", {}) or {}
        trait_type = str(rec.get("trait_type", "quantitative"))
        binary_summary = rec.get("binary_summary", {}) or {}

        realized_var_genetic_fixed_by_population_json = json.dumps(
            {
                str(k): float(v)
                for k, v in realized_fixed_variance_by_population.items()
            },
            sort_keys=True,
        )
        shift_tightness_tier = str(tightness_summary.get("tier", "extreme"))
        shift_tightness_fold_change_cap = float(
            tightness_summary.get("fold_change_cap", float("inf"))
        )
        shift_tightness_fold_change_max_kept = float(
            tightness_summary.get("fold_change_max_kept", float("nan"))
        )
        shift_tightness_fold_change_quantiles_json = json.dumps(
            tightness_summary.get("per_pair_quantiles", {}) or {},
            sort_keys=True,
        )
        shift_tightness_pre_filter_count = int(
            tightness_summary.get("pre_filter_count", 0)
        )
        shift_tightness_post_filter_count = int(
            tightness_summary.get("post_filter_count", 0)
        )

        component_rows.append(
            {
                "run_name": run_name,
                "replicate": replicate,
                "architecture": architecture,
                "cohort": cohort,
                "trait_type": trait_type,
                "genetic_fixed_beta_scale_factor": rec.get("genetic_fixed_beta_scale_factor", 1.0),
                "target_h2_fixed": targets.get("target_h2_fixed", targets.get("genetic_fixed", 0.0)),
                "target_var_genetic_fixed": targets.get("genetic_fixed", 0.0),
                "target_var_genetic_bg": targets.get("genetic_bg", 0.0),
                "target_var_genetic_total": targets.get("genetic_total", 0.0),
                "target_var_covariate": targets.get("covariate", 0.0),
                "target_var_residual": targets.get("residual", 0.0),
                "realized_var_genetic_fixed": realized.get("genetic_fixed", 0.0),
                "realized_var_genetic_bg": realized.get("genetic_bg", 0.0),
                "realized_var_genetic_total": realized.get("genetic_total", 0.0),
                "realized_var_covariate": realized.get("covariate", 0.0),
                "realized_var_residual": realized.get("residual", 0.0),
                "realized_var_phenotype": realized.get("phenotype", 0.0),
                "realized_var_liability": realized.get("liability", realized.get("phenotype", 0.0)),
                "realized_var_genetic_fixed_by_population": (
                    realized_var_genetic_fixed_by_population_json
                ),
                "binary_prevalence_target": binary_summary.get("binary_prevalence_target", float("nan")),
                "binary_liability_logit_scale": binary_summary.get("binary_liability_logit_scale", float("nan")),
                "binary_alpha": binary_summary.get("binary_alpha", float("nan")),
                "binary_alpha_by_population": binary_summary.get("binary_alpha_by_population", "{}"),
                "binary_prevalence_target_by_population": binary_summary.get(
                    "binary_prevalence_target_by_population",
                    "{}",
                ),
                "binary_case_count": binary_summary.get("binary_case_count", 0),
                "binary_control_count": binary_summary.get("binary_control_count", 0),
                "binary_realized_prevalence": binary_summary.get("binary_realized_prevalence", float("nan")),
                "binary_mean_predicted_probability": binary_summary.get(
                    "binary_mean_predicted_probability",
                    float("nan"),
                ),
                "binary_alpha_mode": binary_summary.get("binary_alpha_mode", ""),
                "binary_admix_alpha_mode": binary_summary.get("binary_admix_alpha_mode", ""),
                "binary_admix_sample_count": binary_summary.get("binary_admix_sample_count", 0),
                "binary_admix_realized_prevalence": binary_summary.get(
                    "binary_admix_realized_prevalence",
                    float("nan"),
                ),
                "binary_admix_mean_predicted_probability": binary_summary.get(
                    "binary_admix_mean_predicted_probability",
                    float("nan"),
                ),
                "binary_admix_alpha_mean": binary_summary.get("binary_admix_alpha_mean", float("nan")),
                "binary_admix_alpha_min": binary_summary.get("binary_admix_alpha_min", float("nan")),
                "binary_admix_alpha_max": binary_summary.get("binary_admix_alpha_max", float("nan")),
                "shift_tightness_tier": shift_tightness_tier,
                "shift_tightness_fold_change_cap": shift_tightness_fold_change_cap,
                "shift_tightness_fold_change_max_kept": shift_tightness_fold_change_max_kept,
                "shift_tightness_fold_change_quantiles_json": shift_tightness_fold_change_quantiles_json,
                "shift_tightness_pre_filter_count": shift_tightness_pre_filter_count,
                "shift_tightness_post_filter_count": shift_tightness_post_filter_count,
            }
        )

        manifest_rows.append(
            {
                "run_name": run_name,
                "replicate": replicate,
                "architecture": architecture,
                "cohort": cohort,
                "genetic_fixed_beta_scale_factor": rec.get("genetic_fixed_beta_scale_factor", 1.0),
                "trait_type": trait_type,
                "plugin": rec["plugin"],
                "pgen_prefix": rec["pgen_prefix"],
                "sample_ids_file": rec["sample_ids_file"],
                "keep_file": str(keep_path),
                "phenotype_file": str(phenotype_path),
                "covariates_file": str(covariate_path),
                "global_pcs_file": str(pcs_path),
                "covariate_coefficients_file": str(coeff_path),
                "output_dir": str(out_dir),
                "target_h2_fixed": targets.get("target_h2_fixed", targets.get("genetic_fixed", 0.0)),
                "target_var_genetic_fixed": targets.get("genetic_fixed", 0.0),
                "target_var_genetic_bg": targets.get("genetic_bg", 0.0),
                "target_var_genetic_total": targets.get("genetic_total", 0.0),
                "target_var_covariate": targets.get("covariate", 0.0),
                "target_var_residual": targets.get("residual", 0.0),
                "realized_var_genetic_fixed_by_population": (
                    realized_var_genetic_fixed_by_population_json
                ),
                "binary_prevalence_target": binary_summary.get("binary_prevalence_target", float("nan")),
                "binary_liability_logit_scale": binary_summary.get("binary_liability_logit_scale", float("nan")),
                "binary_alpha": binary_summary.get("binary_alpha", float("nan")),
                "binary_alpha_by_population": binary_summary.get("binary_alpha_by_population", "{}"),
                "binary_prevalence_target_by_population": binary_summary.get(
                    "binary_prevalence_target_by_population",
                    "{}",
                ),
                "binary_case_count": binary_summary.get("binary_case_count", 0),
                "binary_control_count": binary_summary.get("binary_control_count", 0),
                "binary_realized_prevalence": binary_summary.get("binary_realized_prevalence", float("nan")),
                "binary_mean_predicted_probability": binary_summary.get(
                    "binary_mean_predicted_probability",
                    float("nan"),
                ),
                "binary_alpha_mode": binary_summary.get("binary_alpha_mode", ""),
                "binary_admix_alpha_mode": binary_summary.get("binary_admix_alpha_mode", ""),
                "binary_admix_sample_count": binary_summary.get("binary_admix_sample_count", 0),
                "binary_admix_realized_prevalence": binary_summary.get(
                    "binary_admix_realized_prevalence",
                    float("nan"),
                ),
                "binary_admix_mean_predicted_probability": binary_summary.get(
                    "binary_admix_mean_predicted_probability",
                    float("nan"),
                ),
                "binary_admix_alpha_mean": binary_summary.get("binary_admix_alpha_mean", float("nan")),
                "binary_admix_alpha_min": binary_summary.get("binary_admix_alpha_min", float("nan")),
                "binary_admix_alpha_max": binary_summary.get("binary_admix_alpha_max", float("nan")),
                "shift_tightness_tier": shift_tightness_tier,
                "shift_tightness_fold_change_cap": shift_tightness_fold_change_cap,
                "shift_tightness_fold_change_max_kept": shift_tightness_fold_change_max_kept,
                "shift_tightness_fold_change_quantiles_json": shift_tightness_fold_change_quantiles_json,
                "shift_tightness_pre_filter_count": shift_tightness_pre_filter_count,
                "shift_tightness_post_filter_count": shift_tightness_post_filter_count,
            }
        )
        geometry_key = (replicate, architecture)
        if geometry_key not in effect_geometry_seen:
            effect_geometry_seen.add(geometry_key)
            effect_geometry_rows.append(
                {
                    "run_name": run_name,
                    "replicate": replicate,
                    "architecture": architecture,
                    "population_corr_mode": effect_geometry.get("population_corr_mode", ""),
                    "base_jitter_sd": effect_geometry.get("base_jitter_sd", 0.0),
                    "resolved_jitter_sd_by_population": json.dumps(
                        effect_geometry.get("resolved_jitter_sd_by_population", {}),
                        sort_keys=True,
                    ),
                    "population_fst_file": effect_geometry.get("population_fst_file", ""),
                    "fst_corr_transform": effect_geometry.get("fst_corr_transform", ""),
                    "fst_corr_lambda": effect_geometry.get("fst_corr_lambda", 0.0),
                    "resolved_population_corr_matrix": effect_geometry.get(
                        "population_corr_matrix_json", "[]"
                    ),
                }
            )

    membership_df = pd.concat(membership_frames, ignore_index=True)
    effects_df = pd.concat(effect_frames, ignore_index=True)
    manifest_df = pd.DataFrame(manifest_rows)
    component_df = pd.DataFrame(component_rows)
    membership_work = membership_df.copy()
    if "annotation_hit" not in membership_work.columns:
        membership_work["annotation_hit"] = False
    if "maf_bucket" not in membership_work.columns:
        membership_work["maf_bucket"] = ""
    membership_work["annotation_hit"] = membership_work["annotation_hit"].fillna(False).astype(bool)
    membership_summary_df = (
        membership_work.groupby(
            ["replicate", "architecture", "cohort", "maf_bucket"],
            dropna=False,
            as_index=False,
        )["annotation_hit"]
        .agg(selected_causal_count="size", annotation_hit_count="sum")
        .assign(
            annotation_hit_fraction=lambda frame: frame["annotation_hit_count"]
            / frame["selected_causal_count"].clip(lower=1)
        )
    )
    effect_geometry_df = pd.DataFrame(effect_geometry_rows)

    _atomic_to_csv(membership_df, truth_dir / "causal_membership.tsv")
    _atomic_to_csv(membership_summary_df, truth_dir / "causal_membership_summary.tsv")
    _atomic_to_csv(effects_df, truth_dir / "effect_sizes.tsv")
    _atomic_to_csv(component_df, truth_dir / "component_summary.tsv")
    _atomic_to_csv(effect_geometry_df, truth_dir / "effect_geometry.tsv")
    _atomic_to_csv(manifest_df, run_dir / "run_manifest.tsv")
    _atomic_write_json(
        {
            "schema_version": 2,
            "run_name": run_name,
            "run_identity": str(run_identity),
            "manifest": "run_manifest.tsv",
            "n_manifest_rows": int(len(manifest_df)),
            "n_records": int(len(records)),
        },
        completion_path,
    )

    return manifest_df
