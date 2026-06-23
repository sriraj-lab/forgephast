"""Output writers for simulation and manifest artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _write_keep_file(phenotype_df: pd.DataFrame, out_path: Path) -> None:
    """Write normalized FID IID keep file."""
    keep = phenotype_df[["FID", "IID"]].copy()
    keep.to_csv(out_path, sep="\t", index=False)


def write_simulation_outputs(
    run_name: str,
    output_root: Path,
    records: List[Dict[str, object]],
) -> pd.DataFrame:
    """Persist simulation record tables and return run manifest dataframe."""
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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

        phenotype_df.to_csv(phenotype_path, sep="\t", index=False)
        covariate_df.to_csv(covariate_path, sep="\t", index=False)
        pcs_df.to_csv(pcs_path, sep="\t", index=False)
        coeff_df.to_csv(coeff_path, sep="\t", index=False)
        _write_keep_file(phenotype_df, keep_path)

        membership_frames.append(rec["membership_df"])
        effect_frames.append(rec["effects_df"])

        targets = rec["component_targets"]
        realized = rec.get("component_realized", {})
        effect_geometry = rec.get("effect_geometry", {})
        realized_fixed_by_population = rec.get("realized_fixed_by_population", {}) or {}
        tightness_summary = rec.get("tightness_summary", {}) or {}

        realized_h2_fixed_by_population_json = json.dumps(
            {str(k): float(v) for k, v in realized_fixed_by_population.items()},
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
                "genetic_fixed_beta_scale_factor": rec.get("genetic_fixed_beta_scale_factor", 1.0),
                "base_h2_fixed": targets.get("base_h2_fixed", targets.get("genetic_fixed", 0.0)),
                "resolved_h2_fixed": targets.get("resolved_h2_fixed", targets.get("genetic_fixed", 0.0)),
                "h2_jitter_delta": targets.get("h2_jitter_delta", 0.0),
                "h2_jitter_clipped": targets.get("h2_jitter_clipped", False),
                "target_mode": targets.get("target_mode", "single_population"),
                "resolved_h2_fixed_by_population": targets.get(
                    "resolved_h2_fixed_by_population_json",
                    "{}",
                ),
                "resolved_h2_bg_by_population": targets.get(
                    "resolved_h2_bg_by_population_json",
                    "{}",
                ),
                "resolved_covariate_by_population": targets.get(
                    "resolved_covariate_by_population_json",
                    "{}",
                ),
                "resolved_residual_by_population": targets.get(
                    "resolved_residual_by_population_json",
                    "{}",
                ),
                "h2_jitter_delta_by_population": targets.get(
                    "h2_jitter_delta_by_population_json",
                    "{}",
                ),
                "h2_jitter_clipped_by_population": targets.get(
                    "h2_jitter_clipped_by_population_json",
                    "{}",
                ),
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
                "realized_h2_fixed_by_population": realized_h2_fixed_by_population_json,
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
                "plugin": rec["plugin"],
                "pgen_prefix": rec["pgen_prefix"],
                "sample_ids_file": rec["sample_ids_file"],
                "keep_file": str(keep_path),
                "phenotype_file": str(phenotype_path),
                "covariates_file": str(covariate_path),
                "global_pcs_file": str(pcs_path),
                "covariate_coefficients_file": str(coeff_path),
                "output_dir": str(out_dir),
                "base_h2_fixed": targets.get("base_h2_fixed", targets.get("genetic_fixed", 0.0)),
                "resolved_h2_fixed": targets.get("resolved_h2_fixed", targets.get("genetic_fixed", 0.0)),
                "h2_jitter_delta": targets.get("h2_jitter_delta", 0.0),
                "h2_jitter_clipped": targets.get("h2_jitter_clipped", False),
                "target_mode": targets.get("target_mode", "single_population"),
                "resolved_h2_fixed_by_population": targets.get(
                    "resolved_h2_fixed_by_population_json",
                    "{}",
                ),
                "resolved_h2_bg_by_population": targets.get(
                    "resolved_h2_bg_by_population_json",
                    "{}",
                ),
                "resolved_covariate_by_population": targets.get(
                    "resolved_covariate_by_population_json",
                    "{}",
                ),
                "resolved_residual_by_population": targets.get(
                    "resolved_residual_by_population_json",
                    "{}",
                ),
                "h2_jitter_delta_by_population": targets.get(
                    "h2_jitter_delta_by_population_json",
                    "{}",
                ),
                "h2_jitter_clipped_by_population": targets.get(
                    "h2_jitter_clipped_by_population_json",
                    "{}",
                ),
                "target_var_genetic_fixed": targets.get("genetic_fixed", 0.0),
                "target_var_genetic_bg": targets.get("genetic_bg", 0.0),
                "target_var_genetic_total": targets.get("genetic_total", 0.0),
                "target_var_covariate": targets.get("covariate", 0.0),
                "target_var_residual": targets.get("residual", 0.0),
                "realized_h2_fixed_by_population": realized_h2_fixed_by_population_json,
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

    membership_df.to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)
    membership_summary_df.to_csv(
        truth_dir / "causal_membership_summary.tsv",
        sep="\t",
        index=False,
    )
    effects_df.to_csv(truth_dir / "effect_sizes.tsv", sep="\t", index=False)
    component_df.to_csv(truth_dir / "component_summary.tsv", sep="\t", index=False)
    effect_geometry_df.to_csv(truth_dir / "effect_geometry.tsv", sep="\t", index=False)
    manifest_df.to_csv(run_dir / "run_manifest.tsv", sep="\t", index=False)

    return manifest_df
