"""Output writers for simulation and manifest artifacts."""

from __future__ import annotations

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

        component_rows.append(
            {
                "run_name": run_name,
                "replicate": replicate,
                "architecture": architecture,
                "cohort": cohort,
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
            }
        )

        manifest_rows.append(
            {
                "run_name": run_name,
                "replicate": replicate,
                "architecture": architecture,
                "cohort": cohort,
                "plugin": rec["plugin"],
                "pgen_prefix": rec["pgen_prefix"],
                "sample_ids_file": rec["sample_ids_file"],
                "keep_file": str(keep_path),
                "phenotype_file": str(phenotype_path),
                "covariates_file": str(covariate_path),
                "global_pcs_file": str(pcs_path),
                "covariate_coefficients_file": str(coeff_path),
                "output_dir": str(out_dir),
                "target_var_genetic_fixed": targets.get("genetic_fixed", 0.0),
                "target_var_genetic_bg": targets.get("genetic_bg", 0.0),
                "target_var_genetic_total": targets.get("genetic_total", 0.0),
                "target_var_covariate": targets.get("covariate", 0.0),
                "target_var_residual": targets.get("residual", 0.0),
            }
        )

    membership_df = pd.concat(membership_frames, ignore_index=True)
    effects_df = pd.concat(effect_frames, ignore_index=True)
    manifest_df = pd.DataFrame(manifest_rows)
    component_df = pd.DataFrame(component_rows)

    membership_df.to_csv(truth_dir / "causal_membership.tsv", sep="\t", index=False)
    effects_df.to_csv(truth_dir / "effect_sizes.tsv", sep="\t", index=False)
    component_df.to_csv(truth_dir / "component_summary.tsv", sep="\t", index=False)
    manifest_df.to_csv(run_dir / "run_manifest.tsv", sep="\t", index=False)

    return manifest_df
