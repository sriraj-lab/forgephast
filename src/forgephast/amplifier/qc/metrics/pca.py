"""PCA-based QC (donor-fit basis + synthetic projection)."""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import RunConfig
from ..io import haplotypes_to_genotypes_f32
from ...utils import derived_seed
from .statistics import auc_from_scores as _auc_from_scores

logger = logging.getLogger(__name__)

_SVD_FALLBACK_MAX_ELEMENTS = 2_000_000
_SVD_FALLBACK_MAX_DIM = 2000


def _standardize_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x - mean) / std, mean, std


def _randomized_svd_compat(
    x: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Randomized SVD with a safe fallback when scikit-learn isn't usable.

    For large matrices, we require scikit-learn to avoid accidentally running an
    O(n^3) full SVD.
    """
    try:
        from sklearn.utils.extmath import randomized_svd  # type: ignore

        u, s, vt = randomized_svd(x, n_components=int(n_components), random_state=int(random_state))
        return (
            np.asarray(u, dtype=np.float64),
            np.asarray(s, dtype=np.float64),
            np.asarray(vt, dtype=np.float64),
            "sklearn_randomized_svd",
        )
    except Exception as exc:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size > _SVD_FALLBACK_MAX_ELEMENTS or min(x_arr.shape) > _SVD_FALLBACK_MAX_DIM:
            raise RuntimeError(
                "PCA QC requires a working scikit-learn installation for large matrices; "
                "please install/repair `scikit-learn`."
            ) from exc

        u, s, vt = np.linalg.svd(x_arr, full_matrices=False)
        k = min(int(n_components), int(vt.shape[0]))
        return u[:, :k], s[:k], vt[:k], "numpy_svd_fallback"


def _center_scale(
    g: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    x = np.asarray(g, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"genotypes must be 2D, got shape={x.shape}")
    if np.any(~np.isfinite(x)):
        raise ValueError("PCA genotypes must be finite after missing-variant filtering")
    out = x.copy()
    out -= mean
    out /= scale
    return out


def compute_pca_auc_metrics(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    ext_data: Any | None = None,
    ext_data_error: str | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Compute PCA projection QC and donor-vs-synthetic AUC agreement."""
    from ..plots import save_pca_scatter_png

    enabled = bool(config.qc.pca_enabled)
    payload: dict[str, Any] = {
        "enabled": enabled,
        "chrom": int(chrom),
        "status": "disabled" if not enabled else "init",
    }
    created: list[Path] = []
    if not enabled:
        return payload, created

    if ext_data is None:
        payload["status"] = "no_ext_cache"
        payload["error"] = ext_data_error
        payload["auc_by_pair"] = []
        return payload, created

    pops = list(getattr(ext_data, "pops", []))
    if not pops:
        payload["status"] = "no_populations"
        payload["auc_by_pair"] = []
        return payload, created

    global_pos = np.asarray(getattr(ext_data, "global_pos"), dtype=np.int64)
    sampled_pos = np.asarray(getattr(ext_data, "sampled_pos"), dtype=np.int64)
    subset_id = str(getattr(ext_data, "variant_subset_id", ""))

    # Build donor/synth genotype matrices.
    donor_blocks: list[np.ndarray] = []
    synth_blocks: list[np.ndarray] = []
    donor_labels: list[str] = []
    synth_labels: list[str] = []

    for pop in pops:
        pop_data = ext_data.per_pop[pop]
        donor_h = pop_data.donor_haps
        synth_h = pop_data.synth_haps

        donor_g = haplotypes_to_genotypes_f32(donor_h)
        synth_g = haplotypes_to_genotypes_f32(synth_h)

        donor_blocks.append(donor_g)
        synth_blocks.append(synth_g)
        donor_labels.extend([pop] * donor_g.shape[0])
        synth_labels.extend([pop] * synth_g.shape[0])

    donor_g_all = np.vstack(donor_blocks) if donor_blocks else np.zeros((0, sampled_pos.size), dtype=np.float32)
    synth_g_all = np.vstack(synth_blocks) if synth_blocks else np.zeros((0, sampled_pos.size), dtype=np.float32)

    if donor_g_all.shape[0] < 3 or synth_g_all.shape[0] < 3:
        payload.update({"status": "insufficient_samples", "n_donor": int(donor_g_all.shape[0]), "n_synth": int(synth_g_all.shape[0])})
        payload["auc_by_pair"] = []
        return payload, created

    # Donor allele frequency p (from genotypes), filter by MAF.
    observed = np.isfinite(donor_g_all)
    ac = np.nansum(donor_g_all.astype(np.float64), axis=0)
    n_obs = np.sum(observed, axis=0, dtype=np.float64)
    an = 2.0 * n_obs
    with np.errstate(divide="ignore", invalid="ignore"):
        p = ac / an
    maf = np.minimum(p, 1.0 - p)
    complete = np.all(np.isfinite(donor_g_all), axis=0) & np.all(np.isfinite(synth_g_all), axis=0)
    valid = (
        complete
        & np.isfinite(p)
        & np.isfinite(maf)
        & (an >= 4.0)
        & (maf >= float(config.qc.pca_min_maf))
        & (maf <= 0.5)
    )
    if np.count_nonzero(valid) < 2:
        payload.update({"status": "no_variants_after_filter", "n_variants": int(np.count_nonzero(valid))})
        payload["auc_by_pair"] = []
        return payload, created

    p = p[valid]
    mean = 2.0 * p
    scale = np.sqrt(2.0 * p * (1.0 - p))
    keep_scale = scale > 1e-12
    if np.count_nonzero(keep_scale) < 2:
        payload.update({"status": "no_variants_after_scale", "n_variants": int(np.count_nonzero(keep_scale))})
        payload["auc_by_pair"] = []
        return payload, created

    mean = mean[keep_scale]
    scale = scale[keep_scale]
    donor_g_f = donor_g_all[:, valid][:, keep_scale]
    synth_g_f = synth_g_all[:, valid][:, keep_scale]

    x_donor = _center_scale(donor_g_f, mean=mean, scale=scale)
    x_synth = _center_scale(synth_g_f, mean=mean, scale=scale)

    requested_components = int(config.qc.pca_n_components)
    n_components = min(requested_components, x_donor.shape[0] - 1, x_donor.shape[1])
    if n_components <= 0:
        payload.update({"status": "insufficient_rank", "n_components": int(n_components)})
        payload["auc_by_pair"] = []
        return payload, created

    # PCA on donor, projection by matrix multiply.
    random_state = int(derived_seed(int(config.qc.seed), "pca_svd", chrom))
    _, _, vt, svd_backend = _randomized_svd_compat(
        x_donor,
        n_components=int(n_components),
        random_state=random_state,
    )
    donor_scores = x_donor @ vt.T
    synth_scores = x_synth @ vt.T

    donor_scores_std, donor_pc_mean, donor_pc_std = _standardize_scores(donor_scores)
    synth_scores_std = (synth_scores - donor_pc_mean) / donor_pc_std

    auc_k = min(int(config.qc.pca_auc_n_components), int(n_components))
    auc_by_pair: list[dict[str, Any]] = []
    abs_diffs: list[float] = []

    for pop1, pop2 in combinations(pops, 2):
        donor_mask_1 = np.array([x == pop1 for x in donor_labels], dtype=bool)
        donor_mask_2 = np.array([x == pop2 for x in donor_labels], dtype=bool)
        synth_mask_1 = np.array([x == pop1 for x in synth_labels], dtype=bool)
        synth_mask_2 = np.array([x == pop2 for x in synth_labels], dtype=bool)

        d1 = donor_scores_std[donor_mask_1, :auc_k]
        d2 = donor_scores_std[donor_mask_2, :auc_k]
        s1 = synth_scores_std[synth_mask_1, :auc_k]
        s2 = synth_scores_std[synth_mask_2, :auc_k]
        if d1.shape[0] == 0 or d2.shape[0] == 0 or s1.shape[0] == 0 or s2.shape[0] == 0:
            continue

        direction = np.mean(d1, axis=0) - np.mean(d2, axis=0)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:
            auc_d = float("nan")
            auc_s = float("nan")
        else:
            scores_d_pos = d1 @ direction
            scores_d_neg = d2 @ direction
            scores_s_pos = s1 @ direction
            scores_s_neg = s2 @ direction
            auc_d = _auc_from_scores(scores_d_pos, scores_d_neg)
            auc_s = _auc_from_scores(scores_s_pos, scores_s_neg)
            auc_d = float(max(auc_d, 1.0 - auc_d))
            auc_s = float(max(auc_s, 1.0 - auc_s))

        abs_diff = float(abs(auc_s - auc_d)) if np.isfinite(auc_s) and np.isfinite(auc_d) else float("nan")
        if np.isfinite(abs_diff):
            abs_diffs.append(abs_diff)

        auc_by_pair.append(
            {
                "pop1": pop1,
                "pop2": pop2,
                "n_donor_1": int(np.sum(donor_mask_1)),
                "n_donor_2": int(np.sum(donor_mask_2)),
                "n_synth_1": int(np.sum(synth_mask_1)),
                "n_synth_2": int(np.sum(synth_mask_2)),
                "auc_donor": float(auc_d),
                "auc_synthetic": float(auc_s),
                "abs_diff": abs_diff,
            }
        )

    table_path = diagnostics_dir / f"pca_auc_chr{chrom}.tsv"
    pd.DataFrame.from_records(auc_by_pair).to_csv(table_path, sep="\t", index=False)
    created.append(table_path)

    plot_path: Path | None = None
    plot_error: str | None = None
    if config.qc.plots_enabled and donor_scores_std.shape[1] >= 2 and synth_scores_std.shape[1] >= 2:
        try:
            plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"pca_pc12_chr{chrom}.png"
            save_pca_scatter_png(
                plot_path,
                donor_scores=donor_scores_std[:, :2],
                donor_labels=donor_labels,
                synthetic_scores=synth_scores_std[:, :2],
                synthetic_labels=synth_labels,
                title=f"PCA projection chr{chrom}",
            )
            created.append(plot_path)
        except Exception as exc:
            logger.warning("PCA QC plot generation failed; continuing without plot: %s", exc)
            plot_error = str(exc)
            plot_path = None

    payload.update(
        {
            "status": "ok",
            "seed": int(config.qc.seed),
            "ext_max_diploids_per_population": int(config.qc.ext_max_diploids_per_population),
            "ext_max_variants": int(config.qc.ext_max_variants),
            "variant_subset_id": subset_id,
            "pca_n_components_requested": int(config.qc.pca_n_components),
            "pca_n_components_used": int(n_components),
            "pca_auc_n_components": int(auc_k),
            "svd_backend": svd_backend,
            "svd_random_state": int(random_state),
            "pca_min_maf": float(config.qc.pca_min_maf),
            "n_positions_global_common": int(global_pos.size),
            "n_positions_sampled": int(sampled_pos.size),
            "n_variants_used": int(mean.size),
            "n_donor_samples": int(donor_g_all.shape[0]),
            "n_synth_samples": int(synth_g_all.shape[0]),
            "table_path": str(table_path),
            "plot_path": str(plot_path) if plot_path is not None else None,
            "plot_error": plot_error,
            "auc_by_pair": auc_by_pair,
            "abs_diff_mean": float(np.mean(abs_diffs)) if abs_diffs else float("nan"),
            "abs_diff_max": float(np.max(abs_diffs)) if abs_diffs else float("nan"),
        }
    )

    logger.info(
        "PCA QC complete chrom=%s pops=%s n_components=%s table=%s plot=%s",
        chrom,
        len(pops),
        n_components,
        table_path,
        plot_path,
    )
    return payload, created
