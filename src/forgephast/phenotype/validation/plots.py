"""Round-trip plotting helpers for GWAS validation outputs."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from hapstore.ids import align_by_ids, read_sample_ids

from ..config import RunConfig
from .roundtrip import (
    RoundTripComparison,
    _linear_metrics,
    _population_weights,
    _weighted_truth_effects,
    collect_roundtrip_details,
    merge_assoc_truth_effects,
)

logger = logging.getLogger(__name__)
QQ_AXIS_CAP = 7.0


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "item"


def _comparison_dir(run_dir: Path, comparison: RoundTripComparison) -> Path:
    return (
        run_dir
        / "plots"
        / "roundtrip"
        / (
            f"rep{int(comparison.replicate)}__"
            f"{_slugify(comparison.architecture)}__"
            f"{_slugify(comparison.cohort)}__"
            f"{_slugify(comparison.engine)}"
        )
    )


def _save_figure(fig, output_base: Path, formats: list[str], dpi: int) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        out_path = output_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi)
        paths.append(out_path)
    return paths


def _robust_symmetric_limits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    low_q: float = 0.5,
    high_q: float = 99.5,
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)
    lo = float(np.nanpercentile(vals, low_q))
    hi = float(np.nanpercentile(vals, high_q))
    lim = max(abs(lo), abs(hi))
    if not np.isfinite(lim) or lim <= 0.0:
        lim = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    if not np.isfinite(lim) or lim <= 0.0:
        lim = 1.0
    lim *= 1.0 + float(pad_frac)
    return (-lim, lim)


def _chrom_rank(chrom: str) -> int:
    token = str(chrom).replace("chr", "")
    if token.isdigit():
        return int(token)
    special = {"X": 23, "Y": 24, "MT": 25, "M": 25}
    return special.get(token.upper(), 10_000)


def _manhattan_table(assoc_df: pd.DataFrame) -> pd.DataFrame:
    work = assoc_df.copy()
    work["chrom"] = work["chrom"].astype(str).str.replace("^chr", "", regex=True)
    work["pos"] = pd.to_numeric(work["pos"], errors="coerce")
    work["p_value"] = pd.to_numeric(work["p_value"], errors="coerce")
    work = work[np.isfinite(work["pos"]) & np.isfinite(work["p_value"]) & (work["p_value"] > 0)].copy()
    if work.empty:
        return work
    work["chrom_rank"] = work["chrom"].map(_chrom_rank).astype(int)
    work = work.sort_values(["chrom_rank", "pos", "variant_id"]).reset_index(drop=True)
    offsets: dict[int, float] = {}
    centers: list[dict[str, Any]] = []
    running = 0.0
    for chrom_rank, group in work.groupby("chrom_rank", sort=True):
        chrom_label = str(group["chrom"].iloc[0])
        min_pos = float(group["pos"].min())
        max_pos = float(group["pos"].max())
        offsets[int(chrom_rank)] = running - min_pos
        centers.append({"chrom": chrom_label, "center": running + (max_pos - min_pos) / 2.0})
        running += (max_pos - min_pos) + 1e6
    work["plot_x"] = work["pos"].astype(float) + work["chrom_rank"].map(offsets).astype(float)
    work["neg_log10_p"] = -np.log10(work["p_value"].clip(lower=np.nextafter(0.0, 1.0)))
    work.attrs["centers"] = centers
    return work


def _qq_table(assoc_df: pd.DataFrame, max_points: int = 50_000) -> pd.DataFrame:
    pvals = pd.to_numeric(assoc_df["p_value"], errors="coerce")
    pvals = pvals[np.isfinite(pvals) & (pvals > 0) & (pvals <= 1)].to_numpy(dtype=float)
    if pvals.size == 0:
        return pd.DataFrame(columns=["expected", "observed"])
    pvals.sort()
    expected = -np.log10((np.arange(1, pvals.size + 1, dtype=float) - 0.5) / float(pvals.size))
    observed = -np.log10(pvals)
    if pvals.size > max_points:
        idx = np.linspace(0, pvals.size - 1, max_points, dtype=int)
        expected = expected[idx]
        observed = observed[idx]
    return pd.DataFrame({"expected": expected, "observed": observed})


def _topk_curve_table(assoc_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    work = assoc_df.sort_values("p_value", ascending=True).reset_index(drop=True).copy()
    if work.empty:
        return pd.DataFrame(columns=["rank", "cumulative_causal_fraction"])
    work["rank"] = np.arange(1, work.shape[0] + 1, dtype=int)
    total_causal = max(1, int(work["is_causal"].sum()))
    work["cumulative_causal_fraction"] = work["is_causal"].astype(int).cumsum() / float(total_causal)
    work["in_top_k"] = work["rank"] <= int(top_k)
    return work[["rank", "cumulative_causal_fraction", "in_top_k"]].copy()


def _enrichment_rows(comparison: RoundTripComparison) -> list[dict[str, Any]]:
    assoc_df = comparison.assoc_df
    top_df = comparison.top_df
    gwsig = assoc_df[pd.to_numeric(assoc_df["p_value"], errors="coerce") < 5e-8].copy()
    rows = [
        {
            "label": "All variants",
            "fraction": float(assoc_df["is_causal"].mean()) if not assoc_df.empty else np.nan,
        },
        {
            "label": f"Top {int(comparison.metrics['top_k'])}",
            "fraction": float(top_df["is_causal"].mean()) if not top_df.empty else np.nan,
        },
        {
            "label": "P < 5e-8",
            "fraction": float(gwsig["is_causal"].mean()) if not gwsig.empty else np.nan,
        },
    ]
    return rows


def _annotation_color_values(merged_df: pd.DataFrame) -> tuple[np.ndarray, dict[bool, str]]:
    hit = merged_df.get("annotation_hit", pd.Series(False, index=merged_df.index)).fillna(False).astype(bool)
    colors = {False: "#4C78A8", True: "#E45756"}
    return hit.to_numpy(dtype=bool), colors


def _write_beta_scatter(
    comparison: RoundTripComparison,
    *,
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    merged = comparison.merged_df.copy()
    x = merged["beta_true"].to_numpy(dtype=float)
    y = merged["beta_hat_harmonized"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    merged = merged.loc[mask].copy()
    x = merged["beta_true"].to_numpy(dtype=float)
    y = merged["beta_hat_harmonized"].to_numpy(dtype=float)
    hit, colors = _annotation_color_values(merged)
    for label in (False, True):
        sub = merged.loc[hit == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["beta_true"],
            sub["beta_hat_harmonized"],
            s=24,
            alpha=0.80,
            linewidths=0.0,
            color=colors[label],
            label="Annotated" if label else "Not annotated",
        )
    lo, hi = _robust_symmetric_limits(x, y)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0, label="y=x")
    if np.count_nonzero(mask) >= 2 and np.isfinite(comparison.metrics["beta_slope"]):
        fit_x = np.linspace(lo, hi, 200)
        fit_y = (
            float(comparison.metrics["beta_intercept"])
            + float(comparison.metrics["beta_slope"]) * fit_x
        )
        ax.plot(fit_x, fit_y, color="#2F4B7C", linewidth=1.5, label="Fit")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True effect size")
    ax.set_ylabel("GWAS effect estimate")
    ax.set_title(
        f"Round-trip effect recovery\nrep{comparison.replicate} | "
        f"{comparison.architecture} | {comparison.cohort} | {comparison.engine}"
    )
    summary = (
        f"n={int(comparison.metrics['n_overlap'])}\n"
        f"slope={float(comparison.metrics['beta_slope']):.3f}\n"
        f"r={float(comparison.metrics['beta_corr']):.3f}\n"
        f"sign={float(comparison.metrics['sign_concordance']):.3f}"
    )
    ax.text(0.03, 0.97, summary, transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="lower right", frameon=False)
    return _save_figure(fig, output_base, formats, dpi)


def _write_manhattan(
    comparison: RoundTripComparison,
    *,
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path] | None:
    man = _manhattan_table(comparison.assoc_df)
    if man.empty:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    from matplotlib.lines import Line2D

    palette = ["#B8B8B8", "#D8D8D8"]
    for idx, (_, chrom_df) in enumerate(man.groupby("chrom_rank", sort=True)):
        ax.scatter(
            chrom_df["plot_x"],
            chrom_df["neg_log10_p"],
            s=4,
            alpha=0.55,
            linewidths=0.0,
            color=palette[idx % len(palette)],
        )
    causal = man[man["is_causal"]].copy()
    if not causal.empty:
        effect_lookup = comparison.merged_df[["variant_id", "beta_true"]].drop_duplicates("variant_id")
        causal = causal.merge(effect_lookup, on="variant_id", how="left")
        effect_abs = pd.to_numeric(causal["beta_true"], errors="coerce").abs()
        vmin = 0.0
        vmax = float(effect_abs.max()) if np.isfinite(effect_abs.max()) and float(effect_abs.max()) > 0 else 1.0
        causal_scatter = ax.scatter(
            causal["plot_x"],
            causal["neg_log10_p"],
            c=effect_abs,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=34,
            alpha=0.90,
            linewidths=0.3,
            edgecolors="black",
            marker="^",
        )
        annotated = causal[causal["annotation_hit"].fillna(False).astype(bool)]
        if not annotated.empty:
            annotated_effect_abs = pd.to_numeric(annotated["beta_true"], errors="coerce").abs()
            ax.scatter(
                annotated["plot_x"],
                annotated["neg_log10_p"],
                c=annotated_effect_abs,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=70,
                alpha=0.98,
                linewidths=0.7,
                edgecolors="black",
                marker="*",
            )
        cbar = fig.colorbar(causal_scatter, ax=ax, pad=0.02)
        cbar.set_label("|True effect size|")
    ax.axhline(-math.log10(5e-8), linestyle="--", linewidth=1.0, color="black")
    centers = man.attrs.get("centers", [])
    if centers:
        ax.set_xticks([float(item["center"]) for item in centers])
        ax.set_xticklabels([str(item["chrom"]) for item in centers])
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(p)")
    ax.set_title(
        f"GWAS Manhattan plot\nrep{comparison.replicate} | "
        f"{comparison.architecture} | {comparison.cohort} | {comparison.engine}"
    )
    if not causal.empty:
        legend_handles = [
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#666666", markeredgecolor="black", markersize=7, label="True causal"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#666666", markeredgecolor="black", markersize=10, label="Annotated causal"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    return _save_figure(fig, output_base, formats, dpi)


def _write_qq(
    comparison: RoundTripComparison,
    *,
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path] | None:
    qq = _qq_table(comparison.assoc_df)
    if qq.empty:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    ax.scatter(qq["expected"], qq["observed"], s=8, alpha=0.60, linewidths=0.0, color="#4C78A8")
    lim = float(QQ_AXIS_CAP)
    ax.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title(
        f"GWAS QQ plot\nrep{comparison.replicate} | "
        f"{comparison.architecture} | {comparison.cohort} | {comparison.engine}"
    )
    pvals = pd.to_numeric(comparison.assoc_df["p_value"], errors="coerce")
    pvals = pvals[np.isfinite(pvals) & (pvals > 0) & (pvals <= 1)].to_numpy(dtype=float)
    lambda_gc = np.nan
    if pvals.size:
        chisq = chi2.isf(pvals, 1)
        lambda_gc = float(np.median(chisq) / 0.4549364)
    summary = []
    if float(qq["observed"].max()) > lim:
        summary.append(f"Axes clipped at {lim:g}")
    if np.isfinite(lambda_gc):
        summary.append(f"lambda_GC={lambda_gc:.3f}")
    if summary:
        ax.text(0.03, 0.97, "\n".join(summary), transform=ax.transAxes, va="top", ha="left")
    return _save_figure(fig, output_base, formats, dpi)


def _write_topk_recovery(
    comparison: RoundTripComparison,
    *,
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path] | None:
    curve = _topk_curve_table(comparison.assoc_df, int(comparison.metrics["top_k"]))
    if curve.empty:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(
        curve["rank"],
        curve["cumulative_causal_fraction"],
        color="#4C78A8",
        linewidth=1.8,
    )
    top_k = int(comparison.metrics["top_k"])
    top_frac = float(comparison.top_df["is_causal"].sum() / max(1, comparison.truth_df.shape[0]))
    ax.axvline(top_k, linestyle="--", linewidth=1.0, color="black")
    ax.text(
        top_k,
        min(1.0, top_frac + 0.05),
        f"top_k={top_k}\nrecovered={top_frac:.3f}",
        ha="left",
        va="bottom",
    )
    ax.set_xlim(1, max(10, int(curve["rank"].max())))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Rank by GWAS p-value")
    ax.set_ylabel("Cumulative fraction of true causals recovered")
    ax.set_title(
        f"Top-k causal recovery\nrep{comparison.replicate} | "
        f"{comparison.architecture} | {comparison.cohort} | {comparison.engine}"
    )
    return _save_figure(fig, output_base, formats, dpi)


def _write_enrichment_summary(
    comparison: RoundTripComparison,
    *,
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    rows = _enrichment_rows(comparison)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    labels = [row["label"] for row in rows]
    vals = [row["fraction"] for row in rows]
    plot_vals = [0.0 if not np.isfinite(v) else float(v) for v in vals]
    bars = ax.bar(labels, plot_vals, color=["#4C78A8", "#2F4B7C", "#E45756"])
    for bar, raw in zip(bars, vals):
        text = "n/a" if not np.isfinite(raw) else f"{float(raw):.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), text, ha="center", va="bottom")
    ax.set_ylim(0.0, max(0.05, max(plot_vals) * 1.2 if plot_vals else 0.05))
    ax.set_ylabel("Fraction of variants that are causal")
    ax.set_title(
        f"GWAS enrichment summary\nrep{comparison.replicate} | "
        f"{comparison.architecture} | {comparison.cohort} | {comparison.engine}"
    )
    return _save_figure(fig, output_base, formats, dpi)


def _group_assignments(
    *,
    sample_group_file: Path,
    sample_ids: list[str],
    id_column: str,
    group_column: str,
    cohort_name: str,
) -> pd.DataFrame:
    groups = pd.read_csv(sample_group_file, sep="\t")
    if id_column not in groups.columns:
        raise ValueError(
            f"roundtrip_plots.sample_group_file missing id column {id_column!r}: {sample_group_file}"
        )
    if group_column not in groups.columns:
        raise ValueError(
            f"roundtrip_plots.sample_group_file missing group column {group_column!r}: {sample_group_file}"
        )
    aligned = align_by_ids(
        groups[[id_column, group_column]].copy(),
        id_column=id_column,
        ids=sample_ids,
        context=f"roundtrip plots group alignment for cohort {cohort_name}",
    )
    aligned["IID"] = aligned[id_column].astype(str)
    aligned["group"] = aligned[group_column].astype(str)
    return aligned[["IID", "group"]].copy()


def _group_metrics_table(
    *,
    comparison: RoundTripComparison,
    run_cfg: RunConfig,
    run_dir: Path,
    sample_group_file: Path,
    id_column: str,
    group_column: str,
) -> pd.DataFrame:
    cohort = next((item for item in run_cfg.cohorts if item.name == comparison.cohort), None)
    if cohort is None:
        raise ValueError(f"Cohort {comparison.cohort!r} not found in config")
    sample_ids = read_sample_ids(cohort.sample_ids_file)
    assignments = _group_assignments(
        sample_group_file=sample_group_file,
        sample_ids=sample_ids,
        id_column=id_column,
        group_column=group_column,
        cohort_name=comparison.cohort,
    )
    default_pop = run_cfg.effect_model.population_order[0]
    truth_df = _resolve_truth_effects_for_group(run_dir, comparison, cohort.name)
    rows: list[dict[str, Any]] = []
    for group, group_df in assignments.groupby("group", sort=True):
        group_ids = group_df["IID"].astype(str).tolist()
        pop_weights = _population_weights(cohort, group_ids, run_cfg)
        truth_group = _weighted_truth_effects(
            truth_df=truth_df,
            replicate=comparison.replicate,
            architecture=comparison.architecture,
            cohort=comparison.cohort,
            pop_weights=pop_weights,
            default_population=default_pop,
        )
        merged = merge_assoc_truth_effects(comparison.assoc_df, truth_group)
        slope = intercept = corr = sign = np.nan
        n_overlap = int(merged.shape[0])
        if not merged.empty:
            slope, intercept, corr, sign = _linear_metrics(merged)
        rows.append(
            {
                "group": str(group),
                "n_samples": int(group_df.shape[0]),
                "n_overlap": int(n_overlap),
                "beta_slope": slope,
                "beta_intercept": intercept,
                "beta_corr": corr,
                "sign_concordance": sign,
            }
        )
    return pd.DataFrame(rows)


def _resolve_truth_effects_for_group(
    run_dir: Path,
    comparison: RoundTripComparison,
    cohort_name: str,
) -> pd.DataFrame:
    truth_dir = run_dir / "truth"
    if not truth_dir.exists() and run_dir.name.endswith("__ids_only"):
        truth_dir = run_dir.parent / run_dir.name.removesuffix("__ids_only") / "truth"
    truth_path = truth_dir / "effect_sizes.tsv"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth effect file for group plots: {truth_path}")
    truth_df = pd.read_csv(truth_path, sep="\t")
    return truth_df[
        (truth_df["replicate"] == int(comparison.replicate))
        & (truth_df["architecture"] == str(comparison.architecture))
        & (truth_df["cohort"] == str(cohort_name))
    ].copy()


def _write_group_beta_scatter(
    comparison: RoundTripComparison,
    metrics_df: pd.DataFrame,
    *,
    run_cfg: RunConfig,
    run_dir: Path,
    sample_group_file: Path,
    id_column: str,
    group_column: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> list[Path] | None:
    if metrics_df.empty:
        return None
    cohort = next((item for item in run_cfg.cohorts if item.name == comparison.cohort), None)
    if cohort is None:
        return None
    sample_ids = read_sample_ids(cohort.sample_ids_file)
    assignments = _group_assignments(
        sample_group_file=sample_group_file,
        sample_ids=sample_ids,
        id_column=id_column,
        group_column=group_column,
        cohort_name=comparison.cohort,
    )
    default_pop = run_cfg.effect_model.population_order[0]
    truth_df = _resolve_truth_effects_for_group(run_dir, comparison, comparison.cohort)

    group_frames: dict[str, pd.DataFrame] = {}
    for group, group_df in assignments.groupby("group", sort=True):
        group_ids = group_df["IID"].astype(str).tolist()
        pop_weights = _population_weights(cohort, group_ids, run_cfg)
        truth_group = _weighted_truth_effects(
            truth_df=truth_df,
            replicate=comparison.replicate,
            architecture=comparison.architecture,
            cohort=comparison.cohort,
            pop_weights=pop_weights,
            default_population=default_pop,
        )
        group_frames[str(group)] = merge_assoc_truth_effects(comparison.assoc_df, truth_group)

    non_empty = [frame for frame in group_frames.values() if not frame.empty]
    if not non_empty:
        return None
    all_x = np.concatenate([frame["beta_true"].to_numpy(dtype=float) for frame in non_empty])
    all_y = np.concatenate([frame["beta_hat_harmonized"].to_numpy(dtype=float) for frame in non_empty])
    lo, hi = _robust_symmetric_limits(all_x, all_y)

    n_groups = len(group_frames)
    ncols = 2 if n_groups > 1 else 1
    nrows = int(math.ceil(n_groups / ncols))
    plt = _pyplot()
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.0 * ncols, 5.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    hit_colors = {False: "#4C78A8", True: "#E45756"}
    metric_lookup = metrics_df.set_index("group").to_dict(orient="index")

    groups = list(group_frames.keys())
    for idx, group in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        merged = group_frames[group]
        if merged.empty:
            ax.text(0.5, 0.5, "No overlapping variants", ha="center", va="center")
            ax.set_axis_off()
            continue
        hit, _ = _annotation_color_values(merged)
        for label in (False, True):
            sub = merged.loc[hit == label]
            if sub.empty:
                continue
            ax.scatter(
                sub["beta_true"],
                sub["beta_hat_harmonized"],
                s=20,
                alpha=0.80,
                linewidths=0.0,
                color=hit_colors[label],
                label="Annotated" if label else "Not annotated",
            )
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        metrics = metric_lookup.get(group, {})
        ax.set_title(
            f"{group}\n"
            f"n={int(metrics.get('n_samples', 0))}, "
            f"r={float(metrics.get('beta_corr', np.nan)):.3f}, "
            f"slope={float(metrics.get('beta_slope', np.nan)):.3f}"
        )
        ax.set_xlabel("Group-specific true effect size")
        ax.set_ylabel("GWAS effect estimate")

    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(
        f"Round-trip effect recovery by group\nrep{comparison.replicate} | "
        f"{comparison.architecture} | {comparison.cohort} | {comparison.engine}"
    )
    return _save_figure(fig, output_base, formats, dpi)


def build_roundtrip_plots(
    *,
    run_cfg: RunConfig,
    run_dir: Path,
    gwas_manifest_path: Path,
) -> Path:
    """Build standard round-trip plots and return a plot manifest path."""
    report_df, comparisons, _ = collect_roundtrip_details(
        run_cfg=run_cfg,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )
    _ = report_df
    formats = [str(fmt).lower() for fmt in run_cfg.roundtrip_plots.formats]
    dpi = int(run_cfg.roundtrip_plots.dpi)
    plot_rows: list[dict[str, Any]] = []

    for comparison in comparisons:
        out_dir = _comparison_dir(run_dir, comparison)
        families = [
            ("beta_scatter", _write_beta_scatter, {}),
            ("manhattan", _write_manhattan, {}),
            ("qq", _write_qq, {}),
            ("topk_recovery", _write_topk_recovery, {}),
            ("enrichment_summary", _write_enrichment_summary, {}),
        ]
        for family_name, writer, extra_kwargs in families:
            output_base = out_dir / f"roundtrip_{family_name}"
            written = writer(
                comparison,
                output_base=output_base,
                formats=formats,
                dpi=dpi,
                **extra_kwargs,
            )
            if not written:
                plot_rows.append(
                    {
                        "run_name": comparison.run_name,
                        "replicate": comparison.replicate,
                        "architecture": comparison.architecture,
                        "cohort": comparison.cohort,
                        "engine": comparison.engine,
                        "plot_family": family_name,
                        "status": "skipped",
                        "file_path": "",
                    }
                )
                continue
            for path in written:
                plot_rows.append(
                    {
                        "run_name": comparison.run_name,
                        "replicate": comparison.replicate,
                        "architecture": comparison.architecture,
                        "cohort": comparison.cohort,
                        "engine": comparison.engine,
                        "plot_family": family_name,
                        "status": "written",
                        "file_path": str(path),
                    }
                )

        if run_cfg.roundtrip_plots.sample_group_file:
            metrics_df = _group_metrics_table(
                comparison=comparison,
                run_cfg=run_cfg,
                run_dir=run_dir,
                sample_group_file=Path(run_cfg.roundtrip_plots.sample_group_file),
                id_column=run_cfg.roundtrip_plots.id_column,
                group_column=run_cfg.roundtrip_plots.group_column,
            )
            metrics_path = out_dir / "roundtrip_metrics_by_group.tsv"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(metrics_path, sep="\t", index=False)
            plot_rows.append(
                {
                    "run_name": comparison.run_name,
                    "replicate": comparison.replicate,
                    "architecture": comparison.architecture,
                    "cohort": comparison.cohort,
                    "engine": comparison.engine,
                    "plot_family": "metrics_by_group",
                    "status": "written",
                    "file_path": str(metrics_path),
                }
            )
            group_paths = _write_group_beta_scatter(
                comparison,
                metrics_df,
                run_cfg=run_cfg,
                run_dir=run_dir,
                sample_group_file=Path(run_cfg.roundtrip_plots.sample_group_file),
                id_column=run_cfg.roundtrip_plots.id_column,
                group_column=run_cfg.roundtrip_plots.group_column,
                output_base=out_dir / "roundtrip_beta_scatter_by_group",
                formats=formats,
                dpi=dpi,
            )
            if not group_paths:
                plot_rows.append(
                    {
                        "run_name": comparison.run_name,
                        "replicate": comparison.replicate,
                        "architecture": comparison.architecture,
                        "cohort": comparison.cohort,
                        "engine": comparison.engine,
                        "plot_family": "beta_scatter_by_group",
                        "status": "skipped",
                        "file_path": "",
                    }
                )
            else:
                for path in group_paths:
                    plot_rows.append(
                        {
                            "run_name": comparison.run_name,
                            "replicate": comparison.replicate,
                            "architecture": comparison.architecture,
                            "cohort": comparison.cohort,
                            "engine": comparison.engine,
                            "plot_family": "beta_scatter_by_group",
                            "status": "written",
                            "file_path": str(path),
                        }
                    )

    manifest_path = run_dir / "roundtrip_plot_manifest.tsv"
    pd.DataFrame(plot_rows).to_csv(manifest_path, sep="\t", index=False)
    logger.info("[roundtrip-plots] wrote plot manifest: %s", manifest_path)
    return manifest_path
