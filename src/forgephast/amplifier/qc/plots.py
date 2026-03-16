"""Matplotlib helpers for QC diagnostics.

All matplotlib imports are intentionally inside functions so that QC can run in
minimal environments when plotting is disabled.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_hist_overlay_png(
    output_png: str | Path,
    *,
    series: dict[str, np.ndarray],
    bins: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str = "Density",
) -> Path:
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    for label, values in series.items():
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        ax.hist(
            x,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=str(label),
        )

    ax.set_title(str(title))
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(str(ylabel))
    ax.legend(loc="best", frameon=False)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_maf_comparison_png(
    output_png: str | Path,
    *,
    maf_hists: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    title: str = "MAF spectrum",
) -> Path:
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    pops = sorted(maf_hists.keys())
    if not pops:
        raise ValueError("maf_hists cannot be empty")

    n = len(pops)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))

    plt = _pyplot()
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, pop in enumerate(pops):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        ax.set_title(str(pop))
        for source, (x, y) in maf_hists[pop].items():
            ax.plot(x, y, linewidth=2.0, label=str(source))
        ax.set_xlabel("MAF")
        ax.set_ylabel("Probability")
        ax.set_xlim(0.0, 0.5)
        ax.legend(loc="best", frameon=False)

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(str(title))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_pca_scatter_png(
    output_png: str | Path,
    *,
    donor_scores: np.ndarray,
    donor_labels: Iterable[str],
    synthetic_scores: np.ndarray,
    synthetic_labels: Iterable[str],
    title: str,
) -> Path:
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    donor = np.asarray(donor_scores, dtype=np.float64)
    synth = np.asarray(synthetic_scores, dtype=np.float64)
    if donor.ndim != 2 or synth.ndim != 2 or donor.shape[1] < 2 or synth.shape[1] < 2:
        raise ValueError("scores must be 2D with at least 2 components")

    donor_labels_list = [str(x) for x in donor_labels]
    synth_labels_list = [str(x) for x in synthetic_labels]
    if len(donor_labels_list) != donor.shape[0]:
        raise ValueError("donor_labels length mismatch")
    if len(synth_labels_list) != synth.shape[0]:
        raise ValueError("synthetic_labels length mismatch")

    pops = sorted(set(donor_labels_list) | set(synth_labels_list))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    color_by_pop = {pop: cmap(i % 10) for i, pop in enumerate(pops)}

    def _blend_with_white(rgba, *, blend: float) -> tuple[float, float, float, float]:
        r, g, b, a = rgba
        t = float(blend)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return (r + (1.0 - r) * t, g + (1.0 - g) * t, b + (1.0 - b) * t, a)

    synth_size = 8.0
    donor_size = synth_size * 3.0

    # Draw synthetic first so it sits behind donor points. Synthetic gets slightly
    # desaturated coloring and moderate alpha for readability in dense regions.
    for pop in pops:
        mask_s = np.array([x == pop for x in synth_labels_list], dtype=bool)
        if not np.any(mask_s):
            continue
        synthetic_color = _blend_with_white(color_by_pop[pop], blend=0.25)
        ax.scatter(
            synth[mask_s, 0],
            synth[mask_s, 1],
            s=synth_size,
            c=[synthetic_color],
            marker="o",
            alpha=0.75,
            linewidths=0.0,
            zorder=1,
            label=f"{pop} synthetic",
        )

    for pop in pops:
        mask_d = np.array([x == pop for x in donor_labels_list], dtype=bool)
        if not np.any(mask_d):
            continue
        ax.scatter(
            donor[mask_d, 0],
            donor[mask_d, 1],
            s=donor_size,
            c=[color_by_pop[pop]],
            marker="o",
            alpha=1.0,
            linewidths=0.30,
            edgecolors="black",
            zorder=3,
            label=f"{pop} donor",
        )

    ax.set_title(str(title))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_tract_length_ks_png(
    output_png: str | Path,
    *,
    rate: float,
    title: str,
    xlabel: str,
    model: str = "exponential",
    chrom_length: float | None = None,
    values: np.ndarray | None = None,
    hist_counts: np.ndarray | None = None,
    hist_bin_edges: np.ndarray | None = None,
) -> Path:
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not np.isfinite(rate) or float(rate) <= 0.0:
        raise ValueError("rate must be finite and > 0")

    model_name = str(model).strip().lower()
    l_chr: float | None = None
    if model_name == "finite_chr_poisson_switch":
        if chrom_length is None or not np.isfinite(float(chrom_length)) or float(chrom_length) <= 0.0:
            raise ValueError("chrom_length must be finite and > 0 for model='finite_chr_poisson_switch'")
        l_chr = float(chrom_length)

    x_ecdf: np.ndarray
    y_ecdf: np.ndarray
    hist_edges: np.ndarray
    hist_density: np.ndarray

    if values is not None:
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x) & (x >= 0.0)]
        if x.size == 0:
            raise ValueError("values must include at least one finite non-negative element")
        x_sorted = np.sort(x)
        x_ecdf = x_sorted
        y_ecdf = np.arange(1, x_sorted.size + 1, dtype=np.float64) / float(x_sorted.size)
        n_bins = max(16, min(128, int(np.sqrt(float(x_sorted.size)) * 2.0)))
        max_x = float(np.max(x_sorted))
        hist_max = float(l_chr) if l_chr is not None else max_x
        hist_edges = np.linspace(0.0, max(hist_max, 1e-6), n_bins + 1, dtype=np.float64)
        hist_counts_arr, _ = np.histogram(x_sorted, bins=hist_edges)
        widths = np.diff(hist_edges)
        n_total = float(np.sum(hist_counts_arr))
        hist_density = (hist_counts_arr.astype(np.float64) / np.maximum(widths * max(n_total, 1.0), 1e-12)).astype(
            np.float64
        )
    else:
        counts = np.asarray(hist_counts if hist_counts is not None else [], dtype=np.float64)
        edges = np.asarray(hist_bin_edges if hist_bin_edges is not None else [], dtype=np.float64)
        if counts.size == 0 or edges.size != (counts.size + 1):
            raise ValueError("hist_counts/hist_bin_edges must be provided when values is None")
        if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(edges)):
            raise ValueError("hist_counts/hist_bin_edges must be finite")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("hist_bin_edges must be strictly increasing")
        n_total = float(np.sum(counts))
        if n_total <= 0.0:
            raise ValueError("hist_counts sum must be > 0")
        hist_edges = edges.astype(np.float64, copy=False)
        widths = np.diff(hist_edges)
        hist_density = counts / np.maximum(widths * n_total, 1e-12)
        x_ecdf = hist_edges[1:]
        y_ecdf = np.cumsum(counts) / n_total

    x_max = float(max(np.max(hist_edges), np.max(x_ecdf)))
    if model_name == "finite_chr_poisson_switch":
        assert l_chr is not None
        x_grid = np.linspace(0.0, max(l_chr, 1e-6), 400, dtype=np.float64)
        m = float(rate) * l_chr
        p_atom = float(np.exp(-m) / (1.0 + m))
        u = np.clip(x_grid / l_chr, 0.0, 1.0)
        tail = np.exp(-m * u) * (1.0 + m * (1.0 - u)) / (1.0 + m)
        exp_cdf = 1.0 - tail
        exp_cdf = np.where(x_grid >= l_chr, 1.0, exp_cdf)
        exp_cdf = np.where(x_grid <= 0.0, 0.0, exp_cdf)
        # Derivative of finite-window CDF for visual density overlay.
        exp_pdf = np.exp(-float(rate) * x_grid) * float(rate) * (2.0 + m - (float(rate) * x_grid)) / (1.0 + m)
        exp_pdf = np.where(x_grid > l_chr, 0.0, exp_pdf)
        model_label = "Expected finite-window pulse null"
    else:
        x_grid = np.linspace(0.0, max(x_max, 1e-6), 400, dtype=np.float64)
        exp_pdf = float(rate) * np.exp(-float(rate) * x_grid)
        exp_cdf = 1.0 - np.exp(-float(rate) * x_grid)
        model_label = "Expected exponential"

    plt = _pyplot()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), constrained_layout=True)
    ax_h, ax_c = axes

    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    widths = np.diff(hist_edges)
    ax_h.bar(
        centers,
        hist_density,
        width=widths,
        align="center",
        alpha=0.50,
        color="#4C78A8",
        edgecolor="none",
        label="Observed",
    )
    ax_h.plot(x_grid, exp_pdf, color="#D62728", linewidth=2.0, label=model_label)
    ax_h.set_title("Tract-length density")
    ax_h.set_xlabel(str(xlabel))
    ax_h.set_ylabel("Density")
    ax_h.legend(loc="best", frameon=False)

    if model_name == "finite_chr_poisson_switch":
        assert l_chr is not None
        obs_atom: float | None = None
        if values is not None:
            tol = max(1e-9, 1e-9 * l_chr)
            obs_atom = float(np.mean(x_sorted >= (l_chr - tol)))
        atom_note = f"P(full-chr tract at L={l_chr:.3g} cM): exp={p_atom:.3g}"
        if obs_atom is not None:
            atom_note += f", obs={obs_atom:.3g}"
        atom_note += " (point mass; not shown in PDF line)"
        ax_h.axvline(l_chr, color="#D62728", linestyle="--", linewidth=1.0, alpha=0.65)
        ax_h.text(
            0.02,
            0.98,
            atom_note,
            transform=ax_h.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

    ax_c.step(x_ecdf, y_ecdf, where="post", color="#4C78A8", linewidth=2.0, label="Observed ECDF")
    ax_c.plot(x_grid, exp_cdf, color="#D62728", linewidth=2.0, label=model_label)
    ax_c.set_title("CDF fit")
    ax_c.set_xlabel(str(xlabel))
    ax_c.set_ylabel("Cumulative probability")
    ax_c.set_ylim(0.0, 1.0)
    ax_c.legend(loc="best", frameon=False)

    fig.suptitle(str(title))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_ld_heatmap_pair_png(
    output_png: str | Path,
    *,
    donor_mean_r2: np.ndarray,
    synthetic_mean_r2: np.ndarray,
    edges_cm: np.ndarray,
    title: str,
) -> Path:
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    donor = np.asarray(donor_mean_r2, dtype=np.float64)
    synth = np.asarray(synthetic_mean_r2, dtype=np.float64)
    edges = np.asarray(edges_cm, dtype=np.float64)

    if donor.ndim != 2 or synth.ndim != 2 or donor.shape != synth.shape or donor.shape[0] < 5:
        raise ValueError("heatmap arrays must be 2D with matching shapes and >=5 bins")
    if edges.ndim != 1 or edges.shape[0] != donor.shape[0] + 1:
        raise ValueError("edges_cm must have shape (n_bins+1,)")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("edges_cm must be finite and strictly increasing")

    vals = np.concatenate([donor[np.isfinite(donor)], synth[np.isfinite(synth)]])
    if vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(vals, 99.0))
        vmax = max(0.10, min(1.0, vmax))

    plt = _pyplot()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    extent = (float(edges[0]), float(edges[-1]), float(edges[0]), float(edges[-1]))
    im0 = axes[0].imshow(
        np.ma.masked_invalid(donor),
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )
    axes[0].set_title("Donor")

    axes[1].imshow(
        np.ma.masked_invalid(synth),
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )
    axes[1].set_title("Synthetic")

    for ax in axes:
        ax.set_xlabel("cM")
        ax.set_ylabel("cM")

    fig.suptitle(str(title))
    fig.colorbar(im0, ax=list(axes), shrink=0.90, label="mean $r^2$")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def save_roh_total_cm_png(
    output_png: str | Path,
    *,
    pop_to_series: dict[str, dict[str, np.ndarray]],
    title: str = "Total ROH (cM)",
) -> Path:
    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    pops = sorted(pop_to_series.keys())
    if not pops:
        raise ValueError("pop_to_series cannot be empty")

    n = len(pops)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))

    plt = _pyplot()
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, pop in enumerate(pops):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        ax.set_title(str(pop))
        for source, values in pop_to_series[pop].items():
            x = np.asarray(values, dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            ax.hist(
                x,
                bins=40,
                density=True,
                alpha=0.5,
                label=str(source),
            )
        ax.set_xlabel("Total ROH (cM)")
        ax.set_ylabel("Density")
        ax.legend(loc="best", frameon=False)

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(str(title))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
