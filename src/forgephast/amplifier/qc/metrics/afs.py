"""Allele frequency spectrum QC (donor vs synthetic)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forgephast._internal.qc_kernels import maf_from_haplotypes as _utils_maf_from_haplotypes

from ...config import RunConfig
from .statistics import histogram_probabilities as _hist_probs

logger = logging.getLogger(__name__)

_MISSING_SENTINEL = np.uint8(255)


def _maf_from_haplotypes(haps: np.ndarray) -> np.ndarray:
    maf = _utils_maf_from_haplotypes(haps, missing_sentinel=int(_MISSING_SENTINEL), min_an=4)
    maf = maf[np.isfinite(maf)]
    return np.clip(maf.astype(np.float64, copy=False), 0.0, 0.5)


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = (p > 0.0) & np.isfinite(p) & np.isfinite(q) & (q > 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)


def compute_afs_metrics(
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
    """Compute donor-vs-synthetic MAF spectrum agreement."""
    from ..plots import save_maf_comparison_png

    enabled = bool(config.qc.afs_enabled)
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
        payload["population_summary"] = {}
        return payload, created

    pop_names = list(getattr(ext_data, "pops", []))
    if not pop_names:
        payload["status"] = "no_populations"
        payload["population_summary"] = {}
        return payload, created

    subset_id = str(getattr(ext_data, "variant_subset_id", ""))

    edges = np.linspace(0.0, 0.5, int(config.qc.afs_n_bins) + 1, dtype=np.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    plot_hists: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for pop in pop_names:
        pop_data = ext_data.per_pop[pop]
        donor_ids = pop_data.donor_ids
        synth_ids = pop_data.synth_ids
        donor_haps = pop_data.donor_haps
        synth_haps = pop_data.synth_haps

        maf_d = _maf_from_haplotypes(donor_haps)
        maf_s = _maf_from_haplotypes(synth_haps)
        probs_d, counts_d = _hist_probs(maf_d, edges=edges)
        probs_s, counts_s = _hist_probs(maf_s, edges=edges)

        if float(np.sum(counts_d)) <= 0 or float(np.sum(counts_s)) <= 0:
            summary[pop] = {
                "status": "no_variants",
                "n_donor_samples": int(len(donor_ids)),
                "n_synth_samples": int(len(synth_ids)),
                "n_variants_used": 0,
            }
            continue

        tv = 0.5 * float(np.sum(np.abs(probs_s - probs_d)))
        js = float(_js_divergence(probs_s, probs_d))
        ks = float(np.max(np.abs(np.cumsum(probs_s) - np.cumsum(probs_d))))

        summary[pop] = {
            "status": "ok",
            "n_donor_samples": int(len(donor_ids)),
            "n_synth_samples": int(len(synth_ids)),
            "n_variants_used": int(min(int(np.sum(counts_d)), int(np.sum(counts_s)))),
            "tv": tv,
            "js": js,
            "ks": ks,
        }

        plot_hists[pop] = {
            "donor": (mids, probs_d),
            "synthetic": (mids, probs_s),
        }

        for source, probs, counts in (
            ("donor", probs_d, counts_d),
            ("synthetic", probs_s, counts_s),
        ):
            for i in range(edges.size - 1):
                rows.append(
                    {
                        "population": pop,
                        "source": source,
                        "chrom": int(chrom),
                        "bin_start": float(edges[i]),
                        "bin_end": float(edges[i + 1]),
                        "bin_mid": float(mids[i]),
                        "prob": float(probs[i]),
                        "count": int(counts[i]),
                    }
                )

    table_path = diagnostics_dir / f"afs_hist_chr{chrom}.tsv"
    df = pd.DataFrame.from_records(rows)
    df.to_csv(table_path, sep="\t", index=False)
    created.append(table_path)

    plot_path: Path | None = None
    plot_error: str | None = None
    if config.qc.plots_enabled and plot_hists:
        try:
            plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"afs_maf_chr{chrom}.png"
            save_maf_comparison_png(plot_path, maf_hists=plot_hists)
            created.append(plot_path)
        except Exception as exc:
            logger.warning("AFS QC plot generation failed; continuing without plot: %s", exc)
            plot_error = str(exc)
            plot_path = None

    payload.update(
        {
            "status": "ok",
            "seed": int(config.qc.seed),
            "ext_max_diploids_per_population": int(config.qc.ext_max_diploids_per_population),
            "ext_max_variants": int(config.qc.ext_max_variants),
            "variant_subset_id": subset_id,
            "afs_n_bins": int(config.qc.afs_n_bins),
            "maf_range": [0.0, 0.5],
            "table_path": str(table_path),
            "plot_path": str(plot_path) if plot_path is not None else None,
            "plot_error": plot_error,
            "population_summary": summary,
        }
    )
    logger.info(
        "AFS QC complete chrom=%s pops=%s table=%s plot=%s",
        chrom,
        len(pop_names),
        table_path,
        plot_path,
    )
    return payload, created
