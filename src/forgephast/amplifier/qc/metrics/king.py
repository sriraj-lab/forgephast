"""KING-robust kinship QC (donor vs synthetic)."""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...config import RunConfig
from ...utils import derived_seed

from forgephast._internal.qc_kernels import haplotypes_to_genotypes_u8
from forgephast._internal.qc_kernels import king_cross_pairs as _utils_king_cross_pairs
from forgephast._internal.qc_kernels import king_within_pairs as _utils_king_within_pairs
from .statistics import auc_from_scores as _auc_from_scores

logger = logging.getLogger(__name__)

_MISSING_SENTINEL = np.uint8(255)

def _summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    n_total = int(x.size)
    x = x[np.isfinite(x)]
    n_finite = int(x.size)
    n_nan = n_total - n_finite
    if n_finite == 0:
        return {
            "n_pairs": n_total,
            "n_nan": n_nan,
            "mean": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n_pairs": n_total,
        "n_nan": n_nan,
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _hist(values: np.ndarray, *, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    counts, _ = np.histogram(x, bins=edges)
    counts = counts.astype(np.int64, copy=False)
    total = float(np.sum(counts))
    widths = np.diff(edges.astype(np.float64))
    if total <= 0:
        density = np.zeros_like(widths, dtype=np.float64)
    else:
        density = (counts.astype(np.float64) / total) / widths
    return counts, density


def _sample_within_pairs(
    *,
    n_samples: int,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(n_samples)
    k = int(max_pairs)
    if n < 2 or k <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    pairs: set[tuple[int, int]] = set()
    target = min(k, (n * (n - 1)) // 2)

    batch = max(256, target * 4)
    # Deterministic loop: bounded attempts with deterministic RNG.
    attempts = 0
    max_attempts = max(1000, target * 50)
    while len(pairs) < target and attempts < max_attempts:
        attempts += 1
        a = rng.integers(0, n, size=batch, dtype=np.int64)
        b = rng.integers(0, n, size=batch, dtype=np.int64)
        mask = a != b
        a = a[mask]
        b = b[mask]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for i, j in zip(lo.tolist(), hi.tolist()):
            pairs.add((int(i), int(j)))
            if len(pairs) >= target:
                break

    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    ordered = sorted(pairs)
    out_i = np.fromiter((p[0] for p in ordered), dtype=np.int64, count=len(ordered))
    out_j = np.fromiter((p[1] for p in ordered), dtype=np.int64, count=len(ordered))
    return out_i, out_j


def _sample_cross_pairs(
    *,
    n_a: int,
    n_b: int,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    na = int(n_a)
    nb = int(n_b)
    k = int(max_pairs)
    if na <= 0 or nb <= 0 or k <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    total = na * nb
    target = min(k, total)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(total, size=target, replace=False).astype(np.int64, copy=False)
    out_a = idx // nb
    out_b = idx % nb
    return out_a, out_b


def _king_within_pairs(
    genotypes: np.ndarray,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    *,
    numba_threads: int | None,
) -> np.ndarray:
    return _utils_king_within_pairs(
        genotypes,
        pairs_i,
        pairs_j,
        missing_sentinel=int(_MISSING_SENTINEL),
        numba_threads=numba_threads,
    )


def _king_cross_pairs(
    genotypes_a: np.ndarray,
    genotypes_b: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    numba_threads: int | None,
) -> np.ndarray:
    return _utils_king_cross_pairs(
        genotypes_a,
        genotypes_b,
        idx_a,
        idx_b,
        missing_sentinel=int(_MISSING_SENTINEL),
        numba_threads=numba_threads,
    )


def compute_king_kinship_metrics(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int | None = None,
    ext_data: Any | None = None,
    ext_data_error: str | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Compute KING-robust kinship distributions for donor vs synthetic cohorts."""
    from ..plots import save_hist_overlay_png

    enabled = bool(config.qc.king_enabled)
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
        return payload, created

    pops = list(getattr(ext_data, "pops", []))
    if not pops:
        payload["status"] = "no_populations"
        return payload, created

    sampled_pos = np.asarray(getattr(ext_data, "sampled_pos"), dtype=np.int64)
    global_pos = np.asarray(getattr(ext_data, "global_pos"), dtype=np.int64)
    subset_id = str(getattr(ext_data, "variant_subset_id", ""))

    donor_g: dict[str, np.ndarray] = {}
    synth_g: dict[str, np.ndarray] = {}
    for pop in pops:
        pop_data = ext_data.per_pop[pop]
        donor_g[pop] = haplotypes_to_genotypes_u8(pop_data.donor_haps)
        synth_g[pop] = haplotypes_to_genotypes_u8(pop_data.synth_haps)

    within_donor: dict[str, np.ndarray] = {}
    within_synth: dict[str, np.ndarray] = {}
    cross_synth: dict[str, np.ndarray] = {}

    for pop in pops:
        g_d = donor_g[pop]
        g_s = synth_g[pop]

        pairs_i, pairs_j = _sample_within_pairs(
            n_samples=int(g_d.shape[0]),
            max_pairs=int(config.qc.king_max_within_pairs_per_population),
            seed=int(derived_seed(int(config.qc.seed), "king_pairs_within_donor", pop, chrom)),
        )
        within_donor[pop] = _king_within_pairs(g_d, pairs_i, pairs_j, numba_threads=numba_threads)

        pairs_i, pairs_j = _sample_within_pairs(
            n_samples=int(g_s.shape[0]),
            max_pairs=int(config.qc.king_max_within_pairs_per_population),
            seed=int(derived_seed(int(config.qc.seed), "king_pairs_within_synth", pop, chrom)),
        )
        within_synth[pop] = _king_within_pairs(g_s, pairs_i, pairs_j, numba_threads=numba_threads)

    for pop_a, pop_b in combinations(pops, 2):
        ga = synth_g[pop_a]
        gb = synth_g[pop_b]
        idx_a, idx_b = _sample_cross_pairs(
            n_a=int(ga.shape[0]),
            n_b=int(gb.shape[0]),
            max_pairs=int(config.qc.king_max_cross_pairs_per_pop_pair),
            seed=int(derived_seed(int(config.qc.seed), "king_pairs_cross_synth", pop_a, pop_b, chrom)),
        )
        key = f"{pop_a}__{pop_b}"
        cross_synth[key] = _king_cross_pairs(ga, gb, idx_a, idx_b, numba_threads=numba_threads)

    pooled_within_donor = np.concatenate([within_donor[p] for p in sorted(within_donor.keys())]) if within_donor else np.zeros(0, dtype=np.float64)
    pooled_within_synth = np.concatenate([within_synth[p] for p in sorted(within_synth.keys())]) if within_synth else np.zeros(0, dtype=np.float64)
    pooled_cross_synth = np.concatenate([cross_synth[k] for k in sorted(cross_synth.keys())]) if cross_synth else np.zeros(0, dtype=np.float64)

    auc = _auc_from_scores(pooled_within_synth, pooled_cross_synth)
    auc = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")

    edges = np.linspace(
        float(config.qc.king_hist_min),
        float(config.qc.king_hist_max),
        int(config.qc.king_hist_n_bins) + 1,
        dtype=np.float64,
    )
    rows_hist: list[dict[str, Any]] = []
    for bucket, values in (
        ("within_donor", pooled_within_donor),
        ("within_synthetic", pooled_within_synth),
        ("cross_synthetic", pooled_cross_synth),
    ):
        counts, density = _hist(values, edges=edges)
        for i in range(edges.size - 1):
            rows_hist.append(
                {
                    "bucket": bucket,
                    "chrom": int(chrom),
                    "bin_start": float(edges[i]),
                    "bin_end": float(edges[i + 1]),
                    "count": int(counts[i]),
                    "density": float(density[i]),
                }
            )

    hist_path = diagnostics_dir / f"king_hist_chr{chrom}.tsv"
    pd.DataFrame.from_records(rows_hist).to_csv(hist_path, sep="\t", index=False)
    created.append(hist_path)

    rows_summary: list[dict[str, Any]] = []

    for pop in sorted(within_donor.keys()):
        rows_summary.append(
            {
                "bucket": "within_donor",
                "population": pop,
                "pop_pair": "",
                "chrom": int(chrom),
                **_summarize(within_donor[pop]),
                "auc_within_vs_cross_synthetic": float("nan"),
            }
        )
    for pop in sorted(within_synth.keys()):
        rows_summary.append(
            {
                "bucket": "within_synthetic",
                "population": pop,
                "pop_pair": "",
                "chrom": int(chrom),
                **_summarize(within_synth[pop]),
                "auc_within_vs_cross_synthetic": float("nan"),
            }
        )
    for pair in sorted(cross_synth.keys()):
        rows_summary.append(
            {
                "bucket": "cross_synthetic",
                "population": "",
                "pop_pair": pair,
                "chrom": int(chrom),
                **_summarize(cross_synth[pair]),
                "auc_within_vs_cross_synthetic": float("nan"),
            }
        )

    # Pooled rows (ALL) + separation AUC.
    rows_summary.append(
        {
            "bucket": "within_donor",
            "population": "ALL",
            "pop_pair": "",
            "chrom": int(chrom),
            **_summarize(pooled_within_donor),
            "auc_within_vs_cross_synthetic": float("nan"),
        }
    )
    rows_summary.append(
        {
            "bucket": "within_synthetic",
            "population": "ALL",
            "pop_pair": "",
            "chrom": int(chrom),
            **_summarize(pooled_within_synth),
            "auc_within_vs_cross_synthetic": float(auc),
        }
    )
    rows_summary.append(
        {
            "bucket": "cross_synthetic",
            "population": "ALL",
            "pop_pair": "",
            "chrom": int(chrom),
            **_summarize(pooled_cross_synth),
            "auc_within_vs_cross_synthetic": float("nan"),
        }
    )

    summary_path = diagnostics_dir / f"king_summary_chr{chrom}.tsv"
    pd.DataFrame.from_records(rows_summary).to_csv(summary_path, sep="\t", index=False)
    created.append(summary_path)

    plot_path: Path | None = None
    plot_error: str | None = None
    if config.qc.plots_enabled:
        try:
            plot_dir = diagnostics_dir / str(config.qc.plots_dirname)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"king_kinship_chr{chrom}.png"
            save_hist_overlay_png(
                plot_path,
                series={
                    "within_donor": pooled_within_donor,
                    "within_synthetic": pooled_within_synth,
                    "cross_synthetic": pooled_cross_synth,
                },
                bins=edges,
                title=f"KING kinship chr{chrom}",
                xlabel="Kinship",
            )
            created.append(plot_path)
        except Exception as exc:
            logger.warning("KING QC plot generation failed; continuing without plot: %s", exc)
            plot_error = str(exc)
            plot_path = None

    payload.update(
        {
            "status": "ok",
            "seed": int(config.qc.seed),
            "ext_max_diploids_per_population": int(config.qc.ext_max_diploids_per_population),
            "ext_max_variants": int(config.qc.ext_max_variants),
            "variant_subset_id": subset_id,
            "n_positions_global_common": int(global_pos.size),
            "n_positions_sampled": int(sampled_pos.size),
            "auc_within_vs_cross_synthetic": float(auc),
            "summary_path": str(summary_path),
            "hist_path": str(hist_path),
            "plot_path": str(plot_path) if plot_path is not None else None,
            "plot_error": plot_error,
        }
    )

    logger.info(
        "KING QC complete chrom=%s pops=%s positions=%s within_pairs=%s cross_pairs=%s auc=%.4f summary=%s plot=%s",
        chrom,
        len(pops),
        int(sampled_pos.size),
        int(pooled_within_synth.size),
        int(pooled_cross_synth.size),
        float(auc) if np.isfinite(auc) else float("nan"),
        summary_path,
        plot_path,
    )
    return payload, created
