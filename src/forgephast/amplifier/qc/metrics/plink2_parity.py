"""HPC-only PLINK2 parity harness for QC smoke tests.

This module is designed to auto-skip when PLINK2 isn't available (e.g. local
development). It runs a small, deterministic subset of samples/variants and
compares PLINK2 outputs against python implementations for sanity checks.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from ...config import RunConfig
from ..io import haplotypes_to_genotypes_f32, haplotypes_to_genotypes_u8
from ...utils import derived_seed

logger = logging.getLogger(__name__)

_MISSING_SENTINEL = np.uint8(255)


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _read_plink2_version(plink2_bin: str) -> str:
    try:
        proc = subprocess.run([plink2_bin, "--version"], capture_output=True, text=True)
    except Exception:
        return ""
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return out if out else err


def _write_keep_file(path: Path, sample_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for sid in sample_ids:
            sid_s = str(sid)
            handle.write(f"{sid_s}\t{sid_s}\n")


def _dedupe_preserve_order(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sid in ids:
        s = str(sid)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_positions_from_ids(ids: list[str]) -> np.ndarray:
    pos: list[int] = []
    for vid in ids:
        m = re.search(r":(\d+)", str(vid))
        if not m:
            continue
        pos.append(int(m.group(1)))
    if not pos:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.asarray(pos, dtype=np.int64))


def _maf_hist_from_haps(haps: np.ndarray, *, edges: np.ndarray) -> np.ndarray:
    h = np.asarray(haps, dtype=np.uint8)
    if h.ndim != 2:
        raise ValueError(f"haplotypes must be 2D, got shape={h.shape}")
    ac = np.sum(h == 1, axis=0, dtype=np.float64)
    an = np.sum(h != _MISSING_SENTINEL, axis=0, dtype=np.float64)
    valid = (an >= 4.0) & np.isfinite(an)
    if not np.any(valid):
        return np.zeros(edges.size - 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = ac[valid] / an[valid]
    maf = np.minimum(p, 1.0 - p)
    maf = maf[np.isfinite(maf)]
    if maf.size == 0:
        return np.zeros(edges.size - 1, dtype=np.float64)
    counts, _ = np.histogram(np.clip(maf, 0.0, 0.5), bins=edges)
    total = float(np.sum(counts))
    return counts.astype(np.float64) / total if total > 0 else np.zeros(edges.size - 1, dtype=np.float64)


def _maf_hist_from_acount(ac: np.ndarray, an: np.ndarray, *, edges: np.ndarray) -> np.ndarray:
    ac = np.asarray(ac, dtype=np.float64)
    an = np.asarray(an, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = ac / an
    maf = np.minimum(p, 1.0 - p)
    maf = maf[np.isfinite(maf) & (an >= 4.0)]
    if maf.size == 0:
        return np.zeros(edges.size - 1, dtype=np.float64)
    counts, _ = np.histogram(np.clip(maf, 0.0, 0.5), bins=edges)
    total = float(np.sum(counts))
    return counts.astype(np.float64) / total if total > 0 else np.zeros(edges.size - 1, dtype=np.float64)


def _auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Compute ROC AUC without external deps (Mann–Whitney U with tie handling)."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    scores = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate(
        [np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)],
        axis=0,
    )

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    n = sorted_scores.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # 1-based ranks
        ranks[i:j] = avg_rank
        i = j

    pos_rank_sum = float(np.sum(ranks[sorted_labels == 1]))
    u = pos_rank_sum - (pos.size * (pos.size + 1)) / 2.0
    auc = u / float(pos.size * neg.size)
    return float(np.clip(auc, 0.0, 1.0))


def _standardize_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x - mean) / std, mean, std


def _impute_center_scale(g: np.ndarray, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.asarray(g, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"genotypes must be 2D, got shape={x.shape}")
    out = x.copy()
    missing = ~np.isfinite(out)
    if np.any(missing):
        col_idx = np.where(missing)[1]
        out[missing] = mean[col_idx]
    out -= mean
    out /= scale
    return out


def _min_n_for_pairs(target_pairs: int) -> int:
    k = int(target_pairs)
    if k <= 0:
        return 0
    n = 2
    while (n * (n - 1)) // 2 < k:
        n += 1
    return n


def _pick_iid_column(df: pd.DataFrame) -> str | None:
    for candidate in ("IID", "ID"):
        if candidate in df.columns:
            return candidate
    if df.columns.size >= 2:
        return str(df.columns[1])
    return None


def _pick_kinship_columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    cols = [str(c) for c in df.columns]
    upper = {c.upper(): c for c in cols}
    kin_col = upper.get("KINSHIP")
    if kin_col is None:
        return None

    for id1, id2 in (("IID1", "IID2"), ("ID1", "ID2")):
        if id1 in cols and id2 in cols:
            return id1, id2, kin_col
        if f"#{id1}" in cols and f"#{id2}" in cols:
            return f"#{id1}", f"#{id2}", kin_col

    cand1 = [c for c in cols if re.match(r"^(#?IID1|#?ID1)$", c)]
    cand2 = [c for c in cols if re.match(r"^(#?IID2|#?ID2)$", c)]
    if cand1 and cand2:
        return cand1[0], cand2[0], kin_col
    return None


def _filter_pca_weights_by_ref_maf(
    *,
    acount_path: Path,
    weights_path: Path,
    out_path: Path,
    maf_min: float,
) -> dict[str, Any]:
    """Filter PLINK2 `.eigenvec.allele` rows to variants polymorphic in the reference.

    PLINK2 projection via `--score ... variance-standardize` can hard-fail when the
    reference allele frequency for a scored allele is 0/NaN but dosages are
    nonzero in the new dataset. This helper conservatively filters to variants
    with finite MAF in (maf_min, 0.5].
    """
    df_freq = pd.read_csv(acount_path, sep=r"\s+", engine="python")
    if "ID" not in df_freq.columns:
        raise ValueError(f"plink2 .acount missing ID column: {acount_path}")

    if "ALT_CTS" in df_freq.columns:
        ac_s = df_freq["ALT_CTS"].astype(str).str.split(",", n=1).str[0]
        ac = pd.to_numeric(ac_s, errors="coerce").to_numpy(dtype=np.float64)
    elif "ALT_CT" in df_freq.columns:
        ac = pd.to_numeric(df_freq["ALT_CT"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"plink2 .acount missing ALT count column (ALT_CTS/ALT_CT): {acount_path}")

    if "OBS_CT" in df_freq.columns:
        an = pd.to_numeric(df_freq["OBS_CT"], errors="coerce").to_numpy(dtype=np.float64)
    elif "NCHROBS" in df_freq.columns:
        an = pd.to_numeric(df_freq["NCHROBS"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"plink2 .acount missing allele-count denominator (OBS_CT/NCHROBS): {acount_path}")

    with np.errstate(divide="ignore", invalid="ignore"):
        p = ac / an
    maf = np.minimum(p, 1.0 - p)
    valid = (
        np.isfinite(maf)
        & np.isfinite(an)
        & (an >= 4.0)
        & (maf >= float(maf_min))
        & (maf <= 0.5)
    )
    good_ids = set(df_freq.loc[valid, "ID"].astype(str).tolist())

    df_w = pd.read_csv(weights_path, sep=r"\s+", engine="python")
    if df_w.empty:
        raise ValueError(f"empty PCA weights file: {weights_path}")

    id_col = "ID" if "ID" in df_w.columns else str(df_w.columns[1])
    df_w[id_col] = df_w[id_col].astype(str)
    df_f = df_w.loc[df_w[id_col].isin(good_ids)].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(out_path, sep="\t", index=False)

    return {
        "maf_min": float(maf_min),
        "n_variants_acount": int(df_freq.shape[0]),
        "n_variants_good": int(len(good_ids)),
        "n_variants_weights_in": int(df_w.shape[0]),
        "n_variants_weights_out": int(df_f.shape[0]),
        "weights_out_path": str(out_path),
    }


def _auc_by_pair_from_donor_direction(
    *,
    donor_scores_std: np.ndarray,
    donor_labels: list[str],
    synth_scores_std: np.ndarray,
    synth_labels: list[str],
    pops: list[str],
    k: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for pop1, pop2 in combinations(pops, 2):
        dm1 = np.array([x == pop1 for x in donor_labels], dtype=bool)
        dm2 = np.array([x == pop2 for x in donor_labels], dtype=bool)
        sm1 = np.array([x == pop1 for x in synth_labels], dtype=bool)
        sm2 = np.array([x == pop2 for x in synth_labels], dtype=bool)
        if not np.any(dm1) or not np.any(dm2) or not np.any(sm1) or not np.any(sm2):
            continue

        d1 = donor_scores_std[dm1, :k]
        d2 = donor_scores_std[dm2, :k]
        s1 = synth_scores_std[sm1, :k]
        s2 = synth_scores_std[sm2, :k]

        direction = np.mean(d1, axis=0) - np.mean(d2, axis=0)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:
            auc_d = float("nan")
            auc_s = float("nan")
        else:
            auc_d = _auc_from_scores(d1 @ direction, d2 @ direction)
            auc_s = _auc_from_scores(s1 @ direction, s2 @ direction)
            auc_d = float(max(auc_d, 1.0 - auc_d))
            auc_s = float(max(auc_s, 1.0 - auc_s))

        out[f"{pop1}__{pop2}"] = {
            "auc_donor": float(auc_d),
            "auc_synthetic": float(auc_s),
        }
    return out


def run_plink2_parity_smoke(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
) -> dict[str, Any]:
    """Run an optional PLINK2 parity harness (intended for HPC smoke tests)."""
    enabled = bool(config.qc.plink2_parity_enabled)
    out: dict[str, Any] = {
        "enabled": enabled,
        "chrom": int(chrom),
        "status": "disabled" if not enabled else "init",
    }
    if not enabled:
        return out

    if inputs.combined_vcf is None:
        out["status"] = "skipped_no_vcf"
        return out

    plink2_bin = str(config.outputs.plink2_path or "plink2")
    if shutil.which(plink2_bin) is None and not Path(plink2_bin).exists():
        out["status"] = "skipped_no_plink2"
        out["plink2_path"] = plink2_bin
        return out

    out["plink2_path"] = plink2_bin
    out["plink2_version"] = _read_plink2_version(plink2_bin)

    pop_names = sorted(groups.keys())
    if not pop_names:
        out["status"] = "no_populations"
        return out

    from ..pipeline import (  # local import avoids import cycles unless parity enabled
        _align_haplotype_axes_by_position,
        _load_plink_acount,
        _load_sample_haplotypes_from_panel,
        _load_synthetic_population_haplotypes,
        _resolve_panel_vcf_for_population,
        _subsample_ids,
    )
    from .king import (
        _king_cross_pairs,
        _king_within_pairs,
        _sample_cross_pairs,
        _sample_within_pairs,
    )

    synthetic_vcf = Path(inputs.combined_vcf)
    seed = int(derived_seed(int(config.qc.seed), "plink2_parity", chrom))
    max_vars = int(config.qc.ext_max_variants)
    max_pairs = int(config.qc.plink2_parity_max_pairs_per_population)
    n_pcs = int(config.qc.pca_n_components)
    threads = max(1, int(config.qc.plink_threads))

    try:
        donor_vcf_by_pop = {
            pop: _resolve_panel_vcf_for_population(config, chrom=chrom, pop=pop) for pop in pop_names
        }
    except Exception as exc:
        out["status"] = "skipped_no_donor_vcf"
        out["reason"] = str(exc)
        return out
    unique_donor_vcfs = sorted({str(p) for p in donor_vcf_by_pop.values()})
    donor_vcf_for_pca = Path(unique_donor_vcfs[0]) if unique_donor_vcfs else None
    pca_supported = bool(donor_vcf_for_pca is not None and len(unique_donor_vcfs) == 1)

    try:
        with tempfile.TemporaryDirectory(prefix=f"popamp_plink2_parity_chr{chrom}_", dir=str(config.run_root)) as tmp:
            tmp_dir = Path(tmp)

            # 1) Variant snplist for parity.
            vars_prefix = tmp_dir / "parity_vars"
            snplist_path = Path(f"{vars_prefix}.snplist")
            var_id_fmt = "chr@:#$r,$a"
            _run_cmd(
                [
                    plink2_bin,
                    "--vcf",
                    str(synthetic_vcf),
                    "--double-id",
                    "--allow-extra-chr",
                    "--max-alleles",
                    "2",
                    "--snps-only",
                    "just-acgt",
                    "--set-all-var-ids",
                    var_id_fmt,
                    "--thin-count",
                    str(max_vars),
                    "--seed",
                    str(seed),
                    "--write-snplist",
                    "--out",
                    str(vars_prefix),
                ]
            )
            if not snplist_path.exists():
                raise FileNotFoundError(f"plink2 parity snplist missing: {snplist_path}")
            with open(snplist_path, "r", encoding="utf-8") as handle:
                var_ids = [line.strip() for line in handle if line.strip()]
            parity_positions = _parse_positions_from_ids(var_ids)
            out["n_variants_snplist"] = int(len(var_ids))
            out["n_positions_parsed"] = int(parity_positions.size)

            edges_afs = np.linspace(0.0, 0.5, int(config.qc.afs_n_bins) + 1, dtype=np.float64)

            # 2) AFS parity.
            afs_parity: dict[str, Any] = {"status": "ok", "population_summary": {}}
            for pop in pop_names:
                donors_all = donors_by_population.get(pop, [])
                synth_all = groups.get(pop, [])
                if not donors_all or not synth_all:
                    afs_parity["population_summary"][pop] = {"status": "missing_samples"}
                    continue

                donor_ids = _subsample_ids(
                    donors_all,
                    target_n=min(len(donors_all), int(config.qc.ext_max_diploids_per_population)),
                    seed=int(derived_seed(seed, "plink2_parity_afs_donor", pop, chrom)),
                )
                synth_ids = _subsample_ids(
                    synth_all,
                    target_n=min(len(synth_all), int(config.qc.ext_max_diploids_per_population)),
                    seed=int(derived_seed(seed, "plink2_parity_afs_synth", pop, chrom)),
                )

                keep_d = tmp_dir / f"keep_donor_{pop}.txt"
                keep_s = tmp_dir / f"keep_synth_{pop}.txt"
                _write_keep_file(keep_d, donor_ids)
                _write_keep_file(keep_s, synth_ids)

                donor_vcf = donor_vcf_by_pop[pop]
                out_prefix_d = tmp_dir / f"afs_donor_{pop}"
                out_prefix_s = tmp_dir / f"afs_synth_{pop}"

                for vcf_path, keep_path, out_prefix in (
                    (donor_vcf, keep_d, out_prefix_d),
                    (synthetic_vcf, keep_s, out_prefix_s),
                ):
                    _run_cmd(
                        [
                            plink2_bin,
                            "--vcf",
                            str(vcf_path),
                            "--double-id",
                            "--allow-extra-chr",
                            "--max-alleles",
                            "2",
                            "--snps-only",
                            "just-acgt",
                            "--set-all-var-ids",
                            var_id_fmt,
                            "--extract",
                            str(snplist_path),
                            "--keep",
                            str(keep_path),
                            "--freq",
                            "counts",
                            "--threads",
                            str(threads),
                            "--out",
                            str(out_prefix),
                        ]
                    )

                _, ac_d, an_d = _load_plink_acount(Path(f"{out_prefix_d}.acount"))
                _, ac_s, an_s = _load_plink_acount(Path(f"{out_prefix_s}.acount"))
                probs_d_plink = _maf_hist_from_acount(ac_d, an_d, edges=edges_afs)
                probs_s_plink = _maf_hist_from_acount(ac_s, an_s, edges=edges_afs)

                donor_pos, donor_haps = _load_sample_haplotypes_from_panel(
                    config=config,
                    vcf_path=donor_vcf,
                    chrom=chrom,
                    sample_ids=donor_ids,
                )
                synth_pos, synth_haps = _load_synthetic_population_haplotypes(
                    config=config,
                    inputs=inputs,
                    chrom=chrom,
                    pop=pop,
                    sample_ids=synth_ids,
                )
                common_pos, donor_haps_aligned, synth_haps_aligned = _align_haplotype_axes_by_position(
                    donor_pos=donor_pos,
                    donor_haps=donor_haps,
                    synth_pos=synth_pos,
                    synth_haps=synth_haps,
                )
                pos_use = (
                    np.intersect1d(common_pos, parity_positions, assume_unique=False)
                    if parity_positions.size
                    else common_pos
                )
                if pos_use.size < 2:
                    afs_parity["population_summary"][pop] = {"status": "no_common_positions"}
                    continue
                idx = np.searchsorted(common_pos, pos_use)
                donor_haps_sel = donor_haps_aligned[:, idx]
                synth_haps_sel = synth_haps_aligned[:, idx]

                probs_d_py = _maf_hist_from_haps(donor_haps_sel, edges=edges_afs)
                probs_s_py = _maf_hist_from_haps(synth_haps_sel, edges=edges_afs)

                afs_parity["population_summary"][pop] = {
                    "status": "ok",
                    "n_positions_used": int(pos_use.size),
                    "l1_donor_hist": float(0.5 * np.sum(np.abs(probs_d_py - probs_d_plink))),
                    "l1_synth_hist": float(0.5 * np.sum(np.abs(probs_s_py - probs_s_plink))),
                }

            out["afs_parity"] = afs_parity

            # 3) KING parity (synthetic-only, small keep set).
            king_parity: dict[str, Any] = {"status": "ok", "within": {}, "cross": {}}
            n_target = _min_n_for_pairs(max_pairs)
            synth_ids_by_pop: dict[str, list[str]] = {}
            keep_all: list[str] = []
            for pop in pop_names:
                ids = groups.get(pop, [])
                if not ids:
                    continue
                picked = _subsample_ids(
                    ids,
                    target_n=min(len(ids), n_target),
                    seed=int(derived_seed(seed, "plink2_parity_king_samples", pop, chrom)),
                )
                synth_ids_by_pop[pop] = picked
                keep_all.extend(picked)

            keep_all = _dedupe_preserve_order(keep_all)
            if len(keep_all) < 2:
                king_parity["status"] = "skipped_no_samples"
            else:
                keep_path = tmp_dir / "keep_king_all.txt"
                _write_keep_file(keep_path, keep_all)
                king_prefix = tmp_dir / "king"
                _run_cmd(
                    [
                        plink2_bin,
                        "--vcf",
                        str(synthetic_vcf),
                        "--double-id",
                        "--allow-extra-chr",
                        "--max-alleles",
                        "2",
                        "--snps-only",
                        "just-acgt",
                        "--set-all-var-ids",
                        var_id_fmt,
                        "--extract",
                        str(snplist_path),
                        "--keep",
                        str(keep_path),
                        "--make-king-table",
                        "counts",
                        "cols=maybefid,id,nsnp,hethet,ibs0,kinship",
                        "--threads",
                        str(threads),
                        "--out",
                        str(king_prefix),
                    ]
                )
                kin0_path = Path(f"{king_prefix}.kin0")
                df_kin = pd.read_csv(kin0_path, sep=r"\s+", engine="python")
                cols = _pick_kinship_columns(df_kin)
                if cols is None:
                    king_parity["status"] = "error_parse"
                    king_parity["error"] = f"unexpected kin0 columns: {list(df_kin.columns)}"
                else:
                    id1_col, id2_col, kin_col = cols
                    kinship_by_pair: dict[tuple[str, str], float] = {}
                    for _, row in df_kin.iterrows():
                        a = str(row[id1_col])
                        b = str(row[id2_col])
                        if a == b:
                            continue
                        key = (a, b) if a < b else (b, a)
                        kinship_by_pair[key] = float(pd.to_numeric(row[kin_col], errors="coerce"))

                    per_pop_pos: dict[str, np.ndarray] = {}
                    per_pop_g: dict[str, np.ndarray] = {}
                    for pop, ids in synth_ids_by_pop.items():
                        if not ids:
                            continue
                        s_pos, s_haps = _load_synthetic_population_haplotypes(
                            config=config,
                            inputs=inputs,
                            chrom=chrom,
                            pop=pop,
                            sample_ids=ids,
                        )
                        per_pop_pos[pop] = np.asarray(s_pos, dtype=np.int64)
                        per_pop_g[pop] = haplotypes_to_genotypes_u8(s_haps)

                    if not per_pop_pos:
                        king_parity["status"] = "skipped_no_samples"
                    else:
                        pos_use_by_pop: dict[str, np.ndarray] = {}
                        g_use_by_pop: dict[str, np.ndarray] = {}
                        for pop, g_raw in per_pop_g.items():
                            pos_arr = per_pop_pos[pop]
                            pos_use = (
                                np.intersect1d(parity_positions, pos_arr, assume_unique=False)
                                if parity_positions.size
                                else pos_arr
                            )
                            if pos_use.size < 2:
                                continue
                            idx = np.searchsorted(pos_arr, pos_use)
                            if idx.size != pos_use.size or not np.array_equal(pos_arr[idx], pos_use):
                                continue
                            pos_use_by_pop[pop] = pos_use
                            g_use_by_pop[pop] = g_raw[:, idx]

                        if not g_use_by_pop:
                            king_parity["status"] = "skipped_no_common_positions"
                        else:
                            for pop, g in g_use_by_pop.items():
                                pairs_i, pairs_j = _sample_within_pairs(
                                    n_samples=int(g.shape[0]),
                                    max_pairs=max_pairs,
                                    seed=int(derived_seed(seed, "plink2_parity_king_within_pairs", pop, chrom)),
                                )
                                if pairs_i.size == 0:
                                    continue
                                kin_py = _king_within_pairs(g, pairs_i, pairs_j, numba_threads=None)
                                ids = synth_ids_by_pop[pop]
                                diffs: list[float] = []
                                for a_idx, b_idx, k_py in zip(pairs_i.tolist(), pairs_j.tolist(), kin_py.tolist()):
                                    a = str(ids[int(a_idx)])
                                    b = str(ids[int(b_idx)])
                                    key = (a, b) if a < b else (b, a)
                                    k_pl = kinship_by_pair.get(key)
                                    if k_pl is None or not np.isfinite(k_py) or not np.isfinite(k_pl):
                                        continue
                                    diffs.append(float(abs(k_py - k_pl)))
                                king_parity["within"][pop] = {
                                    "n_pairs_eval": int(len(diffs)),
                                    "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
                                    "max_abs_diff": float(np.max(diffs)) if diffs else float("nan"),
                                }

                            for pop_a, pop_b in combinations(sorted(g_use_by_pop.keys()), 2):
                                pos_a = pos_use_by_pop[pop_a]
                                pos_b = pos_use_by_pop[pop_b]
                                common_pos = np.intersect1d(pos_a, pos_b, assume_unique=False)
                                if common_pos.size < 2:
                                    continue
                                idx_a_v = np.searchsorted(pos_a, common_pos)
                                idx_b_v = np.searchsorted(pos_b, common_pos)
                                if idx_a_v.size != common_pos.size or idx_b_v.size != common_pos.size:
                                    continue
                                if not np.array_equal(pos_a[idx_a_v], common_pos) or not np.array_equal(
                                    pos_b[idx_b_v], common_pos
                                ):
                                    continue
                                ga = g_use_by_pop[pop_a][:, idx_a_v]
                                gb = g_use_by_pop[pop_b][:, idx_b_v]
                                samp_idx_a, samp_idx_b = _sample_cross_pairs(
                                    n_a=int(ga.shape[0]),
                                    n_b=int(gb.shape[0]),
                                    max_pairs=max_pairs,
                                    seed=int(derived_seed(seed, "plink2_parity_king_cross_pairs", pop_a, pop_b, chrom)),
                                )
                                if samp_idx_a.size == 0:
                                    continue
                                kin_py = _king_cross_pairs(ga, gb, samp_idx_a, samp_idx_b, numba_threads=None)
                                ids_a = synth_ids_by_pop[pop_a]
                                ids_b = synth_ids_by_pop[pop_b]
                                diffs: list[float] = []
                                for a_idx, b_idx, k_py in zip(
                                    samp_idx_a.tolist(), samp_idx_b.tolist(), kin_py.tolist()
                                ):
                                    a = str(ids_a[int(a_idx)])
                                    b = str(ids_b[int(b_idx)])
                                    key = (a, b) if a < b else (b, a)
                                    k_pl = kinship_by_pair.get(key)
                                    if k_pl is None or not np.isfinite(k_py) or not np.isfinite(k_pl):
                                        continue
                                    diffs.append(float(abs(k_py - k_pl)))
                                king_parity["cross"][f"{pop_a}__{pop_b}"] = {
                                    "n_pairs_eval": int(len(diffs)),
                                    "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
                                    "max_abs_diff": float(np.max(diffs)) if diffs else float("nan"),
                                }

            out["king_parity"] = king_parity

            # 4) PCA parity (PLINK2 projection recipe) + optional python comparison.
            if not pca_supported:
                out["pca_parity"] = {
                    "status": "skipped_multi_panel" if unique_donor_vcfs and len(unique_donor_vcfs) != 1 else "skipped_no_panel",
                    "donor_vcfs": unique_donor_vcfs,
                }
            else:
                donor_keep_raw: list[str] = []
                synth_keep_raw: list[str] = []
                donor_label_by_id: dict[str, str] = {}
                synth_label_by_id: dict[str, str] = {}
                for pop in pop_names:
                    d_all = donors_by_population.get(pop, [])
                    s_all = groups.get(pop, [])
                    if not d_all or not s_all:
                        continue
                    d_ids = _subsample_ids(
                        d_all,
                        target_n=min(len(d_all), int(config.qc.ext_max_diploids_per_population)),
                        seed=int(derived_seed(seed, "plink2_parity_pca_donor", pop, chrom)),
                    )
                    s_ids = _subsample_ids(
                        s_all,
                        target_n=min(len(s_all), int(config.qc.ext_max_diploids_per_population)),
                        seed=int(derived_seed(seed, "plink2_parity_pca_synth", pop, chrom)),
                    )
                    for sid in d_ids:
                        donor_keep_raw.append(str(sid))
                        donor_label_by_id[str(sid)] = pop
                    for sid in s_ids:
                        synth_keep_raw.append(str(sid))
                        synth_label_by_id[str(sid)] = pop

                donor_keep = _dedupe_preserve_order(donor_keep_raw)
                synth_keep = _dedupe_preserve_order(synth_keep_raw)
                donor_labels = [donor_label_by_id[sid] for sid in donor_keep]
                synth_labels = [synth_label_by_id[sid] for sid in synth_keep]

                if len(donor_keep) < 4 or len(synth_keep) < 4:
                    out["pca_parity"] = {"status": "skipped_insufficient_samples"}
                else:
                    try:
                        keep_d_all = tmp_dir / "keep_pca_donor_all.txt"
                        keep_s_all = tmp_dir / "keep_pca_synth_all.txt"
                        _write_keep_file(keep_d_all, donor_keep)
                        _write_keep_file(keep_s_all, synth_keep)

                        ref_prefix = tmp_dir / "pca_ref"
                        ref_pcs_prefix = tmp_dir / "pca_ref_pcs"
                        new_prefix = tmp_dir / "pca_new"
                        new_proj_prefix = tmp_dir / "pca_new_proj"

                        _run_cmd(
                            [
                                plink2_bin,
                                "--vcf",
                                str(donor_vcf_for_pca),
                                "--double-id",
                                "--allow-extra-chr",
                                "--max-alleles",
                                "2",
                                "--snps-only",
                                "just-acgt",
                                "--set-all-var-ids",
                                var_id_fmt,
                                "--extract",
                                str(snplist_path),
                                "--keep",
                                str(keep_d_all),
                                "--make-pgen",
                                "--out",
                                str(ref_prefix),
                            ]
                        )
                        _run_cmd(
                            [
                                plink2_bin,
                                "--pfile",
                                str(ref_prefix),
                                "--maf",
                                str(float(config.qc.pca_min_maf)),
                                "--freq",
                                "counts",
                                "--pca",
                                str(n_pcs),
                                "approx",
                                "allele-wts",
                                "vcols=chrom,ref,alt",
                                "--out",
                                str(ref_pcs_prefix),
                            ]
                        )

                        weights_filtered = tmp_dir / "pca_ref_pcs.eigenvec.allele.filtered"
                        weights_filter = _filter_pca_weights_by_ref_maf(
                            acount_path=Path(f"{ref_pcs_prefix}.acount"),
                            weights_path=Path(f"{ref_pcs_prefix}.eigenvec.allele"),
                            out_path=weights_filtered,
                            maf_min=float(config.qc.pca_min_maf),
                        )
                        if int(weights_filter.get("n_variants_weights_out", 0)) < 2:
                            out["pca_parity"] = {
                                "status": "skipped_no_variants_after_filter",
                                "weights_filter": weights_filter,
                            }
                            raise RuntimeError("plink2 PCA parity: no variants left after MAF filter")

                        _run_cmd(
                            [
                                plink2_bin,
                                "--vcf",
                                str(synthetic_vcf),
                                "--double-id",
                                "--allow-extra-chr",
                                "--max-alleles",
                                "2",
                                "--snps-only",
                                "just-acgt",
                                "--set-all-var-ids",
                                var_id_fmt,
                                "--extract",
                                str(snplist_path),
                                "--keep",
                                str(keep_s_all),
                                "--make-pgen",
                                "--out",
                                str(new_prefix),
                            ]
                        )

                        score_cols = f"6-{5 + int(n_pcs)}"
                        _run_cmd(
                            [
                                plink2_bin,
                                "--pfile",
                                str(new_prefix),
                                "--read-freq",
                                str(Path(f"{ref_pcs_prefix}.acount")),
                                "--score",
                                str(weights_filtered),
                                "2",
                                "5",
                                "header-read",
                                "no-mean-imputation",
                                "variance-standardize",
                                "--score-col-nums",
                                score_cols,
                                "--out",
                                str(new_proj_prefix),
                            ]
                        )

                        eigenvec = Path(f"{ref_pcs_prefix}.eigenvec")
                        sscore = Path(f"{new_proj_prefix}.sscore")
                        df_d = pd.read_csv(eigenvec, sep=r"\s+", engine="python")
                        df_s = pd.read_csv(sscore, sep=r"\s+", engine="python")
                        iid_col_d = _pick_iid_column(df_d)
                        iid_col_s = _pick_iid_column(df_s)
                        if iid_col_d is None or iid_col_s is None:
                            out["pca_parity"] = {"status": "error_parse", "error": "missing IID column"}
                        else:
                            pc_cols_d = [c for c in df_d.columns if re.match(r"^PC\d+$", str(c))]
                            pc_cols_d = pc_cols_d[: int(n_pcs)]
                            avg_cols_s = [c for c in df_s.columns if str(c).endswith("_AVG")]
                            avg_cols_s = avg_cols_s[: int(n_pcs)]
                            if len(pc_cols_d) < int(n_pcs) or len(avg_cols_s) < int(n_pcs):
                                out["pca_parity"] = {
                                    "status": "error_parse",
                                    "error": f"unexpected PC columns: eigenvec={pc_cols_d} sscore_avg={avg_cols_s}",
                                }
                            else:
                                d_scores = df_d[[iid_col_d] + pc_cols_d].copy()
                                s_scores = df_s[[iid_col_s] + avg_cols_s].copy()
                                d_scores.columns = ["IID"] + [f"PC{i+1}" for i in range(len(pc_cols_d))]
                                s_scores.columns = ["IID"] + [f"PC{i+1}" for i in range(len(avg_cols_s))]
                                d_scores["IID"] = d_scores["IID"].astype(str)
                                s_scores["IID"] = s_scores["IID"].astype(str)

                                d_scores = d_scores.set_index("IID").loc[donor_keep].reset_index()
                                s_scores = s_scores.set_index("IID").loc[synth_keep].reset_index()

                                d_mat = d_scores.drop(columns=["IID"]).to_numpy(dtype=np.float64)
                                s_mat = s_scores.drop(columns=["IID"]).to_numpy(dtype=np.float64)

                                d_std, mean_pc, std_pc = _standardize_scores(d_mat)
                                s_std = (s_mat - mean_pc) / std_pc

                                pops = sorted(set(donor_labels) | set(synth_labels))
                                k = min(int(n_pcs), d_std.shape[1], s_std.shape[1])
                                plink_auc = _auc_by_pair_from_donor_direction(
                                    donor_scores_std=d_std,
                                    donor_labels=donor_labels,
                                    synth_scores_std=s_std,
                                    synth_labels=synth_labels,
                                    pops=pops,
                                    k=k,
                                )

                                pca_payload: dict[str, Any] = {
                                    "status": "ok",
                                    "n_pcs_used": int(k),
                                    "plink2_auc_by_pair": plink_auc,
                                    "weights_filter": weights_filter,
                                }

                                try:
                                    from sklearn.utils.extmath import randomized_svd  # type: ignore
                                except Exception as exc:
                                    pca_payload.update(
                                        {
                                            "status": "error_no_sklearn",
                                            "error": f"scikit-learn is required for python PCA comparison: {exc}",
                                        }
                                    )
                                else:
                                    per_pop: dict[str, dict[str, Any]] = {}
                                    for pop in pop_names:
                                        d_all = donors_by_population.get(pop, [])
                                        s_all = groups.get(pop, [])
                                        if not d_all or not s_all:
                                            continue
                                        d_ids = _subsample_ids(
                                            d_all,
                                            target_n=min(len(d_all), int(config.qc.ext_max_diploids_per_population)),
                                            seed=int(derived_seed(seed, "plink2_parity_py_pca_donor", pop, chrom)),
                                        )
                                        s_ids = _subsample_ids(
                                            s_all,
                                            target_n=min(len(s_all), int(config.qc.ext_max_diploids_per_population)),
                                            seed=int(derived_seed(seed, "plink2_parity_py_pca_synth", pop, chrom)),
                                        )
                                        donor_vcf = donor_vcf_by_pop[pop]
                                        d_pos, d_haps = _load_sample_haplotypes_from_panel(
                                            config=config,
                                            vcf_path=donor_vcf,
                                            chrom=chrom,
                                            sample_ids=d_ids,
                                        )
                                        s_pos, s_haps = _load_synthetic_population_haplotypes(
                                            config=config,
                                            inputs=inputs,
                                            chrom=chrom,
                                            pop=pop,
                                            sample_ids=s_ids,
                                        )
                                        common_pos, d_h_al, s_h_al = _align_haplotype_axes_by_position(
                                            donor_pos=d_pos,
                                            donor_haps=d_haps,
                                            synth_pos=s_pos,
                                            synth_haps=s_haps,
                                        )
                                        per_pop[pop] = {
                                            "pos": np.asarray(common_pos, dtype=np.int64),
                                            "donor_h": d_h_al,
                                            "synth_h": s_h_al,
                                        }

                                    pops2 = sorted(per_pop.keys())
                                    if not pops2:
                                        pca_payload["python_status"] = "skipped_no_samples"
                                    else:
                                        global_pos = per_pop[pops2[0]]["pos"]
                                        for pop in pops2[1:]:
                                            global_pos = np.intersect1d(
                                                global_pos, per_pop[pop]["pos"], assume_unique=False
                                            )
                                        pos_use = (
                                            np.intersect1d(global_pos, parity_positions, assume_unique=False)
                                            if parity_positions.size
                                            else global_pos
                                        )
                                        if pos_use.size < 2:
                                            pca_payload["python_status"] = "skipped_no_common_positions"
                                        else:
                                            donor_blocks: list[np.ndarray] = []
                                            synth_blocks: list[np.ndarray] = []
                                            donor_lbls: list[str] = []
                                            synth_lbls: list[str] = []
                                            for pop in pops2:
                                                pos = per_pop[pop]["pos"]
                                                idx = np.searchsorted(pos, pos_use)
                                                if idx.size != pos_use.size or not np.array_equal(pos[idx], pos_use):
                                                    continue
                                                donor_g = haplotypes_to_genotypes_f32(per_pop[pop]["donor_h"][:, idx])
                                                synth_g = haplotypes_to_genotypes_f32(per_pop[pop]["synth_h"][:, idx])
                                                donor_blocks.append(donor_g)
                                                synth_blocks.append(synth_g)
                                                donor_lbls.extend([pop] * donor_g.shape[0])
                                                synth_lbls.extend([pop] * synth_g.shape[0])

                                            if not donor_blocks or not synth_blocks:
                                                pca_payload["python_status"] = "skipped_no_samples"
                                            else:
                                                donor_g_all = np.vstack(donor_blocks)
                                                synth_g_all = np.vstack(synth_blocks)

                                                obs = np.isfinite(donor_g_all)
                                                ac = np.nansum(donor_g_all.astype(np.float64), axis=0)
                                                n_obs = np.sum(obs, axis=0, dtype=np.float64)
                                                an = 2.0 * n_obs
                                                with np.errstate(divide="ignore", invalid="ignore"):
                                                    p = ac / an
                                                maf = np.minimum(p, 1.0 - p)
                                                valid = (
                                                    np.isfinite(p)
                                                    & np.isfinite(maf)
                                                    & (an >= 4.0)
                                                    & (maf >= float(config.qc.pca_min_maf))
                                                    & (maf <= 0.5)
                                                )
                                                if np.count_nonzero(valid) < 2:
                                                    pca_payload["python_status"] = "skipped_no_variants_after_filter"
                                                else:
                                                    p = p[valid]
                                                    mean = 2.0 * p
                                                    scale = np.sqrt(2.0 * p * (1.0 - p))
                                                    keep = scale > 1e-12
                                                    if np.count_nonzero(keep) < 2:
                                                        pca_payload["python_status"] = "skipped_no_variants_after_scale"
                                                    else:
                                                        mean = mean[keep]
                                                        scale = scale[keep]
                                                        d_g = donor_g_all[:, valid][:, keep]
                                                        s_g = synth_g_all[:, valid][:, keep]
                                                        x_d = _impute_center_scale(d_g, mean=mean, scale=scale)
                                                        x_s = _impute_center_scale(s_g, mean=mean, scale=scale)

                                                        n_comp = min(int(n_pcs), x_d.shape[0] - 1, x_d.shape[1])
                                                        if n_comp <= 0:
                                                            pca_payload["python_status"] = "skipped_insufficient_rank"
                                                        else:
                                                            _, _, vt = randomized_svd(
                                                                x_d,
                                                                n_components=int(n_comp),
                                                                random_state=int(seed),
                                                            )
                                                            d_sc = x_d @ vt.T
                                                            s_sc = x_s @ vt.T
                                                            d_sc_std, m2, sd2 = _standardize_scores(d_sc)
                                                            s_sc_std = (s_sc - m2) / sd2

                                                            py_auc = _auc_by_pair_from_donor_direction(
                                                                donor_scores_std=d_sc_std,
                                                                donor_labels=donor_lbls,
                                                                synth_scores_std=s_sc_std,
                                                                synth_labels=synth_lbls,
                                                                pops=pops,
                                                                k=min(int(n_comp), int(n_pcs)),
                                                            )
                                                            common_pairs = sorted(
                                                                set(plink_auc.keys()) & set(py_auc.keys())
                                                            )
                                                            diffs = [
                                                                abs(
                                                                    float(plink_auc[kp]["auc_synthetic"])
                                                                    - float(py_auc[kp]["auc_synthetic"])
                                                                )
                                                                for kp in common_pairs
                                                                if np.isfinite(plink_auc[kp]["auc_synthetic"])
                                                                and np.isfinite(py_auc[kp]["auc_synthetic"])
                                                            ]
                                                            pca_payload.update(
                                                                {
                                                                    "python_status": "ok",
                                                                    "python_auc_by_pair": py_auc,
                                                                    "synth_auc_abs_diff_mean": float(np.mean(diffs))
                                                                    if diffs
                                                                    else float("nan"),
                                                                    "synth_auc_abs_diff_max": float(np.max(diffs))
                                                                    if diffs
                                                                    else float("nan"),
                                                                    "n_pairs_compared": int(len(diffs)),
                                                                }
                                                            )

                                out["pca_parity"] = pca_payload

                    except Exception as exc:
                        if out.get("pca_parity", {}).get("status", "").startswith("skipped_"):
                            pass
                        else:
                            out["pca_parity"] = {"status": "error", "error": str(exc)}

            has_error = False
            for key in ("afs_parity", "king_parity", "pca_parity"):
                section = out.get(key)
                if not isinstance(section, dict):
                    continue
                status = section.get("status")
                if isinstance(status, str) and status.startswith("error"):
                    has_error = True
                    break

            out["status"] = "ok_with_errors" if has_error else "ok"
            return out

    except Exception as exc:
        out["status"] = "error"
        out["error"] = str(exc)
        logger.warning("PLINK2 parity harness failed: %s", exc)
        return out
