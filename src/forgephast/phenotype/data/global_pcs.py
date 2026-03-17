"""Global PCA orchestration utilities for manifest-based runs."""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids, read_sample_ids

from ..._internal.cache import bump_counter, canonical_hash
from ..config import GlobalPCConfig, RunConfig

logger = logging.getLogger(__name__)

_SVD_FALLBACK_MAX_ELEMENTS = 2_000_000
_SVD_FALLBACK_MAX_DIM = 2000


def _randomized_svd_compat(
    x: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD when available, otherwise bounded numpy fallback."""
    try:
        from sklearn.utils.extmath import randomized_svd  # type: ignore

        u, s, vt = randomized_svd(
            x,
            n_components=int(n_components),
            random_state=int(random_state),
        )
        return (
            np.asarray(u, dtype=np.float64),
            np.asarray(s, dtype=np.float64),
            np.asarray(vt, dtype=np.float64),
        )
    except Exception as exc:  # noqa: BLE001 - preserve fallback behavior
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size > _SVD_FALLBACK_MAX_ELEMENTS or min(x_arr.shape) > _SVD_FALLBACK_MAX_DIM:
            raise RuntimeError(
                "HapStore PCA requires scikit-learn randomized_svd for this matrix size; "
                "install scikit-learn or reduce svd_max_variants."
            ) from exc

        u, s, vt = np.linalg.svd(x_arr, full_matrices=False)
        k = min(int(n_components), int(vt.shape[0]))
        return u[:, :k], s[:k], vt[:k]


def _subset_variant_columns(
    x: np.ndarray,
    *,
    max_variants: int,
    random_seed: int,
) -> np.ndarray:
    """Deterministically subset columns when variant count exceeds cap."""
    if x.shape[1] <= int(max_variants):
        return x
    rng = np.random.default_rng(int(random_seed))
    idx = rng.choice(x.shape[1], size=int(max_variants), replace=False)
    idx.sort()
    return x[:, idx]


def _maf_filter_and_standardize(
    x: np.ndarray,
    *,
    min_maf: float,
) -> np.ndarray:
    """Apply MAF filter, mean imputation, and 2pq standardization."""
    observed = np.isfinite(x)
    ac = np.nansum(x.astype(np.float64), axis=0)
    n_obs = np.sum(observed, axis=0, dtype=np.float64)
    an = 2.0 * n_obs

    with np.errstate(divide="ignore", invalid="ignore"):
        p = ac / an
    maf = np.minimum(p, 1.0 - p)

    valid = (
        np.isfinite(p)
        & np.isfinite(maf)
        & (an >= 4.0)
        & (maf >= float(min_maf))
        & (maf <= 0.5)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("No variants remain after HapStore PCA MAF filtering")

    p = p[valid]
    mean = 2.0 * p
    scale = np.sqrt(2.0 * p * (1.0 - p))

    keep_scale = scale > 1e-12
    if np.count_nonzero(keep_scale) < 2:
        raise ValueError("No variants remain after HapStore PCA scale filtering")

    mean = mean[keep_scale]
    scale = scale[keep_scale]

    z = np.asarray(x[:, valid][:, keep_scale], dtype=np.float64)
    missing = ~np.isfinite(z)
    if np.any(missing):
        col_idx = np.where(missing)[1]
        z[missing] = mean[col_idx]

    z -= mean
    z /= scale
    return z


def compute_global_pcs_svd(
    *,
    genotypes: np.ndarray,
    sample_ids: list[str],
    n_components: int,
    max_variants: int,
    min_maf: float,
    random_seed: int,
) -> pd.DataFrame:
    """Compute PCs from dosage matrix using PCA-SVD workflow."""
    x = np.asarray(genotypes, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"genotypes must be 2D, got shape={x.shape}")
    if x.shape[0] != len(sample_ids):
        raise ValueError(
            f"genotype/sample mismatch: rows={x.shape[0]} sample_ids={len(sample_ids)}"
        )
    if int(n_components) <= 0:
        raise ValueError("n_components must be > 0")

    x = _subset_variant_columns(
        x,
        max_variants=int(max_variants),
        random_seed=int(random_seed),
    )
    x = _maf_filter_and_standardize(x, min_maf=float(min_maf))

    n_comp_eff = min(int(n_components), x.shape[0] - 1, x.shape[1])
    if n_comp_eff <= 0:
        raise ValueError("Insufficient rank to compute requested PCA components")

    _, _, vt = _randomized_svd_compat(
        x,
        n_components=n_comp_eff,
        random_state=int(random_seed),
    )
    scores = x @ vt.T

    out = pd.DataFrame({"FID": sample_ids, "IID": sample_ids})
    for i in range(1, int(n_components) + 1):
        if i <= n_comp_eff:
            out[f"PC{i}"] = scores[:, i - 1]
        else:
            out[f"PC{i}"] = 0.0
    return out


def _read_plink2_eigenvec(eigenvec_path: Path, n_components: int) -> pd.DataFrame:
    """Parse PLINK2 .eigenvec output with or without a header row."""
    first_line = ""
    with open(eigenvec_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                first_line = line
                break

    if not first_line:
        raise ValueError(f"Empty PLINK2 .eigenvec file: {eigenvec_path}")

    first_tokens = first_line.split()
    has_header = False
    if len(first_tokens) >= 2:
        header_start = {"#FID", "FID", "#IID", "IID"}
        has_header = (
            first_tokens[0] in header_start
            or first_tokens[1] in {"IID", "#IID"}
        )

    if has_header:
        pcs_df = pd.read_csv(eigenvec_path, sep=r"\s+")
    else:
        parsed_components = max(0, len(first_tokens) - 2)
        names = ["FID", "IID"] + [f"PC{i}" for i in range(1, parsed_components + 1)]
        pcs_df = pd.read_csv(eigenvec_path, sep=r"\s+", header=None, names=names)

    rename_map = {}
    if "#FID" in pcs_df.columns:
        rename_map["#FID"] = "FID"
    if "#IID" in pcs_df.columns:
        rename_map["#IID"] = "IID"
    if rename_map:
        pcs_df = pcs_df.rename(columns=rename_map)

    if "FID" not in pcs_df.columns and "IID" in pcs_df.columns:
        pcs_df["FID"] = pcs_df["IID"].astype(str)

    required = {"FID", "IID"}
    if not required.issubset(pcs_df.columns):
        raise ValueError(f"Unexpected .eigenvec columns: {list(pcs_df.columns)}")

    out = pcs_df[["FID", "IID"]].copy()
    for idx in range(1, n_components + 1):
        col = f"PC{idx}"
        if col in pcs_df.columns:
            out[col] = pd.to_numeric(pcs_df[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    return out


def _run_plink2_pca(
    *,
    pgen_prefix: str,
    plink2_path: str,
    sample_ids: list[str],
    n_components: int,
    out_dir: Path,
    mode_cfg: object,
    chromosomes: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run PLINK2 --pca directly on an exported PGEN prefix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_file = out_dir / "keep_samples.txt"
    with open(keep_file, "w", encoding="utf-8") as handle:
        handle.write("FID IID\n")
        for iid in sample_ids:
            handle.write(f"{iid} {iid}\n")

    approx = bool(getattr(mode_cfg, "plink2_approx", True))
    extra_args = getattr(mode_cfg, "plink2_extra_args", None)
    out_prefix = out_dir / "global_pcs"
    pca_args = [str(n_components)]
    if approx:
        pca_args.append("approx")

    cmd = [plink2_path, "--pfile", pgen_prefix, "--keep", str(keep_file)]
    if chromosomes:
        cmd.extend(["--chr", ",".join(str(int(c)) for c in chromosomes)])
    cmd.extend(["--pca", *pca_args, "--out", str(out_prefix)])
    if extra_args:
        cmd.extend(list(extra_args))

    logger.info("Computing global PCs with PLINK2: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "PLINK2 PCA failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    eigenvec_path = Path(f"{out_prefix}.eigenvec")
    if not eigenvec_path.exists():
        raise RuntimeError(f"Missing PLINK2 PCA output: {eigenvec_path}")

    pcs_df = _read_plink2_eigenvec(eigenvec_path, n_components=n_components)
    aligned = align_by_ids(
        pcs_df,
        id_column="IID",
        ids=sample_ids,
        context="global PCs alignment (PLINK2)",
    )
    aligned["FID"] = aligned["IID"].astype(str)
    ordered_cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, n_components + 1)]
    return aligned[ordered_cols].copy()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_precomputed_pcs(
    *,
    mode_cfg: GlobalPCConfig,
    sample_ids: list[str],
    out_path: Path,
) -> pd.DataFrame:
    pre = pd.read_csv(mode_cfg.precomputed_file, sep="\t")
    aligned = align_by_ids(pre, mode_cfg.id_column, sample_ids, "global PCs alignment")
    aligned = aligned.rename(columns={mode_cfg.id_column: "IID"})
    if "FID" not in aligned.columns:
        aligned.insert(0, "FID", aligned["IID"])
    keep_cols = ["FID", "IID"] + [c for c in aligned.columns if c.startswith("PC")]
    out = aligned[keep_cols].copy()
    out.to_csv(out_path, sep="\t", index=False)
    return out


def refresh_manifest_global_pcs(
    run_cfg: RunConfig,
    manifest_path: Path,
    runtime_cache: dict[str, Any] | None = None,
) -> Path:
    """Recompute/refresh global PCs for each manifest row on the current PGEN source."""
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if manifest_df.empty:
        raise ValueError(f"Empty manifest provided for PCA refresh: {manifest_path}")
    if "global_pcs_file" in manifest_df.columns:
        manifest_df["global_pcs_file"] = manifest_df["global_pcs_file"].fillna("").astype(str)
    else:
        manifest_df["global_pcs_file"] = ""

    mode = run_cfg.gwas.global_pcs.canonical_mode()
    if mode == "none":
        return manifest_path

    cache = None
    if runtime_cache is not None and bool(run_cfg.gwas.global_pcs.reuse_cache):
        cache = runtime_cache.setdefault("global_pcs_cache", {})

    out_manifest = manifest_path.parent / "run_manifest_with_pcs.tsv"
    logger.info(
        "[global-pcs] refresh start rows=%d mode=%s reuse_cache=%s",
        int(len(manifest_df)),
        mode,
        bool(cache is not None),
    )
    for idx, row in manifest_df.iterrows():
        keep_file = str(row["keep_file"])
        sample_ids = read_sample_ids(keep_file)
        pgen_prefix = str(row["pgen_prefix"])
        if not pgen_prefix:
            raise ValueError(
                f"Missing pgen_prefix for manifest row index={idx} during PCA refresh"
            )

        output_dir = Path(str(row["output_dir"]))
        pcs_path = output_dir / "global_pcs.tsv"
        pcs_path.parent.mkdir(parents=True, exist_ok=True)

        cache_key = canonical_hash(
            {
                "mode": mode,
                "pgen_prefix": pgen_prefix,
                "keep_file_hash": _file_hash(keep_file),
                "chromosomes": [int(chrom) for chrom in run_cfg.chromosomes],
                "n_components": int(run_cfg.gwas.global_pcs.n_components),
                "plink2_path": str(run_cfg.gwas.plink2.path),
                "plink2_approx": bool(run_cfg.gwas.global_pcs.plink2_approx),
                "plink2_extra_args": list(run_cfg.gwas.global_pcs.plink2_extra_args),
                "precomputed_file": str(run_cfg.gwas.global_pcs.precomputed_file or ""),
                "id_column": str(run_cfg.gwas.global_pcs.id_column),
            }
        )
        if cache is not None and cache_key in cache:
            bump_counter(runtime_cache, hit=True, name="global_pcs")
            cached_df = cache[cache_key]
            if not isinstance(cached_df, pd.DataFrame):
                raise TypeError("global_pcs_cache values must be pandas DataFrame objects")
            cached_df.to_csv(pcs_path, sep="\t", index=False)
            manifest_df.loc[idx, "global_pcs_file"] = str(pcs_path)
            logger.info("[global-pcs] cache-hit row=%d output=%s", int(idx), pcs_path)
            continue

        bump_counter(runtime_cache, hit=False, name="global_pcs")
        logger.info("[global-pcs] cache-miss row=%d output=%s", int(idx), pcs_path)
        if mode == "plink2_pca":
            pcs_df = _run_plink2_pca(
                pgen_prefix=pgen_prefix,
                plink2_path=str(run_cfg.gwas.plink2.path),
                sample_ids=sample_ids,
                n_components=int(run_cfg.gwas.global_pcs.n_components),
                out_dir=output_dir / "tmp" / "global_pcs_plink2",
                mode_cfg=run_cfg.gwas.global_pcs,
                chromosomes=run_cfg.chromosomes,
            )
            pcs_df.to_csv(pcs_path, sep="\t", index=False)
            logger.info("[global-pcs] refreshed with PLINK2 for %s", output_dir)
        elif mode == "precomputed":
            pcs_df = _write_precomputed_pcs(
                mode_cfg=run_cfg.gwas.global_pcs,
                sample_ids=sample_ids,
                out_path=pcs_path,
            )
            logger.info("[global-pcs] refreshed from precomputed table for %s", output_dir)
        elif mode == "svd_pca":
            if not pcs_path.exists():
                raise FileNotFoundError(
                    f"gwas.global_pcs.mode=svd_pca expects existing simulate-stage PCs: {pcs_path}"
                )
            pcs_df = pd.read_csv(pcs_path, sep="\t")
        else:
            raise ValueError(f"Unsupported global PC mode: {mode}")

        if cache is not None:
            cache[cache_key] = pcs_df.copy()
        manifest_df.loc[idx, "global_pcs_file"] = str(pcs_path)

    manifest_df.to_csv(out_manifest, sep="\t", index=False)
    if runtime_cache is not None:
        logger.info(
            "[global-pcs] refresh done hits=%s misses=%s out_manifest=%s",
            runtime_cache.get("cache_hits", {}).get("global_pcs", 0),
            runtime_cache.get("cache_misses", {}).get("global_pcs", 0),
            out_manifest,
        )
    return out_manifest
