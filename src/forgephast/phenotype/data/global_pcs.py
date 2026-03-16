"""Global PCA orchestration utilities for manifest-based runs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import GlobalPCConfig, RunConfig
from ..utils.ids import align_by_ids, read_sample_ids
from .genotypes import CanonicalPgenAdapter

logger = logging.getLogger(__name__)


def _canonical_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _bump(runtime_cache: dict[str, Any] | None, *, hit: bool) -> None:
    if runtime_cache is None:
        return
    key = "cache_hits" if hit else "cache_misses"
    counters = runtime_cache.setdefault(key, {})
    counters["global_pcs"] = int(counters.get("global_pcs", 0)) + 1


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

        cache_key = _canonical_hash(
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
            _bump(runtime_cache, hit=True)
            cached_df = cache[cache_key]
            if not isinstance(cached_df, pd.DataFrame):
                raise TypeError("global_pcs_cache values must be pandas DataFrame objects")
            cached_df.to_csv(pcs_path, sep="\t", index=False)
            manifest_df.loc[idx, "global_pcs_file"] = str(pcs_path)
            logger.info("[global-pcs] cache-hit row=%d output=%s", int(idx), pcs_path)
            continue

        _bump(runtime_cache, hit=False)
        logger.info("[global-pcs] cache-miss row=%d output=%s", int(idx), pcs_path)
        if mode == "plink2_pca":
            adapter = CanonicalPgenAdapter(
                pgen_prefix=pgen_prefix,
                plink2_path=run_cfg.gwas.plink2.path,
                pgen_dosage_backend=run_cfg.genotype_input.pgen_dosage_backend,
            )
            pcs_df = adapter.compute_global_pcs(
                sample_ids=sample_ids,
                sample_populations=["NA"] * len(sample_ids),
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
