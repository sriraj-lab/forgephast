"""HapStore -> PGEN export orchestration for GWAS round-trip workflows."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hapstore.ids import align_by_ids, read_sample_ids

from ..._internal.cache import bump_counter, canonical_hash
from ..config import CohortConfig, RunConfig
from .genotypes import build_genotype_adapter
from .global_pcs import normalize_psam_fid_column

logger = logging.getLogger(__name__)


@dataclass
class ExportTask:
    """One cohort/replicate export unit."""

    replicate: int
    cohort: str
    sample_ids: List[str]
    sample_populations: List[str]
    variant_ids: List[str] | None
    out_prefix: Path
    cache_key: str


@dataclass
class ExportArtifacts:
    """Resolved PGEN artifacts for one cohort/replicate export unit."""

    pgen_prefix: str
    chrom_pgen_manifest: dict[int, str]


def _pgen_triplet_exists(prefix: str | Path) -> bool:
    base = Path(prefix)
    return all(Path(f"{base}{ext}").exists() for ext in (".pgen", ".pvar", ".psam"))


def _normalize_fid_for_exported_pgen(path: str) -> None:
    """Align FID to IID for tables consumed by PLINK/REGENIE on exported PGEN."""
    table_path = Path(path)
    if not table_path.exists():
        return
    df = pd.read_csv(table_path, sep="\t")
    if "FID" not in df.columns:
        return
    if "IID" not in df.columns:
        return
    df["FID"] = df["IID"].astype(str)
    df.to_csv(table_path, sep="\t", index=False)


def _count_pvar_variants(pvar_path: Path) -> int:
    count = -1
    with open(pvar_path, "r", encoding="utf-8") as handle:
        for count, _ in enumerate(handle):
            pass
    return max(0, count)


def _plink_memory_args(run_cfg: RunConfig) -> list[str]:
    memory_mb = getattr(run_cfg.gwas.plink2, "memory_mb", None)
    if memory_mb is None:
        return []
    return ["--memory", str(int(memory_mb))]


def _keep_file_suffix(keep_file: str | Path | None) -> str:
    """Return a deterministic suffix fragment for keep-specific filtered PGENs."""
    if keep_file is None:
        return ""
    keep_str = str(keep_file).strip()
    if not keep_str:
        return ""
    digest = hashlib.sha1(keep_str.encode("utf-8")).hexdigest()[:12]
    return f".keep_{digest}"


def _maf_filter_suffix(min_maf: float) -> str:
    value = float(min_maf)
    if value == 1e-12:
        return ".maf_gt0"
    token = repr(value).replace("+", "").replace("-", "m").replace(".", "p")
    return f".maf_ge_{token}"


def _keep_matches_pgen_samples(prefix: str | Path, keep_file: str | Path | None) -> bool:
    if keep_file is None or not str(keep_file).strip():
        return True
    try:
        pgen_ids = read_sample_ids(Path(f"{prefix}.psam"))
        keep_ids = read_sample_ids(keep_file)
    except (OSError, ValueError):
        return False
    return len(pgen_ids) == len(keep_ids) and set(pgen_ids) == set(keep_ids)


def _filter_zero_maf_pgen(
    prefix: str | Path,
    *,
    plink2_path: str,
    threads: int,
    memory_mb: int | None = None,
    keep_file: str | Path | None = None,
    min_maf: float = 1e-12,
) -> str:
    """Create a filtered PGEN with monomorphic variants removed."""
    input_prefix = Path(prefix)
    keep_suffix = _keep_file_suffix(keep_file)
    filter_suffix = _maf_filter_suffix(float(min_maf))
    if keep_suffix and input_prefix.name.endswith(f"{filter_suffix}{keep_suffix}"):
        return str(input_prefix)
    already_filtered = input_prefix.name.endswith(filter_suffix)
    if already_filtered and not keep_suffix:
        return str(input_prefix)

    filtered_name = (
        f"{input_prefix.name}{keep_suffix}"
        if already_filtered
        else f"{input_prefix.name}{filter_suffix}{keep_suffix}"
    )
    filtered_prefix = input_prefix.parent / filtered_name
    if _pgen_triplet_exists(filtered_prefix):
        normalize_psam_fid_column(Path(f"{filtered_prefix}.psam"))
        return str(filtered_prefix)

    tmp_prefix = filtered_prefix.parent / f"{filtered_prefix.name}.tmp"
    for ext in (".pgen", ".pvar", ".psam", ".log"):
        Path(f"{tmp_prefix}{ext}").unlink(missing_ok=True)

    cmd = [
        plink2_path,
        "--pfile",
        str(input_prefix),
    ]
    if keep_file is not None and str(keep_file).strip():
        cmd.extend(["--keep", str(keep_file)])
    cmd.extend(
        [
        "--maf",
        str(float(min_maf)),
        "--make-pgen",
        "--threads",
        str(max(1, int(threads))),
        ]
    )
    if memory_mb is not None:
        cmd.extend(["--memory", str(int(memory_mb))])
    cmd.extend(["--out", str(tmp_prefix)])
    logger.info("Filtering monomorphic variants from PGEN: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "PLINK2 monomorphic-variant filter failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    for ext in (".pgen", ".pvar", ".psam"):
        if not Path(f"{tmp_prefix}{ext}").exists():
            raise RuntimeError(f"Filtered PGEN missing output artifact: {tmp_prefix}{ext}")

    for ext in (".pgen", ".pvar", ".psam", ".log"):
        Path(f"{filtered_prefix}{ext}").unlink(missing_ok=True)
    for ext in (".pgen", ".pvar", ".psam", ".log"):
        src = Path(f"{tmp_prefix}{ext}")
        if src.exists():
            shutil.move(str(src), str(Path(f"{filtered_prefix}{ext}")))

    normalize_psam_fid_column(Path(f"{filtered_prefix}.psam"))
    before_n = _count_pvar_variants(Path(f"{input_prefix}.pvar"))
    after_n = _count_pvar_variants(Path(f"{filtered_prefix}.pvar"))
    logger.info(
        "[hapstore-export] monomorphic filter before=%d after=%d removed=%d prefix=%s",
        before_n,
        after_n,
        max(0, before_n - after_n),
        filtered_prefix,
    )
    return str(filtered_prefix)


def ensure_manifest_non_monomorphic_pgens(run_cfg: RunConfig, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite manifest PGEN prefixes to monomorphic-filtered exports."""
    if "pgen_prefix" not in manifest_df.columns:
        raise ValueError("Manifest missing pgen_prefix for monomorphic filtering")
    filtered_df = manifest_df.copy()
    if not bool(run_cfg.pgen_export.filter_monomorphic):
        return filtered_df
    if "merged_pgen_prefix" not in filtered_df.columns:
        filtered_df["merged_pgen_prefix"] = filtered_df["pgen_prefix"].fillna("").astype(str)
    if "filtered_pgen_prefix" not in filtered_df.columns:
        filtered_df["filtered_pgen_prefix"] = ""
    if "pgen_filter_mode" not in filtered_df.columns:
        filtered_df["pgen_filter_mode"] = ""
    if "pgen_filter_min_maf" not in filtered_df.columns:
        filtered_df["pgen_filter_min_maf"] = np.nan

    prefixes = filtered_df["pgen_prefix"].fillna("").astype(str).str.strip()
    keep_files = (
        filtered_df["keep_file"].fillna("").astype(str).str.strip()
        if "keep_file" in filtered_df.columns
        else pd.Series([""] * len(filtered_df), index=filtered_df.index, dtype="string")
    )
    filter_modes = filtered_df["pgen_filter_mode"].fillna("").astype(str)
    recorded_min_mafs = pd.to_numeric(
        filtered_df["pgen_filter_min_maf"], errors="coerce"
    )
    configured_min_maf = float(run_cfg.pgen_export.min_maf)
    known_filter_modes = {
        "native_hapstore",
        "native_hapstore_keep_fallback",
        "plink2_fallback",
    }
    too_strict = [
        prefix
        for prefix, mode, recorded_min_maf in zip(
            prefixes.tolist(),
            filter_modes.tolist(),
            recorded_min_mafs.tolist(),
            strict=False,
        )
        if prefix
        and mode in known_filter_modes
        and np.isfinite(recorded_min_maf)
        and float(recorded_min_maf) > configured_min_maf
    ]
    if too_strict:
        raise ValueError(
            "Manifest PGEN was previously filtered at a stricter min_maf and cannot "
            "recover excluded variants; re-export from HapStore for prefixes: "
            f"{too_strict[:3]}"
        )
    filter_threshold_matches = [
        mode in known_filter_modes
        and np.isfinite(recorded_min_maf)
        and float(recorded_min_maf) == configured_min_maf
        for mode, recorded_min_maf in zip(
            filter_modes.tolist(),
            recorded_min_mafs.tolist(),
            strict=False,
        )
    ]
    filter_ready = [
        bool(prefix)
        and threshold_matches
        and _keep_matches_pgen_samples(prefix, keep_file or None)
        for prefix, keep_file, threshold_matches in zip(
            prefixes.tolist(),
            keep_files.tolist(),
            filter_threshold_matches,
            strict=False,
        )
    ]
    fallback_pairs = sorted(
        {
            (prefix, keep_file)
            for prefix, keep_file, is_filter_ready in zip(
                prefixes.tolist(),
                keep_files.tolist(),
                filter_ready,
                strict=False,
            )
            if prefix and not is_filter_ready
        }
    )
    if not prefixes.ne("").any():
        return filtered_df

    plink2_path = str(run_cfg.gwas.plink2.path or run_cfg.genotype_input.plink2_path)
    threads = max(1, int(run_cfg.gwas.plink2.threads))
    prefix_map = {
        (prefix, keep_file): _filter_zero_maf_pgen(
            prefix,
            plink2_path=plink2_path,
            threads=threads,
            memory_mb=getattr(run_cfg.gwas.plink2, "memory_mb", None),
            keep_file=(keep_file or None),
            min_maf=configured_min_maf,
        )
        for prefix, keep_file in fallback_pairs
    }

    filtered_df["filtered_pgen_prefix"] = [
        prefix if is_filter_ready else prefix_map.get((prefix, keep_file), prefix)
        for prefix, keep_file, is_filter_ready in zip(
            prefixes.tolist(),
            keep_files.tolist(),
            filter_ready,
            strict=False,
        )
    ]
    filtered_df["pgen_filter_mode"] = [
        ""
        if not prefix
        else (
            mode
            if is_filter_ready
            else (
                "native_hapstore_keep_fallback"
                if mode == "native_hapstore" and threshold_matches
                else "plink2_fallback"
            )
        )
        for prefix, mode, threshold_matches, is_filter_ready in zip(
            prefixes.tolist(),
            filter_modes.tolist(),
            filter_threshold_matches,
            filter_ready,
            strict=False,
        )
    ]
    filtered_df.loc[prefixes.ne(""), "pgen_filter_min_maf"] = configured_min_maf
    changed = filtered_df["filtered_pgen_prefix"].astype(str) != prefixes
    filtered_df["pgen_prefix"] = filtered_df["filtered_pgen_prefix"]
    if changed.any() and "global_pcs_file" in filtered_df.columns:
        filtered_df.loc[changed, "global_pcs_file"] = ""
    if changed.any() and "merged_covariates_file" in filtered_df.columns:
        filtered_df.loc[changed, "merged_covariates_file"] = ""
    return filtered_df


def _load_population_labels(
    cohort: CohortConfig,
    sample_ids: List[str],
    default_population: str,
) -> List[str]:
    """Load per-sample population labels aligned to IID order."""
    if not cohort.population_map_file:
        return [default_population] * len(sample_ids)

    pop_df = pd.read_csv(cohort.population_map_file, sep="\t")
    id_col = "IID" if "IID" in pop_df.columns else "#IID" if "#IID" in pop_df.columns else None
    if id_col is None:
        raise ValueError(
            f"cohort {cohort.name}: population_map_file must contain IID or #IID column"
        )

    pop_col = "population" if "population" in pop_df.columns else None
    if pop_col is None:
        other = [c for c in pop_df.columns if c != id_col]
        if not other:
            raise ValueError(f"cohort {cohort.name}: no population column found")
        pop_col = other[0]

    aligned = align_by_ids(
        pop_df[[id_col, pop_col]].copy(),
        id_column=id_col,
        ids=sample_ids,
        context=f"cohort {cohort.name} population map alignment",
    )
    return aligned[pop_col].astype(str).tolist()


def _pick_tag_column(df: pd.DataFrame, causal_col: str) -> str | None:
    """Best-effort tag-ID column resolver from source pairs."""
    candidates = ["tag_id", "tag_variant_id", "tag", "proxy_id", "linked_variant_id"]
    for col in candidates:
        if col in df.columns and col != causal_col:
            return col
    return None


def _read_variant_list_file(path: str | None) -> List[str]:
    if not path:
        raise ValueError("variant_list_file is required for variant_scope=variant_list")
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token:
                out.append(token.split()[0])
    if not out:
        raise ValueError(f"No variants found in variant list file: {path}")
    return out


def _random_hapstore_variant_ids(
    *,
    run_cfg: RunConfig,
    exclude: set[str],
    n_needed: int,
) -> List[str]:
    """Sample additional chr:pos variant IDs from HapStore reference population."""
    if n_needed <= 0:
        return []
    hap_cfg = run_cfg.genotype_input.hapstore
    if hap_cfg is None:
        raise ValueError("Cannot sample random HapStore variants without genotype_input.hapstore")

    try:
        from hapstore import HapStore  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Random HapStore variant sampling requires hapstore") from exc

    ref_pop = run_cfg.effect_model.population_order[0]
    rng = np.random.default_rng(int(run_cfg.pgen_export.random_seed))

    candidates: list[str] = []
    for chrom in run_cfg.chromosomes:
        if getattr(hap_cfg, "layout", "per_pop_per_chr") == "single_store":
            if not getattr(hap_cfg, "store_path", None):
                continue
            path = Path(str(hap_cfg.store_path))
        else:
            path = Path(hap_cfg.root_dir) / hap_cfg.file_pattern.format(
                population=ref_pop, chrom=int(chrom)
            )
        if not path.exists():
            continue
        store = HapStore(path, mode="r")
        pos = np.asarray(store.variant_positions(int(chrom)), dtype=np.int64)
        if pos.size == 0:
            continue
        ids = [f"chr{int(chrom)}:{int(p)}" for p in pos.tolist()]
        candidates.extend([vid for vid in ids if vid not in exclude])

    if not candidates:
        return []
    if len(candidates) <= n_needed:
        return candidates
    pick_idx = rng.choice(len(candidates), size=n_needed, replace=False)
    return [candidates[int(i)] for i in sorted(pick_idx.tolist())]


def _resolve_variant_scope(
    *,
    run_cfg: RunConfig,
    replicate: int,
    cohort: str,
    membership_df: pd.DataFrame,
) -> List[str] | None:
    """Build requested export variant set according to pgen_export.variant_scope."""
    scope = run_cfg.pgen_export.variant_scope
    if scope == "all":
        return None

    causal_col = run_cfg.causal_selection.variant_id_column
    source_pairs = pd.read_csv(run_cfg.causal_selection.source_pairs_file, sep="\t")
    if causal_col not in source_pairs.columns:
        raise ValueError(
            f"source_pairs_file missing configured causal column {causal_col!r} for export scoping"
        )

    if scope == "variant_list":
        base = _read_variant_list_file(run_cfg.pgen_export.variant_list_file)
        return list(dict.fromkeys(base))

    causal_members = membership_df[
        (membership_df["replicate"] == int(replicate))
        & (membership_df["cohort"] == str(cohort))
    ][causal_col].astype(str).tolist()
    variant_set = set(causal_members)

    if scope in {"causal_plus_tags", "causal_plus_tags_random"}:
        tag_col = _pick_tag_column(source_pairs, causal_col=causal_col)
        if tag_col is None:
            logger.warning(
                "pgen_export.variant_scope=%s requested tags but no tag column found; using causal-only scope",
                scope,
            )
        else:
            source_pairs = source_pairs[[causal_col, tag_col]].dropna()
            source_pairs[causal_col] = source_pairs[causal_col].astype(str)
            source_pairs[tag_col] = source_pairs[tag_col].astype(str)
            tag_rows = source_pairs[source_pairs[causal_col].isin(variant_set)]
            variant_set.update(tag_rows[tag_col].tolist())

    if scope == "causal_plus_tags_random":
        target_n = run_cfg.pgen_export.target_n_variants
        if target_n is not None:
            current = len(variant_set)
            if current < int(target_n):
                extras = _random_hapstore_variant_ids(
                    run_cfg=run_cfg,
                    exclude=set(variant_set),
                    n_needed=int(target_n) - current,
                )
                variant_set.update(extras)

    if run_cfg.pgen_export.target_n_variants is not None:
        target_n = int(run_cfg.pgen_export.target_n_variants)
        if len(variant_set) > target_n:
            ordered = sorted(variant_set)
            variant_set = set(ordered[:target_n])

    out = sorted(variant_set)
    if not out:
        raise ValueError(
            f"variant scope {scope} resolved to zero variants for replicate={replicate} cohort={cohort}"
        )
    return out


def _build_export_cache_key(
    run_cfg: RunConfig,
    *,
    sample_ids: list[str],
    sample_populations: list[str],
    variant_ids: list[str] | None,
) -> str:
    hap_cfg = run_cfg.genotype_input.hapstore
    return canonical_hash(
        {
            "layout": None if hap_cfg is None else str(hap_cfg.layout),
            "root_dir": None if hap_cfg is None else str(hap_cfg.root_dir),
            "file_pattern": None if hap_cfg is None else str(hap_cfg.file_pattern),
            "store_path": None if hap_cfg is None else str(hap_cfg.store_path),
            "sample_ids": list(sample_ids),
            "sample_populations": list(sample_populations),
            "variant_ids": None if variant_ids is None else list(variant_ids),
            "chromosomes": [int(chrom) for chrom in run_cfg.chromosomes],
            "plink2_path": str(run_cfg.gwas.plink2.path or run_cfg.genotype_input.plink2_path),
            "merge_threads": int(run_cfg.gwas.plink2.threads),
            "plink2_memory_mb": getattr(run_cfg.gwas.plink2, "memory_mb", None),
            "prep_mode": str(run_cfg.pgen_export.prep_mode),
            "concat_workers": int(run_cfg.pgen_export.concat_workers),
            "pgen_shard_workers": int(run_cfg.pgen_export.pgen_shard_workers),
            "filter_monomorphic": bool(run_cfg.pgen_export.filter_monomorphic),
            "min_maf": float(run_cfg.pgen_export.min_maf),
            "shard_tmp_dir": (
                None
                if run_cfg.pgen_export.shard_tmp_dir is None
                else str(run_cfg.pgen_export.shard_tmp_dir)
            ),
        }
    )


def _variant_ids_by_chrom(
    variant_ids: list[str] | None,
) -> dict[int, list[str]]:
    by_chrom: dict[int, list[str]] = {}
    if variant_ids is None:
        return by_chrom
    for variant_id in variant_ids:
        prefix, _, rest = str(variant_id).partition(":")
        chrom_text = prefix.replace("chr", "", 1)
        if not chrom_text or not rest:
            continue
        by_chrom.setdefault(int(chrom_text), []).append(str(variant_id))
    return by_chrom


def _expected_chrom_prefix(
    out_prefix: Path,
    chrom: int,
    run_cfg: RunConfig,
) -> Path:
    chrom_root = out_prefix.parent / "chrom"
    chrom_root.mkdir(parents=True, exist_ok=True)
    suffix = (
        _maf_filter_suffix(float(run_cfg.pgen_export.min_maf))
        if bool(run_cfg.pgen_export.filter_monomorphic)
        else ""
    )
    return chrom_root / f"{out_prefix.name}.chr{int(chrom)}{suffix}"


def _expected_merged_prefix(out_prefix: Path, run_cfg: RunConfig) -> Path:
    if not bool(run_cfg.pgen_export.filter_monomorphic):
        return out_prefix
    return out_prefix.with_name(
        f"{out_prefix.name}{_maf_filter_suffix(float(run_cfg.pgen_export.min_maf))}"
    )


def _merge_filtered_chrom_pgens(
    *,
    chrom_prefixes: dict[int, str],
    out_prefix: Path,
    run_cfg: RunConfig,
) -> str:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if not chrom_prefixes:
        raise ValueError("No chromosome-level PGEN prefixes available for merge")

    ordered = [Path(chrom_prefixes[chrom]) for chrom in sorted(chrom_prefixes)]
    if _pgen_triplet_exists(out_prefix):
        normalize_psam_fid_column(Path(f"{out_prefix}.psam"))
        return str(out_prefix)

    if len(ordered) == 1:
        src = ordered[0]
        for ext in (".pgen", ".pvar", ".psam"):
            shutil.copy2(f"{src}{ext}", f"{out_prefix}{ext}")
        normalize_psam_fid_column(Path(f"{out_prefix}.psam"))
        return str(out_prefix)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pmerge.list",
        delete=False,
        encoding="utf-8",
    ) as handle:
        merge_list = Path(handle.name)
        for prefix in ordered:
            handle.write(f"{prefix}\n")

    merge_threads = max(1, int(run_cfg.pgen_export.concat_workers))
    cmd = [
        str(run_cfg.gwas.plink2.path or run_cfg.genotype_input.plink2_path),
        "--threads",
        str(merge_threads),
        "--pmerge-list",
        str(merge_list),
        "pfile",
        "--indiv-sort",
        "none",
        "--make-pgen",
        *(_plink_memory_args(run_cfg)),
        "--out",
        str(out_prefix),
    ]
    logger.info("[hapstore-export] merging chromosome PGENs: %s", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "PLINK2 --pmerge-list failed during chromosome-level merge\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
    finally:
        merge_list.unlink(missing_ok=True)

    normalize_psam_fid_column(Path(f"{out_prefix}.psam"))
    return str(out_prefix)


def _export_one_task_legacy(task: ExportTask, run_cfg: RunConfig, merge_cfg: SimpleNamespace) -> str:
    adapter = build_genotype_adapter(
        hapstore_cfg=run_cfg.genotype_input.hapstore,
    )
    adapter.validate_files()
    legacy_export_cfg = SimpleNamespace(**vars(merge_cfg))
    legacy_export_cfg.filter_monomorphic = False
    return adapter.export_to_pgen(
        sample_ids=task.sample_ids,
        sample_populations=task.sample_populations,
        out_prefix=task.out_prefix,
        chromosomes=run_cfg.chromosomes,
        variant_ids=task.variant_ids,
        export_cfg=legacy_export_cfg,
    )


def _export_one_task_per_chrom(
    task: ExportTask,
    run_cfg: RunConfig,
    merge_cfg: SimpleNamespace,
) -> ExportArtifacts:
    adapter = build_genotype_adapter(hapstore_cfg=run_cfg.genotype_input.hapstore)
    adapter.validate_files()

    variant_ids_by_chrom = _variant_ids_by_chrom(task.variant_ids)
    chrom_prefixes: dict[int, str] = {}
    for chrom in [int(chrom) for chrom in run_cfg.chromosomes]:
        requested_ids = variant_ids_by_chrom.get(chrom) if task.variant_ids is not None else None
        if task.variant_ids is not None and not requested_ids:
            continue
        chrom_prefix = _expected_chrom_prefix(task.out_prefix, chrom, run_cfg)
        if _pgen_triplet_exists(chrom_prefix):
            normalize_psam_fid_column(Path(f"{chrom_prefix}.psam"))
            chrom_prefixes[chrom] = str(chrom_prefix)
            continue

        adapter.export_to_pgen(
            sample_ids=task.sample_ids,
            sample_populations=task.sample_populations,
            out_prefix=chrom_prefix,
            chromosomes=[chrom],
            variant_ids=None if requested_ids is None else list(requested_ids),
            export_cfg=merge_cfg,
        )
        chrom_prefixes[chrom] = str(chrom_prefix)

    merged_prefix = _merge_filtered_chrom_pgens(
        chrom_prefixes=chrom_prefixes,
        out_prefix=_expected_merged_prefix(task.out_prefix, run_cfg),
        run_cfg=run_cfg,
    )
    return ExportArtifacts(pgen_prefix=merged_prefix, chrom_pgen_manifest=chrom_prefixes)


def export_manifest_pgens(
    run_cfg: RunConfig,
    manifest_path: Path,
    runtime_cache: dict[str, Any] | None = None,
) -> Path:
    """Export HapStore cohorts to PGEN and emit updated manifest path."""
    if not run_cfg.pgen_export.enabled:
        raise ValueError("pgen_export.enabled must be true for genotype_input.format=hapstore")

    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if manifest_df.empty:
        raise ValueError(f"Empty manifest provided for export: {manifest_path}")
    if "pgen_prefix" not in manifest_df.columns:
        raise ValueError(f"Manifest missing required column 'pgen_prefix': {manifest_path}")
    manifest_df["pgen_prefix"] = manifest_df["pgen_prefix"].fillna("").astype(str)
    if "exported_pgen_prefix" not in manifest_df.columns:
        manifest_df["exported_pgen_prefix"] = ""
    if "merged_pgen_prefix" not in manifest_df.columns:
        manifest_df["merged_pgen_prefix"] = ""
    if "chrom_pgen_manifest" not in manifest_df.columns:
        manifest_df["chrom_pgen_manifest"] = ""
    if "filtered_pgen_prefix" not in manifest_df.columns:
        manifest_df["filtered_pgen_prefix"] = ""
    if "pgen_filter_mode" not in manifest_df.columns:
        manifest_df["pgen_filter_mode"] = ""
    if "pgen_filter_min_maf" not in manifest_df.columns:
        manifest_df["pgen_filter_min_maf"] = np.nan

    truth_membership_path = manifest_path.parent / "truth" / "causal_membership.tsv"
    if not truth_membership_path.exists():
        raise FileNotFoundError(
            f"Missing causal truth file required for export scoping: {truth_membership_path}"
        )
    membership_df = pd.read_csv(truth_membership_path, sep="\t")

    cohort_by_name: Dict[str, CohortConfig] = {cohort.name: cohort for cohort in run_cfg.cohorts}

    export_root = (
        Path(run_cfg.pgen_export.out_dir)
        if run_cfg.pgen_export.out_dir
        else manifest_path.parent / "exports" / "pgen"
    )
    export_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[hapstore-export] start rows=%d variant_scope=%s task_workers=%d "
        "reuse_exports=%s filter_monomorphic=%s min_maf=%g shard_tmp_dir=%s",
        int(len(manifest_df)),
        str(run_cfg.pgen_export.variant_scope),
        int(run_cfg.pgen_export.task_workers),
        bool(run_cfg.pgen_export.reuse_exports),
        bool(run_cfg.pgen_export.filter_monomorphic),
        float(run_cfg.pgen_export.min_maf),
        run_cfg.pgen_export.shard_tmp_dir,
    )

    default_pop = run_cfg.effect_model.population_order[0]
    tasks: list[ExportTask] = []

    keys = (
        manifest_df[["replicate", "cohort"]]
        .drop_duplicates()
        .sort_values(["replicate", "cohort"])
        .itertuples(index=False, name=None)
    )
    for replicate, cohort_name in keys:
        cohort = cohort_by_name.get(str(cohort_name))
        if cohort is None:
            raise ValueError(f"Manifest cohort {cohort_name!r} not found in config cohorts")

        sample_ids = read_sample_ids(cohort.sample_ids_file)
        sample_populations = _load_population_labels(
            cohort=cohort,
            sample_ids=sample_ids,
            default_population=default_pop,
        )
        variant_ids = _resolve_variant_scope(
            run_cfg=run_cfg,
            replicate=int(replicate),
            cohort=str(cohort_name),
            membership_df=membership_df,
        )
        out_prefix = export_root / f"rep{int(replicate)}_{str(cohort_name)}"
        cache_key = _build_export_cache_key(
            run_cfg,
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            variant_ids=variant_ids,
        )
        tasks.append(
            ExportTask(
                replicate=int(replicate),
                cohort=str(cohort_name),
                sample_ids=sample_ids,
                sample_populations=sample_populations,
                variant_ids=variant_ids,
                out_prefix=out_prefix,
                cache_key=cache_key,
            )
        )

    merge_cfg = SimpleNamespace(
        plink2_path=str(run_cfg.gwas.plink2.path or run_cfg.genotype_input.plink2_path),
        threads=max(1, int(run_cfg.gwas.plink2.threads)),
        concat_workers=max(1, int(run_cfg.pgen_export.concat_workers)),
        pgen_shard_workers=max(1, int(run_cfg.pgen_export.pgen_shard_workers)),
        filter_monomorphic=bool(run_cfg.pgen_export.filter_monomorphic),
        min_maf=float(run_cfg.pgen_export.min_maf),
        shard_tmp_dir=run_cfg.pgen_export.shard_tmp_dir,
    )

    export_cache: dict[str, Any] | None = None
    if runtime_cache is not None and bool(run_cfg.pgen_export.reuse_exports):
        export_cache = runtime_cache.setdefault("pgen_export_cache", {})

    pending_unique: dict[str, ExportTask] = {}
    resolved_artifacts_by_key: dict[str, ExportArtifacts] = {}
    for task in tasks:
        if export_cache is not None and task.cache_key in export_cache:
            cached = export_cache[task.cache_key]
            if isinstance(cached, str):
                cached_artifacts = ExportArtifacts(pgen_prefix=cached, chrom_pgen_manifest={})
            else:
                cached_artifacts = ExportArtifacts(
                    pgen_prefix=str(cached["pgen_prefix"]),
                    chrom_pgen_manifest={
                        int(chrom): str(prefix)
                        for chrom, prefix in dict(cached.get("chrom_pgen_manifest", {})).items()
                    },
                )
            chrom_ready = all(
                _pgen_triplet_exists(prefix)
                for prefix in cached_artifacts.chrom_pgen_manifest.values()
            )
            if _pgen_triplet_exists(cached_artifacts.pgen_prefix) and chrom_ready:
                bump_counter(runtime_cache, hit=True, name="pgen_export")
                resolved_artifacts_by_key[task.cache_key] = cached_artifacts
                continue
        bump_counter(runtime_cache, hit=False, name="pgen_export")
        if task.cache_key not in pending_unique:
            pending_unique[task.cache_key] = task

    pending_tasks = [pending_unique[key] for key in sorted(pending_unique.keys())]
    logger.info(
        "[hapstore-export] planned exports total=%d unique=%d cached=%d",
        len(tasks),
        len(pending_tasks),
        len(tasks) - len(pending_tasks),
    )
    if pending_tasks:
        workers = max(1, int(run_cfg.pgen_export.task_workers))
        if workers == 1 or len(pending_tasks) <= 1:
            for task in pending_tasks:
                logger.info(
                    "[hapstore-export] replicate=%d cohort=%s variants=%s",
                    task.replicate,
                    task.cohort,
                    "all" if task.variant_ids is None else len(task.variant_ids),
                )
                if run_cfg.pgen_export.prep_mode == "legacy_whole_store":
                    prefix = _export_one_task_legacy(task, run_cfg, merge_cfg)
                    resolved_artifacts_by_key[task.cache_key] = ExportArtifacts(
                        pgen_prefix=prefix,
                        chrom_pgen_manifest={},
                    )
                else:
                    resolved_artifacts_by_key[task.cache_key] = _export_one_task_per_chrom(
                        task, run_cfg, merge_cfg
                    )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        _export_one_task_legacy if run_cfg.pgen_export.prep_mode == "legacy_whole_store" else _export_one_task_per_chrom,
                        task,
                        run_cfg,
                        merge_cfg,
                    ): task
                    for task in pending_tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    task = future_map[future]
                    logger.info(
                        "[hapstore-export] replicate=%d cohort=%s variants=%s",
                        task.replicate,
                        task.cohort,
                        "all" if task.variant_ids is None else len(task.variant_ids),
                    )
                    result = future.result()
                    if isinstance(result, str):
                        resolved_artifacts_by_key[task.cache_key] = ExportArtifacts(
                            pgen_prefix=result,
                            chrom_pgen_manifest={},
                        )
                    else:
                        resolved_artifacts_by_key[task.cache_key] = result

    for key, artifacts in resolved_artifacts_by_key.items():
        if export_cache is not None:
            export_cache[key] = {
                "pgen_prefix": artifacts.pgen_prefix,
                "chrom_pgen_manifest": {str(chrom): prefix for chrom, prefix in artifacts.chrom_pgen_manifest.items()},
            }

    for task in tasks:
        artifacts = resolved_artifacts_by_key[task.cache_key]
        exported_prefix = artifacts.pgen_prefix
        mask = (
            (manifest_df["replicate"] == int(task.replicate))
            & (manifest_df["cohort"] == str(task.cohort))
        )
        fid_table_cols = ["keep_file", "phenotype_file", "covariates_file", "global_pcs_file"]
        existing_fid_cols = [col for col in fid_table_cols if col in manifest_df.columns]
        for _, row in manifest_df.loc[mask].iterrows():
            for col in existing_fid_cols:
                value = row[col]
                if pd.isna(value):
                    continue
                path = str(value).strip()
                if not path:
                    continue
                _normalize_fid_for_exported_pgen(path)

        manifest_df.loc[mask, "pgen_prefix"] = str(exported_prefix)
        manifest_df.loc[mask, "exported_pgen_prefix"] = str(exported_prefix)
        manifest_df.loc[mask, "merged_pgen_prefix"] = str(exported_prefix)
        manifest_df.loc[mask, "filtered_pgen_prefix"] = str(exported_prefix)
        manifest_df.loc[mask, "pgen_filter_mode"] = (
            "native_hapstore"
            if bool(run_cfg.pgen_export.filter_monomorphic)
            and run_cfg.pgen_export.prep_mode == "per_chrom_plink_merge"
            else "none"
        )
        if bool(run_cfg.pgen_export.filter_monomorphic):
            manifest_df.loc[mask, "pgen_filter_min_maf"] = float(
                run_cfg.pgen_export.min_maf
            )
        manifest_df.loc[mask, "chrom_pgen_manifest"] = json.dumps(
            {str(chrom): prefix for chrom, prefix in sorted(artifacts.chrom_pgen_manifest.items())},
            sort_keys=True,
        )

    if (
        run_cfg.pgen_export.prep_mode == "legacy_whole_store"
        and bool(run_cfg.pgen_export.filter_monomorphic)
    ):
        manifest_df = ensure_manifest_non_monomorphic_pgens(run_cfg, manifest_df)
        if "pgen_prefix" in manifest_df.columns:
            manifest_df["merged_pgen_prefix"] = manifest_df["pgen_prefix"]

    out_manifest = manifest_path.parent / "run_manifest_pgen.tsv"
    manifest_df.to_csv(out_manifest, sep="\t", index=False)
    if runtime_cache is not None:
        logger.info(
            "[hapstore-export] done hits=%s misses=%s out_manifest=%s",
            runtime_cache.get("cache_hits", {}).get("pgen_export", 0),
            runtime_cache.get("cache_misses", {}).get("pgen_export", 0),
            out_manifest,
        )
    return out_manifest
