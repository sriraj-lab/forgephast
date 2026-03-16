"""HapStore -> PGEN export orchestration for GWAS round-trip workflows."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import CohortConfig, RunConfig
from ..utils.ids import align_by_ids, read_sample_ids
from .genotypes import build_genotype_adapter

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


def _canonical_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _pgen_triplet_exists(prefix: str | Path) -> bool:
    base = Path(prefix)
    return all(Path(f"{base}{ext}").exists() for ext in (".pgen", ".pvar", ".psam"))


def _normalize_fid_for_exported_pgen(path: str) -> None:
    """Set FID column to 0 for tables consumed by PLINK/REGENIE on exported PGEN."""
    table_path = Path(path)
    if not table_path.exists():
        return
    df = pd.read_csv(table_path, sep="\t")
    if "FID" not in df.columns:
        return
    df["FID"] = "0"
    df.to_csv(table_path, sep="\t", index=False)


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
    return _canonical_hash(
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
            "concat_workers": int(run_cfg.pgen_export.concat_workers),
            "pgen_shard_workers": int(run_cfg.pgen_export.pgen_shard_workers),
        }
    )


def _bump(runtime_cache: dict[str, Any] | None, *, hit: bool) -> None:
    if runtime_cache is None:
        return
    key = "cache_hits" if hit else "cache_misses"
    counters = runtime_cache.setdefault(key, {})
    counters["pgen_export"] = int(counters.get("pgen_export", 0)) + 1


def _export_one_task(task: ExportTask, run_cfg: RunConfig, merge_cfg: SimpleNamespace) -> str:
    adapter = build_genotype_adapter(
        genotype_format=run_cfg.genotype_input.format,
        pgen_prefix=None,
        plink2_path=run_cfg.genotype_input.plink2_path,
        hapstore_cfg=run_cfg.genotype_input.hapstore,
        pgen_dosage_backend=run_cfg.genotype_input.pgen_dosage_backend,
    )
    adapter.validate_files()
    return adapter.export_to_pgen(
        sample_ids=task.sample_ids,
        sample_populations=task.sample_populations,
        out_prefix=task.out_prefix,
        chromosomes=run_cfg.chromosomes,
        variant_ids=task.variant_ids,
        export_cfg=merge_cfg,
    )


def export_manifest_pgens(
    run_cfg: RunConfig,
    manifest_path: Path,
    runtime_cache: dict[str, Any] | None = None,
) -> Path:
    """Export HapStore cohorts to PGEN and emit updated manifest path."""
    if run_cfg.genotype_input.format != "hapstore":
        return manifest_path
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
        "[hapstore-export] start rows=%d variant_scope=%s task_workers=%d reuse_exports=%s",
        int(len(manifest_df)),
        str(run_cfg.pgen_export.variant_scope),
        int(run_cfg.pgen_export.task_workers),
        bool(run_cfg.pgen_export.reuse_exports),
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
    )

    export_cache: dict[str, str] | None = None
    if runtime_cache is not None and bool(run_cfg.pgen_export.reuse_exports):
        export_cache = runtime_cache.setdefault("pgen_export_cache", {})

    pending_unique: dict[str, ExportTask] = {}
    resolved_prefix_by_key: dict[str, str] = {}
    for task in tasks:
        if export_cache is not None and task.cache_key in export_cache:
            cached_prefix = export_cache[task.cache_key]
            if _pgen_triplet_exists(cached_prefix):
                _bump(runtime_cache, hit=True)
                resolved_prefix_by_key[task.cache_key] = cached_prefix
                continue
        _bump(runtime_cache, hit=False)
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
                resolved_prefix_by_key[task.cache_key] = _export_one_task(task, run_cfg, merge_cfg)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_export_one_task, task, run_cfg, merge_cfg): task
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
                    resolved_prefix_by_key[task.cache_key] = future.result()

    for key, prefix in resolved_prefix_by_key.items():
        if export_cache is not None:
            export_cache[key] = prefix

    for task in tasks:
        exported_prefix = resolved_prefix_by_key[task.cache_key]
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
