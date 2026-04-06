#!/usr/bin/env python3
"""CLI for polygenic phenotype simulation, GWAS orchestration, and grid runs.

Subcommands
-----------
simulate  Run phenotype simulation stage only.
gwas      Run GWAS stage from an existing simulation manifest.
full      Run simulate + GWAS end-to-end.
grid      Run a parameter-grid sweep.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

from .._internal.cache import bump_counter, canonical_hash, counter_delta, snapshot_counters
from .config import RunConfig
from .data.genotypes import build_genotype_adapter
from .data.global_pcs import refresh_manifest_global_pcs
from .data.pgen_export import ensure_manifest_non_monomorphic_pgens, export_manifest_pgens
from .grid import (
    build_grouped_cases,
    execute_group,
    expand_grid_cases,
    load_grid_spec,
    write_grid_manifest,
)
from .gwas import GwasTask
from .gwas.base import prepare_manifest_covariates_for_gwas
from .gwas.registry import available_engine_names, build_engines
from .io.write_outputs import write_simulation_outputs
from .phenotypes.simulate import run_simulation
from .utils.logging import setup_logging
from .validation import build_roundtrip_plots, build_roundtrip_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> RunConfig:
    """Load validated run config."""
    config = RunConfig.from_yaml(config_path)
    config.paths.create_directories()
    return config


def _write_stage_timing(
    run_dir: Path,
    command: str,
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Write stage timing rows for one run."""
    truth_dir = run_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    out_path = truth_dir / f"stage_timing_{command}.tsv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, sep="\t", index=False)
    total_elapsed = (
        float(frame["elapsed_seconds"].sum())
        if "elapsed_seconds" in frame.columns and not frame.empty
        else 0.0
    )
    logger.info(
        "STAGE_TIMING_WRITTEN command=%s rows=%d total_elapsed_seconds=%.3f path=%s",
        command,
        int(len(frame)),
        total_elapsed,
        out_path,
    )
    return out_path


def _pgen_triplet_exists(prefix: str | Path) -> bool:
    base = Path(prefix)
    return all(Path(f"{base}{ext}").exists() for ext in (".pgen", ".pvar", ".psam"))


def _manifest_has_ready_pgens(manifest_df: pd.DataFrame) -> bool:
    if "pgen_prefix" not in manifest_df.columns:
        return False
    prefixes = manifest_df["pgen_prefix"].fillna("").astype(str).str.strip()
    if prefixes.empty or (prefixes == "").any():
        return False
    return all(_pgen_triplet_exists(prefix) for prefix in prefixes.tolist())


def _manifest_has_ready_global_pcs(manifest_df: pd.DataFrame) -> bool:
    if "global_pcs_file" not in manifest_df.columns:
        return False
    paths = manifest_df["global_pcs_file"].fillna("").astype(str).str.strip()
    if paths.empty or (paths == "").any():
        return False
    return all(Path(path).exists() for path in paths.tolist())


def _manifest_has_ready_merged_covariates(manifest_df: pd.DataFrame) -> bool:
    if "merged_covariates_file" not in manifest_df.columns:
        return False
    paths = manifest_df["merged_covariates_file"].fillna("").astype(str).str.strip()
    if paths.empty or (paths == "").any():
        return False
    return all(Path(path).exists() for path in paths.tolist())



def _get_adapter(
    config: RunConfig,
    *,
    runtime_state: dict[str, Any] | None,
) -> object:
    """Return an adapter, reusing one in-memory for compatible configs."""
    if runtime_state is None:
        adapter = build_genotype_adapter(
            hapstore_cfg=config.genotype_input.hapstore,
        )
        adapter.validate_files()
        return adapter

    hap = config.genotype_input.hapstore
    key = canonical_hash(
        {
            "hapstore_layout": hap.layout,
            "hapstore_root": hap.root_dir,
            "hapstore_file_pattern": hap.file_pattern,
            "hapstore_store_path": hap.store_path,
        }
    )
    cache = runtime_state.setdefault("adapter_cache", {})
    if key in cache:
        bump_counter(runtime_state, hit=True, name="adapter")
        return cache[key]

    bump_counter(runtime_state, hit=False, name="adapter")
    adapter = build_genotype_adapter(
        hapstore_cfg=config.genotype_input.hapstore,
    )
    adapter.validate_files()
    cache[key] = adapter
    return adapter


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_simulate(
    config: RunConfig,
    runtime_state: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Execute simulation stage and return run manifest path + stage timing rows."""
    run_dir = Path(config.paths.output_root) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict[str, Any]] = []
    logger.info("SIMULATE_START run=%s run_dir=%s", config.name, run_dir)

    adapter = _get_adapter(config, runtime_state=runtime_state)

    logger.info("[simulate] Starting simulation for run=%s", config.name)
    t1 = time.perf_counter()
    records = run_simulation(
        run_cfg=config,
        run_dir=run_dir,
        adapter=adapter,
        runtime_cache=runtime_state,
        timing_rows=stage_rows,
    )
    stage_rows.append(
        {
            "stage": "simulate_records",
            "elapsed_seconds": float(time.perf_counter() - t1),
            "n_records": int(len(records)),
        }
    )
    logger.info(
        "SIMULATE_STAGE_DONE run=%s stage=simulate_records n_records=%d",
        config.name,
        int(len(records)),
    )

    t2 = time.perf_counter()
    manifest = write_simulation_outputs(
        run_name=config.name,
        output_root=Path(config.paths.output_root),
        records=records,
    )
    manifest_path = run_dir / "run_manifest.tsv"
    stage_rows.append(
        {
            "stage": "write_outputs",
            "elapsed_seconds": float(time.perf_counter() - t2),
            "n_manifest_rows": int(len(manifest)),
        }
    )
    logger.info(
        "SIMULATE_STAGE_DONE run=%s stage=write_outputs n_manifest_rows=%d",
        config.name,
        int(len(manifest)),
    )

    logger.info("[simulate] completed successfully")
    logger.info("[simulate] records=%d", len(manifest))
    logger.info("[simulate] manifest=%s", manifest_path)
    return manifest_path, stage_rows


def _make_engines(config: RunConfig) -> Dict[str, object]:
    """Create configured GWAS engine objects."""
    return dict(build_engines(config))


def _manifest_to_tasks(manifest_df: pd.DataFrame, chromosomes: List[int]) -> List[GwasTask]:
    """Convert manifest dataframe into GWAS task objects."""
    tasks = []
    for _, row in manifest_df.iterrows():
        pgen_prefix = str(row["pgen_prefix"])
        if not pgen_prefix:
            raise ValueError(
                "run manifest row has empty pgen_prefix. For HapStore input, run export stage first."
            )
        tasks.append(
            GwasTask(
                run_name=str(row["run_name"]),
                replicate=int(row["replicate"]),
                architecture=str(row["architecture"]),
                cohort=str(row["cohort"]),
                pgen_prefix=pgen_prefix,
                keep_file=str(row["keep_file"]),
                phenotype_file=str(row["phenotype_file"]),
                covariates_file=str(row["covariates_file"]),
                global_pcs_file=str(row["global_pcs_file"]),
                output_dir=str(row["output_dir"]),
                merged_covariates_file=str(row.get("merged_covariates_file", "")),
                chromosomes=list(chromosomes),
            )
        )
    return tasks


def _run_gwas(
    config: RunConfig,
    manifest_path: Path,
    engines_filter: List[str] | None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Execute GWAS stage from run manifest and return summary path + stage timing rows."""
    run_dir = Path(config.paths.output_root) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict[str, Any]] = []
    logger.info("GWAS_START run=%s manifest=%s", config.name, manifest_path)

    logger.info("[gwas] loading manifest: %s", manifest_path)
    t0 = time.perf_counter()
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    tasks = _manifest_to_tasks(manifest_df, chromosomes=config.chromosomes)
    if not tasks:
        raise ValueError("run manifest contains zero tasks")
    stage_rows.append(
        {
            "stage": "load_gwas_manifest",
            "elapsed_seconds": float(time.perf_counter() - t0),
            "n_tasks": int(len(tasks)),
        }
    )
    logger.info("GWAS_STAGE_DONE run=%s stage=load_gwas_manifest n_tasks=%d", config.name, len(tasks))

    engines = _make_engines(config)
    if engines_filter:
        engines = {k: v for k, v in engines.items() if k in set(engines_filter)}
        if not engines:
            raise ValueError(f"No engines remain after filter={engines_filter}")

    prepare_t0 = time.perf_counter()
    engine_prepared_state: dict[str, Any] = {}
    for engine_name, engine in engines.items():
        batch_dir = run_dir / "_gwas_engine_state" / engine_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        engine_prepared_state[engine_name] = engine.prepare_batch(tasks, batch_dir)
    stage_rows.append(
        {
            "stage": "prepare_gwas_engines",
            "elapsed_seconds": float(time.perf_counter() - prepare_t0),
            "n_engines": int(len(engines)),
        }
    )
    logger.info("GWAS_STAGE_DONE run=%s stage=prepare_gwas_engines n_engines=%d", config.name, len(engines))

    jobs: list[tuple[int, GwasTask, str, object]] = []
    idx = 0
    for task in tasks:
        for engine_name, engine in engines.items():
            jobs.append((idx, task, engine_name, engine))
            idx += 1

    for task, engine_name in [(task, engine_name) for task in tasks for engine_name in engines]:
        task_out_dir = Path(task.output_dir) / "gwas" / engine_name
        logger.info(
            "GWAS_MONITOR run=%s rep=%d arch=%s cohort=%s engine=%s output_dir=%s",
            task.run_name,
            int(task.replicate),
            task.architecture,
            task.cohort,
            engine_name,
            task_out_dir,
        )

    def _run_one(job: tuple[int, GwasTask, str, object]) -> tuple[int, dict[str, Any]]:
        job_idx, task, engine_name, engine = job
        task_out_dir = Path(task.output_dir) / "gwas" / engine_name
        task_out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[gwas] run=%s rep=%d arch=%s cohort=%s engine=%s",
            task.run_name,
            task.replicate,
            task.architecture,
            task.cohort,
            engine_name,
        )
        try:
            result = engine.run(
                task=task,
                run_dir=task_out_dir,
                prepared_state=engine_prepared_state.get(engine_name),
            )
            row = {
                "run_name": task.run_name,
                "replicate": task.replicate,
                "architecture": task.architecture,
                "cohort": task.cohort,
                "engine": engine_name,
                "status": "success",
                "output_dir": str(result.output_dir),
                "primary_output": str(result.primary_output),
            }
            row.update(dict(result.metadata))
            logger.info(
                "GWAS_JOB_DONE run=%s rep=%d arch=%s cohort=%s engine=%s output_dir=%s primary_output=%s",
                task.run_name,
                int(task.replicate),
                task.architecture,
                task.cohort,
                engine_name,
                result.output_dir,
                result.primary_output,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[gwas] failure for engine=%s task=%s", engine_name, task)
            row = {
                "run_name": task.run_name,
                "replicate": task.replicate,
                "architecture": task.architecture,
                "cohort": task.cohort,
                "engine": engine_name,
                "status": "failed",
                "output_dir": str(task_out_dir),
                "primary_output": "",
                "error": str(exc),
            }
        return job_idx, row

    t1 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    workers = max(1, int(config.gwas.task_workers))
    logger.info("GWAS_EXECUTION_PLAN run=%s n_jobs=%d workers=%d", config.name, len(jobs), workers)
    if workers == 1 or len(jobs) <= 1:
        for job in jobs:
            _, row = _run_one(job)
            rows.append(row)
    else:
        ordered: dict[int, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                job_idx, row = future.result()
                ordered[job_idx] = row
        rows = [ordered[i] for i in sorted(ordered)]
    stage_rows.append(
        {
            "stage": "gwas_jobs",
            "elapsed_seconds": float(time.perf_counter() - t1),
            "n_jobs": int(len(jobs)),
            "task_workers": int(workers),
        }
    )
    logger.info("GWAS_STAGE_DONE run=%s stage=gwas_jobs n_jobs=%d", config.name, len(jobs))

    gwas_summary = pd.DataFrame(rows)
    out_path = run_dir / "gwas_manifest.tsv"
    gwas_summary.to_csv(out_path, sep="\t", index=False)

    failures = int((gwas_summary["status"] == "failed").sum())
    logger.info("[gwas] completed with %d failures; summary=%s", failures, out_path)
    if failures:
        raise RuntimeError(f"GWAS stage finished with {failures} failed task(s)")
    return out_path, stage_rows


def _prepare_manifest_for_gwas(
    config: RunConfig,
    manifest_path: Path,
    runtime_state: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Prepare manifest for GWAS (HapStore export + global PCA refresh)."""
    stage_rows: list[dict[str, Any]] = []
    current = manifest_path
    manifest_df = pd.read_csv(manifest_path, sep="\t")

    if _manifest_has_ready_pgens(manifest_df):
        logger.info("[prep] detected ready PGEN prefixes in manifest=%s; skipping HapStore export", manifest_path)
    else:
        t0 = time.perf_counter()
        logger.info("[prep] exporting HapStore cohorts to PGEN")
        current = export_manifest_pgens(config, current, runtime_cache=runtime_state)
        stage_rows.append(
            {
                "stage": "hapstore_export",
                "elapsed_seconds": float(time.perf_counter() - t0),
            }
        )
        logger.info("PREP_STAGE_DONE run=%s stage=hapstore_export", config.name)
        manifest_df = pd.read_csv(current, sep="\t")
    t_filter = time.perf_counter()
    filtered_manifest_df = ensure_manifest_non_monomorphic_pgens(config, manifest_df)
    if not filtered_manifest_df.equals(manifest_df):
        current = current.parent / "run_manifest_pgen.tsv"
        filtered_manifest_df.to_csv(current, sep="\t", index=False)
        manifest_df = filtered_manifest_df
        stage_rows.append(
            {
                "stage": "ensure_non_monomorphic_pgen",
                "elapsed_seconds": float(time.perf_counter() - t_filter),
            }
        )
        logger.info("PREP_STAGE_DONE run=%s stage=ensure_non_monomorphic_pgen", config.name)
    if config.gwas.global_pcs.canonical_mode() in {"plink2_pca", "precomputed"}:
        if _manifest_has_ready_global_pcs(manifest_df):
            logger.info("[prep] detected ready global PCs in manifest=%s; skipping PCA refresh", current)
        else:
            t1 = time.perf_counter()
            logger.info("[prep] refreshing global PCs from manifest PGENs")
            current = refresh_manifest_global_pcs(config, current, runtime_cache=runtime_state)
            stage_rows.append(
                {
                    "stage": "refresh_global_pcs",
                    "elapsed_seconds": float(time.perf_counter() - t1),
                }
            )
            logger.info("PREP_STAGE_DONE run=%s stage=refresh_global_pcs", config.name)
            manifest_df = pd.read_csv(current, sep="\t")
    if _manifest_has_ready_merged_covariates(manifest_df):
        logger.info("[prep] detected ready merged GWAS covariates in manifest=%s; skipping covariate prep", current)
    else:
        t2 = time.perf_counter()
        logger.info("[prep] preparing one-time merged GWAS covariates")
        manifest_df = prepare_manifest_covariates_for_gwas(manifest_df)
        current = current.parent / current.name.replace(".tsv", "_prepared.tsv")
        manifest_df.to_csv(current, sep="\t", index=False)
        stage_rows.append(
            {
                "stage": "prepare_gwas_covariates",
                "elapsed_seconds": float(time.perf_counter() - t2),
            }
        )
        logger.info("PREP_STAGE_DONE run=%s stage=prepare_gwas_covariates", config.name)
    return current, stage_rows


def _run_roundtrip_validation(config: RunConfig, gwas_manifest_path: Path) -> Path | None:
    """Run optional GWAS-vs-truth calibration checks."""
    if not config.roundtrip_validation.enabled:
        return None
    run_dir = Path(config.paths.output_root) / config.name
    return build_roundtrip_report(
        run_cfg=config,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )


def _run_roundtrip_plots(
    config: RunConfig,
    gwas_manifest_path: Path,
    *,
    run_dir: Path | None = None,
    force: bool = False,
) -> Path | None:
    """Run optional or explicitly requested round-trip diagnostic plotting."""
    if not force and not config.roundtrip_plots.enabled:
        return None
    if run_dir is None:
        run_dir = Path(config.paths.output_root) / config.name
    return build_roundtrip_plots(
        run_cfg=config,
        run_dir=run_dir,
        gwas_manifest_path=gwas_manifest_path,
    )


# ---------------------------------------------------------------------------
# Unified pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    config: RunConfig,
    stages: list[str],
    *,
    engines_filter: list[str] | None = None,
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one or more pipeline stages in sequence.

    Parameters
    ----------
    config:
        Validated run configuration.
    stages:
        Ordered list of stages to execute (``"simulate"``, ``"gwas"``).
    engines_filter:
        Optional GWAS engine name filter.
    runtime_state:
        Mutable dict for adapter/cache reuse across grid cases.

    Returns
    -------
    dict with keys: ``manifest_path``, ``prepared_manifest``, ``gwas_manifest``,
    ``validation_report``, ``roundtrip_plot_manifest``, ``timing_rows``, ``timing_path``.
    """
    run_dir = Path(config.paths.output_root) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    if "simulate" in stages:
        manifest_path, sim_rows = _run_simulate(config, runtime_state=runtime_state)
        timing_rows.extend(sim_rows)
        result["manifest_path"] = manifest_path

    if "gwas" in stages:
        manifest_path = result.get("manifest_path") or (
            Path(config.paths.output_root) / config.name / "run_manifest.tsv"
        )
        prepared_path, prep_rows = _prepare_manifest_for_gwas(
            config, manifest_path, runtime_state=runtime_state
        )
        timing_rows.extend(prep_rows)
        result["prepared_manifest"] = prepared_path

        gwas_path, gwas_rows = _run_gwas(config, prepared_path, engines_filter=engines_filter)
        timing_rows.extend(gwas_rows)
        result["gwas_manifest"] = gwas_path

        t0 = time.perf_counter()
        validation = _run_roundtrip_validation(config, gwas_path)
        timing_rows.append({
            "stage": "roundtrip_validation",
            "elapsed_seconds": float(time.perf_counter() - t0),
            "enabled": bool(config.roundtrip_validation.enabled),
        })
        if validation is not None:
            result["validation_report"] = validation

        t1 = time.perf_counter()
        plot_manifest = _run_roundtrip_plots(config, gwas_path)
        timing_rows.append({
            "stage": "roundtrip_plots",
            "elapsed_seconds": float(time.perf_counter() - t1),
            "enabled": bool(config.roundtrip_plots.enabled),
        })
        if plot_manifest is not None:
            result["roundtrip_plot_manifest"] = plot_manifest

    result["timing_rows"] = timing_rows
    stage_label = "+".join(stages)
    timing_path = _write_stage_timing(run_dir, stage_label, timing_rows)
    result["timing_path"] = timing_path
    return result


def _run_case(
    payload: dict[str, Any],
    stage: str,
    runtime_state: dict[str, Any],
    case_index: int,
    engines_filter: List[str] | None = None,
) -> dict[str, Any]:
    """Run one grid case and return summary metadata."""
    config = RunConfig.from_dict(copy.deepcopy(payload))
    config.paths.create_directories()
    run_dir = Path(config.paths.output_root) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    before_hits, before_misses = snapshot_counters(runtime_state)
    logger.info("GRID_CASE_START run=%s case_index=%d stage=%s", config.name, int(case_index), stage)

    stage_map = {"simulate": ["simulate"], "full": ["simulate", "gwas"]}
    if stage not in stage_map:
        raise ValueError(f"Unsupported grid stage: {stage}")

    result = _run_pipeline(
        config,
        stage_map[stage],
        engines_filter=engines_filter,
        runtime_state=runtime_state,
    )

    after_hits, after_misses = snapshot_counters(runtime_state)
    delta_hits = counter_delta(after_hits, before_hits)
    delta_misses = counter_delta(after_misses, before_misses)
    logger.info(
        "GRID_CASE_DONE run=%s case_index=%d stage=%s timing=%s cache_hits_delta=%s cache_misses_delta=%s",
        config.name,
        int(case_index),
        stage,
        result.get("timing_path", ""),
        json.dumps(delta_hits, sort_keys=True),
        json.dumps(delta_misses, sort_keys=True),
    )
    return {
        "run_name": config.name,
        "run_dir": str(run_dir),
        "manifest_path": str(result.get("manifest_path", "")),
        "prepared_manifest_path": str(result.get("prepared_manifest", "")),
        "gwas_manifest_path": str(result.get("gwas_manifest", "")),
        "validation_report": str(result.get("validation_report", "")),
        "roundtrip_plot_manifest": str(result.get("roundtrip_plot_manifest", "")),
        "timing_file": str(result.get("timing_path", "")),
        "cache_hits": json.dumps(runtime_state.get("cache_hits", {}), sort_keys=True),
        "cache_misses": json.dumps(runtime_state.get("cache_misses", {}), sort_keys=True),
    }


def _load_base_payload_from_grid(spec_path: str | Path) -> tuple[dict[str, Any], str]:
    """Load base payload using a grid spec path."""
    spec = load_grid_spec(spec_path)
    spec_dir = Path(spec_path).resolve().parent
    base_config_path = Path(spec.base_config)
    if not base_config_path.is_absolute():
        base_config_path = (spec_dir / base_config_path).resolve()
    with open(base_config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Base config payload must be a mapping")
    return payload, str(base_config_path)


def _run_grid(grid_path: str, engines_filter: List[str] | None) -> Path:
    """Execute a grid spec and return aggregate manifest path."""
    spec = load_grid_spec(grid_path)
    base_payload, resolved_base_config = _load_base_payload_from_grid(grid_path)
    cases = expand_grid_cases(base_payload=base_payload, factors=spec.factors)
    groups = build_grouped_cases(cases)

    logger.info("grid spec=%s", grid_path)
    logger.info("grid base_config=%s", resolved_base_config)
    logger.info("grid stage=%s", spec.stage)
    logger.info("grid cases=%d groups=%d", len(cases), len(groups))
    logger.info("grid parallel_runs=%d continue_on_error=%s", spec.parallel_runs, spec.continue_on_error)

    rows: list[dict[str, Any]] = []

    def _group_weight(group_cases: list[Any]) -> float:
        try:
            overrides = getattr(group_cases[0], "overrides", {}) or {}
            fixed = None
            for path, value in overrides.items():
                if "causal_selection" in str(path) and str(path).endswith("fixed_count"):
                    fixed = int(value)
                    break
            if fixed is None:
                fixed = 1
            fixed = max(1, fixed)
            return float(math.log10(float(fixed)) ** 1.8)
        except Exception:
            return 1.0

    def _allocate_budgets(total: int, weights: list[float], *, min_each: int = 1) -> list[int]:
        n = len(weights)
        if n == 0:
            return []
        total = max(n * min_each, int(total))
        base = [min_each] * n
        remaining = int(total - n * min_each)
        if remaining <= 0:
            return base
        w = [max(0.0, float(x)) for x in weights]
        if sum(w) <= 0:
            w = [1.0] * n
        shares = [remaining * (x / sum(w)) for x in w]
        floors = [int(math.floor(s)) for s in shares]
        frac = [s - f for s, f in zip(shares, floors)]
        out = [b + f for b, f in zip(base, floors)]
        left = remaining - sum(floors)
        order = sorted(range(n), key=lambda i: (-frac[i], i))
        for i in order[:left]:
            out[i] += 1
        return out

    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))
    budgets_enabled = str(os.environ.get("POLY_GRID_GROUP_CPU_BUDGETS", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if budgets_enabled:
        weights = [_group_weight(group) for group in groups]
        hapstore_budgets: list[int | None] = _allocate_budgets(total_cpus, weights, min_each=1)
        logger.info(
            "grid hapstore_io_threads enabled=1 total_cpus=%d budgets=%s",
            total_cpus,
            hapstore_budgets,
        )
    else:
        hapstore_budgets = [None] * len(groups)
        logger.info("grid hapstore_io_threads enabled=0")

    def _runner(group_cases: list[Any], hapstore_io_threads: int | None) -> list[dict[str, Any]]:
        return execute_group(
            group_cases,
            stage=spec.stage,
            max_cache_gb=spec.runtime.max_cache_gb,
            run_case=lambda payload, stage, runtime_state, case_index: _run_case(
                payload=payload,
                stage=stage,
                runtime_state=runtime_state,
                case_index=case_index,
                engines_filter=engines_filter,
            ),
            continue_on_error=spec.continue_on_error,
            hapstore_io_threads=(None if hapstore_io_threads is None else int(hapstore_io_threads)),
        )

    if spec.parallel_runs <= 1 or len(groups) <= 1:
        for idx, group in enumerate(groups):
            rows.extend(_runner(group, hapstore_budgets[idx]))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(spec.parallel_runs)) as executor:
            futures = [
                executor.submit(_runner, group, hapstore_budgets[idx])
                for idx, group in enumerate(groups)
            ]
            for future in concurrent.futures.as_completed(futures):
                rows.extend(future.result())

    out_path = write_grid_manifest(
        base_output_root=str(base_payload["paths"]["output_root"]),
        base_name=str(base_payload.get("name", "run")),
        rows=rows,
    )
    failures = sum(1 for row in rows if row.get("status") == "failed")
    total_elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in rows)
    logger.info("GRID_COMPLETED rows=%d failures=%d manifest=%s", len(rows), failures, out_path)
    logger.info("GRID_TIMING_SUMMARY rows=%d elapsed_seconds=%.3f", len(rows), float(total_elapsed))
    if failures and not spec.continue_on_error:
        raise RuntimeError(f"Grid finished with {failures} failed case(s)")
    return out_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Build CLI parser and parse arguments."""
    engine_choices = available_engine_names()
    parser = argparse.ArgumentParser(
        prog="forgephast-phenotype",
        description="Polygenic phenotype simulation pipeline (forgephast)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="Run simulation stage")
    simulate.add_argument("--config", required=True, help="Path to run config YAML")

    gwas = sub.add_parser("gwas", help="Run GWAS stage")
    gwas.add_argument("--config", required=True, help="Path to run config YAML")
    gwas.add_argument(
        "--manifest",
        default=None,
        help="Optional simulation manifest path (default: {output_root}/{run_name}/run_manifest.tsv)",
    )
    gwas.add_argument(
        "--engine",
        choices=engine_choices,
        action="append",
        help="Optional engine filter; can be passed multiple times",
    )

    full = sub.add_parser("full", help="Run simulate + gwas")
    full.add_argument("--config", required=True, help="Path to run config YAML")
    full.add_argument(
        "--engine",
        choices=engine_choices,
        action="append",
        help="Optional GWAS engine filter",
    )

    grid = sub.add_parser("grid", help="Run a parameter grid")
    grid.add_argument("--grid", required=True, help="Path to grid spec YAML")
    grid.add_argument(
        "--engine",
        choices=engine_choices,
        action="append",
        help="Optional GWAS engine filter (only for stage=full grids)",
    )

    rt_plots = sub.add_parser("roundtrip-plots", help="Render round-trip plots from existing GWAS outputs")
    rt_plots.add_argument("--config", required=True, help="Path to run config YAML")
    rt_plots.add_argument(
        "--gwas-manifest",
        default=None,
        help="Optional GWAS manifest path (default: {output_root}/{run_name}/gwas_manifest.tsv)",
    )
    rt_plots.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory override (default: {output_root}/{run_name})",
    )
    rt_plots.add_argument(
        "--sample-group-file",
        default=None,
        help="Optional TSV with sample-to-group assignments for group-stratified scatter plots",
    )
    rt_plots.add_argument(
        "--id-column",
        default=None,
        help="Override the sample ID column name in --sample-group-file",
    )
    rt_plots.add_argument(
        "--group-column",
        default=None,
        help="Override the grouping column name in --sample-group-file",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for ``forgephast-phenotype``."""
    args = _parse_args()

    if args.command == "grid":
        base_payload, _ = _load_base_payload_from_grid(args.grid)
        log_dir = Path(str(base_payload["paths"]["output_root"])) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{str(base_payload.get('name', 'run'))}_grid.log"
        setup_logging(log_file)
        out = _run_grid(args.grid, engines_filter=args.engine)
        logger.info("GRID_COMPLETED manifest=%s", out)
        return

    config = _load_config(args.config)
    log_file = Path(config.paths.log_dir) / f"{config.name}_{args.command}.log"
    setup_logging(log_file)
    logger.info("pipeline command=%s", args.command)
    logger.info("config=%s", args.config)

    if args.command == "simulate":
        result = _run_pipeline(config, ["simulate"])
        logger.info("SIMULATE_TIMINGS %s", result["timing_path"])
        logger.info("SIMULATE_COMPLETED manifest=%s", result["manifest_path"])
        return

    if args.command == "gwas":
        # For gwas-only, an explicit --manifest overrides the default path.
        if args.manifest:
            # Seed the manifest_path so _run_pipeline picks it up for the gwas stage.
            run_dir = Path(config.paths.output_root) / config.name
            run_dir.mkdir(parents=True, exist_ok=True)
            # Copy the explicit manifest into the default location so the gwas
            # stage finds it without simulate having run.
            import shutil

            default_manifest = run_dir / "run_manifest.tsv"
            src = Path(args.manifest)
            if src.resolve() != default_manifest.resolve():
                shutil.copy2(src, default_manifest)
        result = _run_pipeline(config, ["gwas"], engines_filter=args.engine)
        logger.info("GWAS_TIMINGS %s", result["timing_path"])
        if result.get("validation_report") is not None:
            logger.info("GWAS_VALIDATION_COMPLETED report=%s", result["validation_report"])
        logger.info("GWAS_COMPLETED summary=%s", result["gwas_manifest"])
        return

    if args.command == "full":
        result = _run_pipeline(config, ["simulate", "gwas"], engines_filter=args.engine)
        logger.info("FULL_TIMINGS %s", result["timing_path"])
        if result.get("validation_report") is not None:
            logger.info("FULL_VALIDATION_COMPLETED report=%s", result["validation_report"])
        if result.get("roundtrip_plot_manifest") is not None:
            logger.info("FULL_ROUNDTRIP_PLOTS_COMPLETED manifest=%s", result["roundtrip_plot_manifest"])
        logger.info(
            "FULL_COMPLETED manifest=%s prepared_manifest=%s gwas_summary=%s",
            result["manifest_path"],
            result["prepared_manifest"],
            result["gwas_manifest"],
        )
        return

    if args.command == "roundtrip-plots":
        if args.sample_group_file:
            config.roundtrip_plots.sample_group_file = str(args.sample_group_file)
        if args.id_column:
            config.roundtrip_plots.id_column = str(args.id_column)
        if args.group_column:
            config.roundtrip_plots.group_column = str(args.group_column)
        config.roundtrip_plots.validate()
        run_dir = Path(args.run_dir) if args.run_dir else Path(config.paths.output_root) / config.name
        gwas_manifest = Path(args.gwas_manifest) if args.gwas_manifest else run_dir / "gwas_manifest.tsv"
        out = _run_roundtrip_plots(config, gwas_manifest, run_dir=run_dir, force=True)
        logger.info("ROUNDTRIP_PLOTS_COMPLETED manifest=%s", out)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
