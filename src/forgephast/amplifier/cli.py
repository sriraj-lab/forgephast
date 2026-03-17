"""Command-line interface for population-amplifier."""

from __future__ import annotations

import argparse
import json
import logging

from .api import (
    build_cache,
    build_manifest,
    merge,
    qc,
    qc_donor_round_robin,
    simulate,
)
from .utils import resolve_log_file_path, setup_logging

logger = logging.getLogger(__name__)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="population-amplifier CLI")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Python log level (DEBUG/INFO/WARNING/ERROR/CRITICAL). Overrides POPAMP_LOG_LEVEL.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit path for dedicated Python run log file.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory for auto-named Python run logs.",
    )
    parser.add_argument(
        "--disable-file-log",
        action="store_true",
        help="Disable dedicated Python file logging for this invocation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("build-manifest", help="Build donor manifests")
    p_manifest.add_argument("--config", required=True)

    p_cache = sub.add_parser("build-cache", help="Stage 0: build donor HapStore caches")
    p_cache.add_argument("--config", required=True)
    p_cache.add_argument("--chrom", type=int, action="append", default=None)
    p_cache.add_argument("--force", action="store_true")

    p_sim = sub.add_parser(
        "simulate",
        help="Simulate a chromosome with the shard-pool engine (`n_shards=1` preserves single-shard behavior)",
    )
    p_sim.add_argument("--config", required=True)
    p_sim.add_argument("--chrom", required=True, type=int)
    p_sim.add_argument("--n-shards", type=int, default=None)
    p_sim.add_argument("--workers", type=int, default=None)
    p_sim.add_argument("--threads-per-task", type=int, default=None)
    p_sim.add_argument("--task-start-method", choices=["fork", "spawn", "forkserver"], default=None)

    p_merge = sub.add_parser("merge", help="Merge shard outputs for a chromosome")
    p_merge.add_argument("--config", required=True)
    p_merge.add_argument(
        "--input-root",
        default=None,
        help="Optional run root to read shard inputs from (synthetic/ + manifests/). "
        "Outputs are still written under the run root implied by --config.",
    )
    p_merge.add_argument("--chrom", required=True, type=int)
    p_merge.add_argument("--n-shards", required=True, type=int)
    p_merge.add_argument("--merge-combined", dest="merge_combined", action="store_true")
    p_merge.add_argument("--no-merge-combined", dest="merge_combined", action="store_false")
    p_merge.set_defaults(merge_combined=None)

    p_qc = sub.add_parser("qc", help="Compute QC metrics and gating diagnostics")
    p_qc.add_argument("--config", required=True)
    p_qc.add_argument("--chrom", required=True, type=int)
    p_qc.add_argument("--n-shards", type=int, default=1)

    p_qc_rr = sub.add_parser(
        "qc-donor-round-robin",
        help="Compute within-pop donor FST spread from round-robin folds sized by fold fraction",
    )
    p_qc_rr.add_argument("--config", required=True)
    p_qc_rr.add_argument("--chrom", required=True, type=int)
    p_qc_rr.add_argument("--fold-fraction", type=float, default=0.2)
    p_qc_rr.add_argument("--seed", type=int, default=42)
    p_qc_rr.add_argument("--workers", type=int, default=1)
    p_qc_rr.add_argument("--population", action="append", default=None)

    return parser.parse_args()

def _resolve_log_level(value: str | None) -> int | None:
    if value is None:
        return None
    val = str(value).strip()
    if not val:
        return None
    upper = val.upper()
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    if upper in mapping:
        return mapping[upper]
    try:
        return int(upper)
    except Exception as exc:
        raise ValueError(f"Invalid --log-level: {value!r}") from exc



def main() -> None:
    args = _parse_args()
    config_path = getattr(args, "config", None)
    log_file = resolve_log_file_path(
        command=args.command,
        config_path=config_path,
        explicit_log_file=args.log_file,
        explicit_log_dir=args.log_dir,
        disable_file_logging=args.disable_file_log,
    )
    resolved_level = _resolve_log_level(args.log_level)
    if resolved_level is None:
        resolved_level = logging.INFO
    _, active_log_file = setup_logging(
        level=int(resolved_level),
        log_file=log_file,
        stream_stdout=(log_file is None),
        respect_env=(args.log_level is None),
    )
    logger.info("python_log_file=%s", active_log_file if active_log_file is not None else "disabled")

    if args.command == "build-manifest":
        result = build_manifest(args.config)
        logger.info("manifest=%s", result.manifest_path)
        logger.info("counts=%s", result.donor_counts_path)
        return

    if args.command == "build-cache":
        result = build_cache(args.config, chroms=args.chrom, force=args.force)
        logger.info("cache_root=%s", result.cache_root)
        logger.info("cache_entries=%s", len(result.entries))
        logger.info("cache_hits=%s", sum(1 for x in result.entries if x.cache_hit))
        logger.info("cache_misses=%s", sum(1 for x in result.entries if not x.cache_hit))
        logger.info("cache_elapsed_sec=%.3f", result.elapsed_seconds)
        return

    if args.command == "simulate":
        result = simulate(
            config_path=args.config,
            chrom=args.chrom,
            n_shards=args.n_shards,
            workers=args.workers,
            threads_per_task=args.threads_per_task,
            task_start_method=args.task_start_method,
        )
        logger.info("task_count=%s", result.task_count)
        logger.info("sample_manifest_paths=%s", json.dumps([str(x) for x in result.sample_manifest_paths], sort_keys=True))
        logger.info("elapsed_sec=%.3f", result.elapsed_seconds)
        logger.info("stage_timings=%s", json.dumps(result.stage_timings, sort_keys=True))
        if result.n_shards == 1:
            logger.info("combined_vcf=%s", result.combined_vcf)
            logger.info("combined_hapstore=%s", result.combined_hapstore)
            logger.info("sample_manifest=%s", result.sample_manifest_path)
            if result.combined_pgen_prefix is not None:
                logger.info("combined_pgen_prefix=%s", result.combined_pgen_prefix)
        return

    if args.command == "merge":
        result = merge(
            args.config,
            input_root=args.input_root,
            chrom=args.chrom,
            n_shards=args.n_shards,
            merge_combined=args.merge_combined,
        )
        logger.info("combined_hapstore=%s", result.combined_hapstore)
        logger.info("merged_manifest=%s", result.merged_manifest)
        logger.info("population_merged_hapstores=%s", json.dumps({k: str(v) for k, v in result.population_merged_hapstores.items()}, sort_keys=True))
        if result.merged_pgen_prefix is not None:
            logger.info("merged_pgen_prefix=%s", result.merged_pgen_prefix)
        logger.info("thread_plan=%s", json.dumps(result.thread_plan, sort_keys=True))
        logger.info("stage_timings=%s", json.dumps(result.stage_timings, sort_keys=True))
        return

    if args.command == "qc":
        result = qc(args.config, chrom=args.chrom, n_shards=args.n_shards)
        logger.info("gate_pass=%s", result.gate_pass)
        logger.info("elapsed_sec=%.3f", result.elapsed_seconds)
        logger.info("stage_timings=%s", json.dumps(result.stage_timings or {}, sort_keys=True))
        logger.info("diagnostics=%s", result.diagnostics_path)
        return

    if args.command == "qc-donor-round-robin":
        result = qc_donor_round_robin(
            args.config,
            chrom=args.chrom,
            fold_fraction=args.fold_fraction,
            seed=args.seed,
            workers=args.workers,
            populations=args.population,
        )
        logger.info("diagnostics=%s", result.diagnostics_path)
        logger.info("table=%s", result.table_path)
        logger.info("n_records=%s", result.n_records)
        logger.info("summary_by_population=%s", json.dumps(result.summary_by_population, sort_keys=True))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
