"""Metric registry for run_qc orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import RunConfig
from .metrics.afs import compute_afs_metrics
from .metrics.hwe import compute_hwe_metrics
from .metrics.inbreeding_f import compute_inbreeding_f_metrics
from .metrics.king import compute_king_kinship_metrics
from .metrics.pca import compute_pca_auc_metrics
from .metrics.roh import compute_roh_metrics
from .metrics.synthetic_shard_round_robin import compute_synthetic_shard_round_robin_metrics


@dataclass(frozen=True)
class MetricSpec:
    name: str
    payload_key: str
    enabled: Callable[[RunConfig], bool]
    runner: Callable[..., dict[str, Any]]
    contributes_gate: bool = False


def _run_king(**kwargs: Any) -> dict[str, Any]:
    payload, _ = compute_king_kinship_metrics(**kwargs)
    return payload


def _run_synthetic_shard_round_robin(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    ext_data: Any | None,
    ext_data_error: str | None,
) -> dict[str, Any]:
    del donors_by_population, ext_data, ext_data_error
    return compute_synthetic_shard_round_robin_metrics(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=numba_threads,
    )


def _run_ld_decay(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    ext_data: Any | None,
    ext_data_error: str | None,
) -> dict[str, Any]:
    del donors_by_population, ext_data, ext_data_error
    from . import pipeline as qc_pipeline

    payload, _ = qc_pipeline._compute_ld_decay_qc(
        config=config,
        chrom=int(chrom),
        inputs=inputs,
        groups=groups,
        diagnostics_dir=diagnostics_dir,
        numba_threads=int(numba_threads),
    )
    return payload


def _run_afs(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    ext_data: Any | None,
    ext_data_error: str | None,
) -> dict[str, Any]:
    del numba_threads
    payload, _ = compute_afs_metrics(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        donors_by_population=donors_by_population,
        diagnostics_dir=diagnostics_dir,
        ext_data=ext_data,
        ext_data_error=ext_data_error,
    )
    return payload


def _run_hwe(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    ext_data: Any | None,
    ext_data_error: str | None,
) -> dict[str, Any]:
    del numba_threads
    payload, _ = compute_hwe_metrics(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        donors_by_population=donors_by_population,
        diagnostics_dir=diagnostics_dir,
        ext_data=ext_data,
        ext_data_error=ext_data_error,
    )
    return payload


def _run_inbreeding_f(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    ext_data: Any | None,
    ext_data_error: str | None,
) -> dict[str, Any]:
    del numba_threads
    payload, _ = compute_inbreeding_f_metrics(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        donors_by_population=donors_by_population,
        diagnostics_dir=diagnostics_dir,
        ext_data=ext_data,
        ext_data_error=ext_data_error,
    )
    return payload


def _run_pca(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    ext_data: Any | None,
    ext_data_error: str | None,
) -> dict[str, Any]:
    del numba_threads
    payload, _ = compute_pca_auc_metrics(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        donors_by_population=donors_by_population,
        diagnostics_dir=diagnostics_dir,
        ext_data=ext_data,
        ext_data_error=ext_data_error,
    )
    return payload


def _run_roh(
    *,
    config: RunConfig,
    chrom: int,
    inputs: Any,
    groups: dict[str, list[str]],
    donors_by_population: dict[str, list[str]],
    diagnostics_dir: Path,
    numba_threads: int,
    ext_data: Any | None,
    ext_data_error: str | None,
) -> dict[str, Any]:
    del numba_threads
    payload, _ = compute_roh_metrics(
        config=config,
        chrom=chrom,
        inputs=inputs,
        groups=groups,
        donors_by_population=donors_by_population,
        diagnostics_dir=diagnostics_dir,
        ext_data=ext_data,
        ext_data_error=ext_data_error,
    )
    return payload


METRIC_REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="synthetic_shard_round_robin",
        payload_key="synthetic_shard_round_robin",
        enabled=lambda cfg: bool(cfg.qc.synthetic_shard_round_robin_enabled),
        runner=_run_synthetic_shard_round_robin,
        contributes_gate=False,
    ),
    MetricSpec(
        name="ld_decay",
        payload_key="ld_decay",
        enabled=lambda cfg: bool(cfg.qc.ld_decay_enabled),
        runner=_run_ld_decay,
        contributes_gate=True,
    ),
    MetricSpec(
        name="king",
        payload_key="king",
        enabled=lambda cfg: bool(cfg.qc.king_enabled),
        runner=_run_king,
        contributes_gate=False,
    ),
    MetricSpec(
        name="afs",
        payload_key="afs",
        enabled=lambda cfg: bool(cfg.qc.afs_enabled),
        runner=_run_afs,
        contributes_gate=False,
    ),
    MetricSpec(
        name="hwe",
        payload_key="hwe",
        enabled=lambda cfg: bool(cfg.qc.hwe_enabled),
        runner=_run_hwe,
        contributes_gate=True,
    ),
    MetricSpec(
        name="inbreeding_f",
        payload_key="inbreeding_f",
        enabled=lambda cfg: bool(cfg.qc.inbreeding_f_enabled),
        runner=_run_inbreeding_f,
        contributes_gate=True,
    ),
    MetricSpec(
        name="pca",
        payload_key="pca",
        enabled=lambda cfg: bool(cfg.qc.pca_enabled),
        runner=_run_pca,
        contributes_gate=False,
    ),
    MetricSpec(
        name="roh",
        payload_key="roh",
        enabled=lambda cfg: bool(cfg.qc.roh_enabled),
        runner=_run_roh,
        contributes_gate=False,
    ),
)


__all__ = ["MetricSpec", "METRIC_REGISTRY"]
