from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class CpuBudget:
    cpus: int
    source: str


def _positive_env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def cpu_budget() -> CpuBudget:
    env_names = ["FORGEPHAST_CPU_BUDGET", "SLURM_CPUS_PER_TASK"]
    if os.environ.get("PBS_JOBID", "").strip():
        env_names.extend(["PBS_NP", "PBS_NCPUS", "NCPUS"])
    env_names.extend(["LSB_DJOB_NUMPROC", "NSLOTS"])

    for name in env_names:
        value = _positive_env_int(name)
        if value is not None:
            return CpuBudget(cpus=value, source=name)

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            affinity_count = len(get_affinity(0))
        except (OSError, NotImplementedError):
            affinity_count = 0
        if affinity_count > 0:
            return CpuBudget(cpus=affinity_count, source="process affinity")

    return CpuBudget(cpus=max(1, int(os.cpu_count() or 1)), source="os.cpu_count")


def require_cpu_budget(*, workers: int, threads_per_worker: int, context: str) -> None:
    workers = int(workers)
    threads_per_worker = int(threads_per_worker)
    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("workers and threads_per_worker must be > 0")
    budget = cpu_budget()
    requested = workers * threads_per_worker
    if requested > budget.cpus:
        raise ValueError(
            f"{context} requests {workers} workers x {threads_per_worker} threads = "
            f"{requested} CPUs, exceeding available CPU budget {budget.cpus} "
            f"from {budget.source}"
        )


def environment_scratch_root() -> Path | None:
    for name in (
        "FORGEPHAST_SCRATCH_ROOT",
        "SLURM_TMPDIR",
        "LOCAL_SCRATCH",
        "SCRATCH",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value)

    scheduler_job = any(
        os.environ.get(name, "").strip()
        for name in (
            "SLURM_JOB_ID",
            "PBS_JOBID",
            "LSB_JOBID",
            "JOB_ID",
            "AWS_BATCH_JOB_ID",
        )
    )
    if scheduler_job:
        for name in ("TMPDIR", "TMP", "TEMP"):
            value = os.environ.get(name, "").strip()
            if value:
                return Path(value)

    user = os.environ.get("USER", "").strip()
    cluster_scratch = Path("/scratch/tmp") / user if user else None
    if cluster_scratch is not None and cluster_scratch.is_dir() and os.access(cluster_scratch, os.W_OK):
        return cluster_scratch
    return None


def execution_token() -> str:
    for name in (
        "FORGEPHAST_JOB_ID",
        "SLURM_JOB_ID",
        "PBS_JOBID",
        "LSB_JOBID",
        "JOB_ID",
        "AWS_BATCH_JOB_ID",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
            if token:
                return token
    return f"pid{os.getpid()}"
