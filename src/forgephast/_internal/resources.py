from __future__ import annotations

import os


def slurm_cpu_budget() -> int | None:
    value = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if not value:
        return None
    budget = int(value)
    if budget <= 0:
        raise ValueError("SLURM_CPUS_PER_TASK must be > 0")
    return budget


def require_slurm_cpu_budget(*, workers: int, threads_per_worker: int, context: str) -> None:
    budget = slurm_cpu_budget()
    if budget is None:
        return
    requested = int(workers) * int(threads_per_worker)
    if requested > budget:
        raise ValueError(
            f"{context} requests {workers} workers x {threads_per_worker} threads = "
            f"{requested} CPUs, exceeding SLURM_CPUS_PER_TASK={budget}"
        )
