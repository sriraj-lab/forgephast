from __future__ import annotations

import pytest

from forgephast._internal.resources import require_slurm_cpu_budget, slurm_cpu_budget


def test_slurm_cpu_budget_rejects_nested_oversubscription(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert slurm_cpu_budget() == 4

    require_slurm_cpu_budget(workers=2, threads_per_worker=2, context="test")
    with pytest.raises(ValueError, match="exceeding SLURM_CPUS_PER_TASK=4"):
        require_slurm_cpu_budget(workers=3, threads_per_worker=2, context="test")
