from __future__ import annotations

import os

import pytest

from forgephast._internal.resources import (
    cpu_budget,
    environment_scratch_root,
    execution_token,
    require_cpu_budget,
)


RESOURCE_ENV = (
    "FORGEPHAST_CPU_BUDGET",
    "SLURM_CPUS_PER_TASK",
    "PBS_JOBID",
    "PBS_NP",
    "PBS_NCPUS",
    "NCPUS",
    "LSB_DJOB_NUMPROC",
    "NSLOTS",
)


def _clear_resource_env(monkeypatch):
    for name in RESOURCE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("environment", "expected_cpus", "expected_source"),
    [
        ({"SLURM_CPUS_PER_TASK": "4"}, 4, "SLURM_CPUS_PER_TASK"),
        ({"PBS_JOBID": "123.server", "PBS_NP": "5"}, 5, "PBS_NP"),
        ({"PBS_JOBID": "123.server", "NCPUS": "6"}, 6, "NCPUS"),
        ({"LSB_DJOB_NUMPROC": "7"}, 7, "LSB_DJOB_NUMPROC"),
        ({"NSLOTS": "8"}, 8, "NSLOTS"),
    ],
)
def test_cpu_budget_supports_scheduler_allocations(
    monkeypatch,
    environment,
    expected_cpus,
    expected_source,
):
    _clear_resource_env(monkeypatch)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    budget = cpu_budget()

    assert budget.cpus == expected_cpus
    assert budget.source == expected_source


def test_explicit_cpu_budget_overrides_scheduler(monkeypatch):
    _clear_resource_env(monkeypatch)
    monkeypatch.setenv("FORGEPHAST_CPU_BUDGET", "3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")

    assert cpu_budget().cpus == 3
    assert cpu_budget().source == "FORGEPHAST_CPU_BUDGET"


def test_cpu_budget_falls_back_to_process_affinity(monkeypatch):
    _clear_resource_env(monkeypatch)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)

    assert cpu_budget().cpus == 3
    assert cpu_budget().source == "process affinity"


def test_cpu_budget_rejects_nested_oversubscription(monkeypatch):
    _clear_resource_env(monkeypatch)
    monkeypatch.setenv("FORGEPHAST_CPU_BUDGET", "4")

    require_cpu_budget(workers=2, threads_per_worker=2, context="test")
    with pytest.raises(ValueError, match="available CPU budget 4 from FORGEPHAST_CPU_BUDGET"):
        require_cpu_budget(workers=3, threads_per_worker=2, context="test")


def test_cpu_budget_rejects_invalid_environment_value(monkeypatch):
    _clear_resource_env(monkeypatch)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "unknown")

    with pytest.raises(ValueError, match="SLURM_CPUS_PER_TASK must be a positive integer"):
        cpu_budget()


def test_environment_scratch_root_uses_generic_temp_in_scheduler_job(monkeypatch, tmp_path):
    for name in (
        "FORGEPHAST_SCRATCH_ROOT",
        "SLURM_TMPDIR",
        "TMPDIR",
        "LOCAL_SCRATCH",
        "SCRATCH",
        "TMP",
        "TEMP",
        "SLURM_JOB_ID",
        "PBS_JOBID",
        "LSB_JOBID",
        "JOB_ID",
        "AWS_BATCH_JOB_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PBS_JOBID", "123.server")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    assert environment_scratch_root() == tmp_path


def test_environment_scratch_root_does_not_redirect_ordinary_local_tmpdir(monkeypatch, tmp_path):
    for name in (
        "FORGEPHAST_SCRATCH_ROOT",
        "SLURM_TMPDIR",
        "TMPDIR",
        "LOCAL_SCRATCH",
        "SCRATCH",
        "TMP",
        "TEMP",
        "SLURM_JOB_ID",
        "PBS_JOBID",
        "LSB_JOBID",
        "JOB_ID",
        "AWS_BATCH_JOB_ID",
        "USER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    assert environment_scratch_root() is None


def test_environment_scratch_root_accepts_explicit_portable_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGEPHAST_SCRATCH_ROOT", str(tmp_path))

    assert environment_scratch_root() == tmp_path


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("SLURM_JOB_ID", "123", "123"),
        ("PBS_JOBID", "123.server", "123.server"),
        ("LSB_JOBID", "job/42", "job_42"),
        ("JOB_ID", "77", "77"),
        ("AWS_BATCH_JOB_ID", "batch:88", "batch_88"),
    ],
)
def test_execution_token_supports_scheduler_and_batch_ids(monkeypatch, name, value, expected):
    for env_name in (
        "FORGEPHAST_JOB_ID",
        "SLURM_JOB_ID",
        "PBS_JOBID",
        "LSB_JOBID",
        "JOB_ID",
        "AWS_BATCH_JOB_ID",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv(name, value)

    assert execution_token() == expected
