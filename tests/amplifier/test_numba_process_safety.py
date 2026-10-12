from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from unittest.mock import Mock

import pytest

from forgephast.amplifier import simulation


@pytest.mark.parametrize("backend", ["tbb", "workqueue"])
def test_fork_accepts_safe_numba_backend(monkeypatch, backend, caplog):
    monkeypatch.setattr(simulation, "HAS_NUMBA", True)
    initialize = Mock()
    monkeypatch.setattr(simulation, "get_num_threads", initialize)
    monkeypatch.setattr(simulation, "threading_layer", lambda: backend)
    monkeypatch.setattr(simulation.sys, "platform", "linux")
    with caplog.at_level("INFO", logger=simulation.__name__):
        simulation._check_numba_fork_backend("fork")
    initialize.assert_called_once_with()
    assert f"threading_layer={backend}" in caplog.text


def test_fork_rejects_initialized_openmp_before_pool_creation(monkeypatch):
    monkeypatch.setattr(simulation, "HAS_NUMBA", True)
    monkeypatch.setattr(simulation, "get_num_threads", Mock())
    monkeypatch.setattr(simulation, "threading_layer", lambda: "omp")
    monkeypatch.setattr(simulation.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="not fork-safe on Linux"):
        simulation._check_numba_fork_backend("fork")


@pytest.mark.parametrize("method,has_numba", [("spawn", True), ("forkserver", True), ("fork", False)])
def test_backend_check_leaves_other_execution_modes_unchanged(monkeypatch, method, has_numba):
    monkeypatch.setattr(simulation, "HAS_NUMBA", has_numba)
    initialize = Mock(side_effect=AssertionError("Numba must not be initialized"))
    monkeypatch.setattr(simulation, "get_num_threads", initialize)
    simulation._check_numba_fork_backend(method)
    initialize.assert_not_called()


def test_tbb_fork_rejects_non_main_thread(monkeypatch):
    monkeypatch.setattr(simulation, "HAS_NUMBA", True)
    monkeypatch.setattr(simulation, "get_num_threads", Mock())
    monkeypatch.setattr(simulation, "threading_layer", lambda: "tbb")
    monkeypatch.setattr(simulation.threading, "current_thread", lambda: object())
    with pytest.raises(RuntimeError, match="main thread"):
        simulation._check_numba_fork_backend("fork")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux Numba/fork regression")
@pytest.mark.parametrize("backend", ["tbb", "omp", "workqueue"])
def test_linux_initialized_backend_in_subprocess(backend):
    if importlib.util.find_spec("numba") is None:
        pytest.skip("requires the accel extra")
    # Isolate backend selection and bound the original deadlock regression.
    code = textwrap.dedent("""
        import multiprocessing as mp
        import os
        import numpy as np
        from forgephast.amplifier import simulation as s

        kwargs = dict(donor_haps=np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8),
                      switch_probs=np.array([0.2, 0.2]), n_haps=4, seed=42, threads=2)
        expected = s.simulate_haplotypes(**kwargs)
        assert s.threading_layer() == os.environ['NUMBA_THREADING_LAYER']
        if os.environ['NUMBA_THREADING_LAYER'] == 'omp':
            try:
                s._check_numba_fork_backend('fork')
            except RuntimeError as exc:
                assert 'not fork-safe on Linux' in str(exc)
            else:
                raise AssertionError('unsafe fork was accepted')
        else:
            s._check_numba_fork_backend('fork')
            with mp.get_context('fork').Pool(2) as pool:
                actual = pool.apply(s.simulate_haplotypes, kwds=kwargs)
            np.testing.assert_array_equal(actual, expected)
        print('BACKEND_CHECK_PASSED')
    """)
    env = dict(os.environ, NUMBA_THREADING_LAYER=backend, NUMBA_NUM_THREADS="2")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BACKEND_CHECK_PASSED" in result.stdout
