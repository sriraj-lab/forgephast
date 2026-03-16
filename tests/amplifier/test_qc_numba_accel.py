from __future__ import annotations

import numpy as np

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.qc.pipeline import _resolve_python_qc_numba_threads
from forgephast.amplifier.qc.ld_decay import ld_curve_from_genotypes
from forgephast.amplifier.qc.fst import hudson_fst_from_counts

from .conftest import write_test_config


def test_resolve_python_qc_numba_threads_precedence(tmp_path):
    cfg_path = write_test_config(tmp_path, name="qc_numba_threads")
    cfg = from_yaml(cfg_path)
    cfg.qc.matrix_workers = 9
    cfg.qc.synthetic_rr_min_cpus_per_match = None
    cfg.qc.numba_threads = None
    assert _resolve_python_qc_numba_threads(cfg) == 9

    cfg.qc.synthetic_rr_min_cpus_per_match = 4
    assert _resolve_python_qc_numba_threads(cfg) == 4

    cfg.qc.numba_threads = 7
    assert _resolve_python_qc_numba_threads(cfg) == 7


def test_hudson_fst_numba_threads_parity():
    ac_a = np.asarray([10, 12, 8, 15, 9], dtype=np.float64)
    an_a = np.asarray([20, 20, 20, 20, 20], dtype=np.float64)
    ac_b = np.asarray([2, 6, 3, 7, 5], dtype=np.float64)
    an_b = np.asarray([20, 20, 20, 20, 20], dtype=np.float64)

    fst_default = hudson_fst_from_counts(ac_a, an_a, ac_b, an_b)
    fst_threaded = hudson_fst_from_counts(ac_a, an_a, ac_b, an_b, numba_threads=2)

    assert np.isfinite(fst_default)
    assert np.isfinite(fst_threaded)
    assert np.isclose(fst_default, fst_threaded, atol=1e-12)


def test_hudson_fst_uses_haplotype_an_correction():
    ac_a = np.asarray([80, 20, 60, 35], dtype=np.float64)
    an_a = np.asarray([400, 400, 400, 400], dtype=np.float64)
    ac_b = np.asarray([40, 10, 20, 70], dtype=np.float64)
    an_b = np.asarray([400, 400, 400, 400], dtype=np.float64)

    p1 = ac_a / an_a
    p2 = ac_b / an_b
    num = (p1 - p2) ** 2 - (p1 * (1.0 - p1) / (an_a - 1.0)) - (p2 * (1.0 - p2) / (an_b - 1.0))
    den = p1 * (1.0 - p2) + p2 * (1.0 - p1)
    expected = float(np.sum(num) / np.sum(den))

    got = hudson_fst_from_counts(ac_a, an_a, ac_b, an_b)
    assert np.isfinite(got)
    assert np.isclose(got, expected, atol=1e-12)


def test_ld_curve_accepts_numba_threads():
    genotypes = np.asarray(
        [
            [0, 1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    pos = np.asarray([100, 200, 300, 400, 500, 600], dtype=np.int64)
    map_bp = np.asarray([100, 200, 300, 400, 500, 600], dtype=np.float64)
    map_cm = np.asarray([0.00, 0.05, 0.10, 0.15, 0.20, 0.25], dtype=np.float64)

    curve = ld_curve_from_genotypes(
        genotypes=genotypes,
        variant_positions_bp=pos,
        map_bp=map_bp,
        map_cm=map_cm,
        max_distance_cm=0.20,
        bin_width_cm=0.05,
        min_maf=0.05,
        seed=7,
        max_pairs_per_bin=200,
        max_neighbors_per_anchor=8,
        numba_threads=2,
    )

    assert not curve.empty
    assert {"bin_idx", "n_pairs", "mean_r2", "median_r2"}.issubset(set(curve.columns))
    assert int(curve["n_pairs"].sum()) > 0
