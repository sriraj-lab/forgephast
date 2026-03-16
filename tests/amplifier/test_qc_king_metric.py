from __future__ import annotations

import numpy as np
import pytest

from forgephast.amplifier.qc.metrics.king import _king_cross_pairs, _king_within_pairs


def test_king_kinship_matches_manual_calculation():
    g = np.asarray(
        [
            [0, 1, 2, 1],
            [0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    pairs_i = np.asarray([0], dtype=np.int64)
    pairs_j = np.asarray([1], dtype=np.int64)

    kin = _king_within_pairs(g, pairs_i, pairs_j, numba_threads=None)
    assert kin.shape == (1,)
    assert kin[0] == pytest.approx(0.0)

    g_missing = g.copy()
    g_missing[0, 2] = np.uint8(255)
    kin_missing = _king_within_pairs(g_missing, pairs_i, pairs_j, numba_threads=None)
    assert kin_missing[0] == pytest.approx(0.5)


def test_king_cross_pair_can_be_negative():
    ga = np.asarray([[0, 1, 2, 1]], dtype=np.uint8)
    gb = np.asarray([[2, 1, 0, 1]], dtype=np.uint8)
    idx_a = np.asarray([0], dtype=np.int64)
    idx_b = np.asarray([0], dtype=np.int64)

    kin = _king_cross_pairs(ga, gb, idx_a, idx_b, numba_threads=None)
    assert kin.shape == (1,)
    assert kin[0] == pytest.approx(-0.5)


def test_king_kinship_uses_hmin_hmax_adjustment():
    g = np.asarray(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 2],
        ],
        dtype=np.uint8,
    )
    pairs_i = np.asarray([0], dtype=np.int64)
    pairs_j = np.asarray([1], dtype=np.int64)

    # Counts (all variants non-missing):
    # hethet=2 (v0,v1), ibs0=0, het1=4, het2=2.
    # KING between-family estimator (PLINK2-compatible):
    # kin = (2*hethet - 4*ibs0 + hmin - hmax) / (4*hmin)
    #     = (4 + 2 - 4) / 8 = 0.25
    kin = _king_within_pairs(g, pairs_i, pairs_j, numba_threads=None)
    assert kin[0] == pytest.approx(0.25)
