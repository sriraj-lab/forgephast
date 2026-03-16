from __future__ import annotations

import numpy as np
import pytest

from forgephast.amplifier.qc.metrics.afs import _hist_probs, _js_divergence, _maf_from_haplotypes


def test_afs_hist_and_divergences_on_toy_haplotypes():
    edges = np.linspace(0.0, 0.5, 6, dtype=np.float64)

    donor_haps = np.asarray(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    synth_haps = np.asarray(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    maf_d = _maf_from_haplotypes(donor_haps)
    maf_s = _maf_from_haplotypes(synth_haps)
    probs_d, _ = _hist_probs(maf_d, edges=edges)
    probs_s, _ = _hist_probs(maf_s, edges=edges)

    expected_d = np.asarray([1 / 3, 0.0, 1 / 3, 0.0, 1 / 3], dtype=np.float64)
    expected_s = np.asarray([1 / 3, 0.0, 0.0, 0.0, 2 / 3], dtype=np.float64)
    assert probs_d == pytest.approx(expected_d)
    assert probs_s == pytest.approx(expected_s)

    tv = 0.5 * float(np.sum(np.abs(probs_s - probs_d)))
    ks = float(np.max(np.abs(np.cumsum(probs_s) - np.cumsum(probs_d))))
    js = float(_js_divergence(probs_s, probs_d))

    assert tv == pytest.approx(1 / 3)
    assert ks == pytest.approx(1 / 3)
    assert js == pytest.approx(0.143841, rel=1e-6, abs=1e-6)

