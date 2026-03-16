from __future__ import annotations

import numpy as np
import pytest

from forgephast.amplifier.qc.metrics.roh import _roh_metrics


def test_roh_segments_and_total_cm():
    cm = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    g = np.asarray([[0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    total_cm, n_segments, max_seg = _roh_metrics(g, cm, min_segment_cm=0.2, max_gap_cm=0.25)
    assert total_cm.shape == (1,)
    assert total_cm[0] == pytest.approx(0.2)
    assert int(n_segments[0]) == 1
    assert max_seg[0] == pytest.approx(0.2)


def test_roh_breaks_on_large_variant_gap():
    cm = np.asarray([0.0, 0.2, 0.6, 0.8], dtype=np.float64)
    g = np.asarray([[0, 0, 0, 0]], dtype=np.uint8)

    total_cm, n_segments, max_seg = _roh_metrics(g, cm, min_segment_cm=0.2, max_gap_cm=0.25)
    assert total_cm[0] == pytest.approx(0.4)
    assert int(n_segments[0]) == 2
    assert max_seg[0] == pytest.approx(0.2)

