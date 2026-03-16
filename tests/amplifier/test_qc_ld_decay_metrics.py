from __future__ import annotations

import numpy as np
import pandas as pd

from forgephast.amplifier.qc.pipeline import _ld_decay_metrics_from_curves


def _curve(values: list[float], n_pairs: int = 500) -> pd.DataFrame:
    rows = []
    for idx, val in enumerate(values):
        start = 0.05 * idx
        end = start + 0.05
        rows.append(
            {
                "bin_idx": idx,
                "bin_start_cm": start,
                "bin_end_cm": end,
                "bin_mid_cm": 0.5 * (start + end),
                "n_pairs": n_pairs,
                "mean_r2": val,
                "median_r2": val,
            }
        )
    return pd.DataFrame(rows)


def test_ld_decay_metrics_identical_curves():
    donor = _curve([0.60, 0.45, 0.35, 0.25, 0.18])
    synth = _curve([0.60, 0.45, 0.35, 0.25, 0.18])
    metrics = _ld_decay_metrics_from_curves(
        donor_curve=donor,
        synthetic_curve=synth,
        min_pairs_per_bin=100,
        half_decay_r2_threshold=0.20,
    )

    assert metrics["status"] == "ok"
    assert metrics["n_bins_used"] == 5
    assert float(metrics["rmse_median_r2"]) == 0.0
    assert float(metrics["auc_rel_diff"]) == 0.0
    assert np.isfinite(float(metrics["half_decay_cm_donor"]))
    assert np.isfinite(float(metrics["half_decay_cm_synthetic"]))
    assert float(metrics["half_decay_rel_diff"]) == 0.0


def test_ld_decay_metrics_insufficient_pairs():
    donor = _curve([0.60, 0.45, 0.35], n_pairs=10)
    synth = _curve([0.58, 0.44, 0.34], n_pairs=10)
    metrics = _ld_decay_metrics_from_curves(
        donor_curve=donor,
        synthetic_curve=synth,
        min_pairs_per_bin=200,
        half_decay_r2_threshold=0.20,
    )

    assert metrics["status"] == "insufficient_pairs_per_bin"
    assert metrics["n_bins_used"] == 0
