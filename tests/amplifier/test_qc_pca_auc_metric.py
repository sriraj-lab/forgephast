from __future__ import annotations

import numpy as np
import pytest

from forgephast.amplifier.qc.metrics.pca import _auc_from_scores, _randomized_svd_compat, _standardize_scores


def test_pca_auc_is_deterministic_under_fixed_seed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 10))
    labels = np.asarray(["A"] * 10 + ["B"] * 10, dtype=object)
    x[:10, 0] += 3.0
    x[10:, 0] -= 3.0

    def _run() -> float:
        _, _, vt, _ = _randomized_svd_compat(x, n_components=3, random_state=123)
        scores = x @ vt.T
        scores_std, _, _ = _standardize_scores(scores)
        direction = np.mean(scores_std[labels == "A"], axis=0) - np.mean(scores_std[labels == "B"], axis=0)
        auc = _auc_from_scores(scores_std[labels == "A"] @ direction, scores_std[labels == "B"] @ direction)
        return float(max(auc, 1.0 - auc))

    auc1 = _run()
    auc2 = _run()
    assert auc2 == pytest.approx(auc1)
    assert 0.5 <= auc1 <= 1.0
