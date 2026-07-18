from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from forgephast.amplifier.msprime_tracts import (
    _warn_ignored_nonuniform_donor_weights,
)


def _panel(weights: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        donor_haps=np.zeros((len(weights), 1), dtype=np.uint8),
        donor_hap_sampling_weights=np.asarray(weights, dtype=np.float64),
    )


def test_msprime_warns_when_nonuniform_donor_weights_are_ignored(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _warn_ignored_nonuniform_donor_weights(
            population_name="ADMIX",
            source_order=["EUR"],
            source_panels={"EUR": _panel([0.4, 0.1, 0.4, 0.1])},
        )

    assert "MSPRIME_TRACTS_DONOR_WEIGHTS_IGNORED" in caplog.text
    assert "target_population=ADMIX" in caplog.text
    assert "source_population=EUR" in caplog.text


def test_msprime_does_not_warn_for_uniform_donor_weights(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _warn_ignored_nonuniform_donor_weights(
            population_name="ADMIX",
            source_order=["EUR"],
            source_panels={"EUR": _panel([0.25, 0.25, 0.25, 0.25])},
        )

    assert "MSPRIME_TRACTS_DONOR_WEIGHTS_IGNORED" not in caplog.text
