from __future__ import annotations

import logging

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.qc.gate_support import evaluate_main_qc_gates
from forgephast.amplifier.qc.metric_support import (
    ExtendedMetricContext,
    run_registered_metrics,
)
from forgephast.amplifier.qc.registry import MetricSpec

from .conftest import write_test_config


def test_registered_gate_metric_error_fails_closed(tmp_path) -> None:
    config = from_yaml(write_test_config(tmp_path, name="metric_fail_closed"))

    def fail_metric(**kwargs):
        del kwargs
        raise RuntimeError("metric failure")

    evaluation = run_registered_metrics(
        config=config,
        chrom=21,
        inputs=object(),
        groups={},
        diagnostics_dir=tmp_path,
        numba_threads=1,
        metric_registry=(
            MetricSpec(
                name="ld_decay",
                payload_key="ld_decay",
                enabled=lambda cfg: True,
                runner=fail_metric,
                contributes_gate=True,
            ),
        ),
        ext_context=ExtendedMetricContext(
            donors_by_population={},
            donor_manifest_error=None,
            ext_data=None,
            ext_data_error=None,
        ),
        admixed_qc_enabled=False,
        admixed_pop_set=set(),
        apply_admixed_non_gating_fn=lambda payload, **kwargs: payload,
        initial_gate_pass=True,
        logger=logging.getLogger(__name__),
    )

    assert evaluation.gate_pass is False
    assert evaluation.payloads["ld_decay"]["status"] == "error"
    assert evaluation.payloads["ld_decay"]["gate_pass"] is False


def test_donor_baseline_error_fails_closed(tmp_path) -> None:
    config = from_yaml(write_test_config(tmp_path, name="donor_fail_closed"))

    def fail_donor_baseline(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("donor baseline failure")

    evaluation = evaluate_main_qc_gates(
        config=config,
        chrom=21,
        n_shards=1,
        pairwise={"P1__P2": 0.1},
        within={},
        groups={},
        pairs=[("P1", "P2")],
        pop_names=["P1", "P2"],
        source_source_pairs=[("P1", "P2")],
        admixed_involved_pairs=[],
        pair_key_to_pair={"P1__P2": ("P1", "P2")},
        population_msprime_is_admixed={},
        admixed_qc_enabled=False,
        diagnostics_dir=tmp_path,
        fst_gate_tolerance=0.2,
        python_qc_numba_threads=1,
        donor_baseline_fst_fn=fail_donor_baseline,
        load_manifest_fn=lambda cfg: object(),
        stochastic_gate_activation_fn=lambda **kwargs: {"enabled": False},
        stochastic_gate_fn=lambda **kwargs: {"available": False},
        msprime_tract_composition_gate_fn=lambda **kwargs: {"enabled": False},
        admixed_tract_length_gate_fn=lambda **kwargs: {"enabled": False},
        admixed_ancestry_variance_gate_fn=lambda **kwargs: {"enabled": False},
        is_admixed_involved_pair_fn=lambda pair, flags: False,
        logger=logging.getLogger(__name__),
    )

    assert evaluation.gate_pass is False
    assert evaluation.fst_gate_effective_pass is False
    assert evaluation.fst_pairwise_gate_status["__gate_error__"]["pair_pass"] is False
