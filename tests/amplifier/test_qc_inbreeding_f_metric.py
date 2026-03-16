from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.qc.metrics.inbreeding_f import compute_inbreeding_f_metrics

from .conftest import write_test_config


@dataclass(frozen=True)
class _PopData:
    donor_ids: list[str]
    synth_ids: list[str]
    donor_haps: np.ndarray
    synth_haps: np.ndarray


@dataclass(frozen=True)
class _ExtData:
    pops: list[str]
    variant_subset_id: str
    per_pop: dict[str, _PopData]


def test_inbreeding_f_metric_runs_and_matches_when_inputs_identical(tmp_path):
    cfg_path = write_test_config(tmp_path, name="inbreeding_f_metric")
    cfg = from_yaml(cfg_path)
    cfg.qc.inbreeding_f_enabled = True
    cfg.qc.inbreeding_f_min_n_obs = 1
    cfg.qc.inbreeding_f_min_maf = 0.01
    cfg.qc.plots_enabled = False

    rng = np.random.default_rng(11)
    n = 50
    v = 200
    donor_haps = rng.binomial(1, 0.25, size=(2 * n, v)).astype(np.uint8)
    synth_haps = donor_haps.copy()

    ext = _ExtData(
        pops=["P1"],
        variant_subset_id="test",
        per_pop={
            "P1": _PopData(
                donor_ids=[f"D{i}" for i in range(n)],
                synth_ids=[f"S{i}" for i in range(n)],
                donor_haps=donor_haps,
                synth_haps=synth_haps,
            )
        },
    )

    payload, created = compute_inbreeding_f_metrics(
        config=cfg,
        chrom=21,
        inputs=None,
        groups={},
        donors_by_population={},
        diagnostics_dir=tmp_path,
        ext_data=ext,
        ext_data_error=None,
    )

    assert payload["status"] == "ok"
    assert "population_summary" in payload
    assert payload["population_summary"]["P1"]["status"] in {"ok", "no_variants"}
    if payload["population_summary"]["P1"]["status"] == "ok":
        assert float(payload["population_summary"]["P1"]["hist_rmse"]) == 0.0
        assert float(payload["population_summary"]["P1"]["mean_shift"]) == 0.0
    assert any(str(p).endswith("inbreeding_f_hist_chr21.tsv") for p in created)

