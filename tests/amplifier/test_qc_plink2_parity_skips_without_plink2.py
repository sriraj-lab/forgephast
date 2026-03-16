from __future__ import annotations

from pathlib import Path
import types

import pytest
import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.qc.metrics.plink2_parity import run_plink2_parity_smoke

from .conftest import write_test_config


def test_qc_plink2_parity_skips_without_plink2(tmp_path: Path):
    cfg_path = write_test_config(tmp_path, name="plink2_parity_skip")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    payload.setdefault("qc", {})
    payload["qc"]["plink2_parity_enabled"] = True
    payload["outputs"]["plink2_path"] = "plink2_DOES_NOT_EXIST_FOR_TEST"

    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    dummy_inputs = types.SimpleNamespace(combined_vcf=Path("/tmp/does_not_matter_chr21.vcf.gz"))
    out = run_plink2_parity_smoke(
        config=cfg,
        chrom=21,
        inputs=dummy_inputs,
        groups={},
        donors_by_population={},
    )
    assert out["status"] == "skipped_no_plink2"

