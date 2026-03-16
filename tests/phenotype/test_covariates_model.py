"""Covariate effects tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forgephast.phenotype.config import CovariateConfig
from forgephast.phenotype.covariates.model import build_covariate_effects


def _write_covariates(path: Path) -> None:
    df = pd.DataFrame(
        {
            "IID": ["s1", "s2", "s3"],
            "covar1": [1.0, 2.0, 3.0],
            "covar2": [0.5, 0.5, 0.5],
        }
    )
    df.to_csv(path, sep="\t", index=False)


def _write_fixed(path: Path) -> None:
    df = pd.DataFrame({"covariate": ["covar1", "covar2"], "beta": [2.0, 1.0]})
    df.to_csv(path, sep="\t", index=False)


def test_fixed_effect_exactness(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.tsv"
    fixed_path = tmp_path / "fixed.tsv"
    _write_covariates(cov_path)
    _write_fixed(fixed_path)

    cfg = CovariateConfig(
        effects_mode="fixed",
        covariate_file=str(cov_path),
        id_column="IID",
        fixed_effects_file=str(fixed_path),
        random_effect_sd=0.1,
    )

    out = build_covariate_effects(cfg, sample_ids=["s1", "s2", "s3"], replicate_seed=1)
    expected = np.array([2.5, 4.5, 6.5])
    np.testing.assert_allclose(out.contribution, expected)


def test_random_effect_reproducible(tmp_path: Path) -> None:
    cov_path = tmp_path / "cov.tsv"
    fixed_path = tmp_path / "fixed.tsv"
    _write_covariates(cov_path)
    _write_fixed(fixed_path)

    cfg = CovariateConfig(
        effects_mode="random",
        covariate_file=str(cov_path),
        id_column="IID",
        random_effect_sd=0.3,
    )

    out1 = build_covariate_effects(cfg, sample_ids=["s1", "s2", "s3"], replicate_seed=777)
    out2 = build_covariate_effects(cfg, sample_ids=["s1", "s2", "s3"], replicate_seed=777)
    np.testing.assert_allclose(out1.contribution, out2.contribution)

