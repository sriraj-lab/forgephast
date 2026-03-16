"""Grid expansion and class-based grouping tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from forgephast.phenotype.grid import GridFactor, build_grouped_cases, expand_grid_cases, load_grid_spec


def _base_payload() -> dict:
    return {
        "name": "grid_unit",
        "architectures": [
            {
                "name": "arch1",
                "params": {
                    "h2": 0.2,
                    "sigma_beta": 1.0,
                },
            }
        ],
        "effect_model": {
            "base_beta_distribution": "normal",
            "population_order": ["EUR"],
            "population_corr_matrix": [[1.0]],
            "jitter_sd_by_population": {"EUR": 0.1},
        },
        "causal_selection": {
            "maf_buckets": [
                {
                    "name": "bucket",
                    "maf_min": 0.01,
                    "maf_max": 0.02,
                    "fixed_count": 10,
                    "subset_count": 100,
                }
            ]
        },
        "variance_budget": {"h2_fixed": 0.2, "covariate": 0.2, "residual": 0.6},
    }


def test_load_grid_spec_defaults(tmp_path: Path) -> None:
    spec_path = tmp_path / "grid.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "base.yaml",
                "factors": [
                    {"path": "architectures[0].params.h2", "values": [0.2, 0.4]},
                ],
            }
        ),
        encoding="utf-8",
    )
    spec = load_grid_spec(spec_path)
    assert spec.base_config == "base.yaml"
    assert spec.stage == "simulate"
    assert spec.parallel_runs == 1
    assert spec.continue_on_error is False
    assert spec.runtime.max_cache_gb == 8.0


def test_expand_grid_cases_and_group_class_a_boundary() -> None:
    factors = [
        GridFactor(
            paths=("causal_selection.maf_buckets[0].subset_count",),
            values=[(100,), (200,)],
        ),
        GridFactor(
            paths=("effect_model.base_beta_distribution",),
            values=[("normal",), ("laplace",)],
        ),
        GridFactor(
            paths=("architectures[0].params.h2",),
            values=[(0.2,), (0.5,)],
        ),
    ]
    cases = expand_grid_cases(_base_payload(), factors=factors)
    assert len(cases) == 8
    assert [case.payload["name"] for case in cases] == [
        "grid_unit_grid_0001",
        "grid_unit_grid_0002",
        "grid_unit_grid_0003",
        "grid_unit_grid_0004",
        "grid_unit_grid_0005",
        "grid_unit_grid_0006",
        "grid_unit_grid_0007",
        "grid_unit_grid_0008",
    ]

    grouped = build_grouped_cases(cases)
    assert len(grouped) == 2
    assert all(len(group) == 4 for group in grouped)

    subset_counts = []
    for group in grouped:
        a_keys = {case.a_key for case in group}
        assert len(a_keys) == 1
        count = group[0].payload["causal_selection"]["maf_buckets"][0]["subset_count"]
        subset_counts.append(count)
        distributions = [
            case.payload["effect_model"]["base_beta_distribution"] for case in group
        ]
        h2_values = [case.payload["architectures"][0]["params"]["h2"] for case in group]
        assert sorted(distributions) == ["laplace", "laplace", "normal", "normal"]
        assert sorted(h2_values) == [0.2, 0.2, 0.5, 0.5]

    assert sorted(subset_counts) == [100, 200]


def test_load_grid_spec_supports_coupled_factors(tmp_path: Path) -> None:
    spec_path = tmp_path / "grid.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "base.yaml",
                "factors": [
                    {
                        "paths": [
                            "effect_model.population_corr_matrix[0][1]",
                            "effect_model.population_corr_matrix[1][0]",
                        ],
                        "values": [
                            [0.8, 0.8],
                            [1.0, 1.0],
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec = load_grid_spec(spec_path)
    assert len(spec.factors) == 1
    assert spec.factors[0].paths == (
        "effect_model.population_corr_matrix[0][1]",
        "effect_model.population_corr_matrix[1][0]",
    )
    assert spec.factors[0].values == [(0.8, 0.8), (1.0, 1.0)]


def test_expand_grid_cases_applies_coupled_overrides() -> None:
    base = _base_payload()
    base["effect_model"]["population_corr_matrix"] = [[1.0, 1.0], [1.0, 1.0]]
    base["variance_budget"] = {"h2_fixed": 0.05, "covariate": 0.2, "residual": 0.75, "h2_bg": 0.0}
    factors = [
        GridFactor(
            paths=(
                "effect_model.population_corr_matrix[0][1]",
                "effect_model.population_corr_matrix[1][0]",
            ),
            values=[(0.8, 0.8), (1.0, 1.0)],
        ),
        GridFactor(
            paths=(
                "architectures[0].params.h2",
                "variance_budget.h2_fixed",
                "variance_budget.residual",
            ),
            values=[
                (0.05, 0.05, 0.75),
                (0.25, 0.25, 0.55),
                (0.50, 0.50, 0.30),
            ],
        ),
    ]
    cases = expand_grid_cases(base, factors=factors)
    assert len(cases) == 6
    for case in cases:
        corr = case.payload["effect_model"]["population_corr_matrix"]
        assert corr[0][1] == corr[1][0]
        vb = case.payload["variance_budget"]
        total = float(vb["h2_fixed"]) + float(vb["h2_bg"]) + float(vb["covariate"]) + float(vb["residual"])
        assert abs(total - 1.0) < 1e-6
