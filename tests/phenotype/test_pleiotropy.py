"""Pleiotropic trait clusters: shared causal variants with correlated effects."""

from __future__ import annotations

import copy
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forgephast.phenotype.config import RunConfig
from forgephast.phenotype.config_schema import PleiotropyClusterConfig
from forgephast.phenotype.effects.pleiotropy import (
    cluster_roles,
    lower_factor,
    mix_cluster_betas,
    mix_cluster_residuals,
)
from forgephast.phenotype.io.write_outputs import pleiotropy_summary
from forgephast.phenotype.phenotypes.simulate import run_simulation


def _cluster(traits=("a", "b", "c"), corr=None, shared=0.6, residual=None, seed=7):
    k = len(traits)
    corr = corr if corr is not None else np.eye(k).tolist()
    return PleiotropyClusterConfig(name="C", traits=list(traits), effect_correlation=corr,
                                   shared_fraction=shared, residual_correlation=residual, random_seed=seed)


def _betas(variant_ids, populations, seed, sd=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in variant_ids:
        base = rng.normal(0, sd)
        for p in populations:
            delta = rng.normal(0, 0.1 * sd)
            rows.append(dict(variant_id=v, population=p, beta_base=base, beta_delta=delta,
                             beta=base + delta, effect_scale=1.0))
    return pd.DataFrame(rows)


def _wide(df, col="beta"):
    return df.pivot(index="variant_id", columns="population", values=col)


def test_lower_factor_handles_singular_and_rejects_indefinite():
    for r in ([[1, .8, .8], [.8, 1, .8], [.8, .8, 1]], [[1, 1], [1, 1]], [[1, -.5], [-.5, 1]]):
        f = lower_factor(r)
        np.testing.assert_allclose(f @ f.T, r, atol=1e-12)
        assert np.allclose(f, np.tril(f)) and f[0, 0] == 1.0
    with pytest.raises(ValueError):
        lower_factor([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])


def test_roles_partition_the_causal_set():
    ids = [f"chr1:{i}" for i in range(1, 104)]
    roles = cluster_roles(ids, _cluster(shared=0.6), replicate=1)
    assert roles.shared.sum() == round(0.6 * 103)
    masks = [roles.causal_mask(t) for t in "abc"]
    private = [m & ~roles.shared for m in masks]
    assert not (private[0] & private[1]).any() and not (private[1] & private[2]).any()
    assert (roles.shared | private[0] | private[1] | private[2]).all()
    assert sorted(p.sum() for p in private) == [13, 14, 14]
    again = cluster_roles(ids, _cluster(shared=0.6), replicate=1)
    assert (again.shared == roles.shared).all() and (again.private_owner == roles.private_owner).all()
    assert not (cluster_roles(ids, _cluster(shared=0.6), replicate=2).shared == roles.shared).all()
    with pytest.raises(ValueError):
        cluster_roles(ids[:1], _cluster(shared=0.0), replicate=1)


def test_identity_cluster_reproduces_independent_draws_exactly():
    ids = [f"chr1:{i}" for i in range(1, 51)]
    draws = {t: _betas(ids, ["EUR", "AFR"], seed=s) for s, t in enumerate("abc")}
    roles = cluster_roles(ids, _cluster(shared=1.0), replicate=1)
    out = mix_cluster_betas(draws, {t: 1.0 for t in "abc"}, roles, np.eye(3))
    for t in "abc":
        pd.testing.assert_frame_equal(out[t], draws[t])


def test_mixed_effects_have_target_correlation_and_keep_private_draws():
    ids = [f"chr1:{i}" for i in range(1, 30001)]
    r = [[1, .8, -.5], [.8, 1, -.2], [-.5, -.2, 1]]
    sds = {"a": 1.0, "b": 0.3, "c": 2.0}  # different sigma_beta per trait
    draws = {t: _betas(ids, ["EUR", "AFR"], seed=10 + s, sd=sds[t]) for s, t in enumerate("abc")}
    roles = cluster_roles(ids, _cluster(corr=r, shared=0.5), replicate=1)
    out = mix_cluster_betas(draws, sds, roles, r)
    wide = {t: _wide(out[t]).loc[ids] for t in "abc"}
    shared = roles.shared
    for i, a in enumerate("abc"):
        for j, b in enumerate("abc"):
            if i < j:
                c = np.corrcoef(wide[a].to_numpy()[shared, 0], wide[b].to_numpy()[shared, 0])[0, 1]
                assert c == pytest.approx(r[i][j], abs=0.03)
        # private: own draw unchanged; other traits' private variants have zero effect
        own = roles.private_owner == "abc".index(a)
        np.testing.assert_array_equal(wide[a].to_numpy()[own], _wide(draws[a]).loc[ids].to_numpy()[own])
        assert (wide[a].to_numpy()[~roles.causal_mask(a)] == 0).all()
        # scale preserved: mixed shared effects keep the trait's own SD
        assert np.std(wide[a].to_numpy()[shared, 0]) == pytest.approx(np.std(_wide(draws[a]).loc[ids].to_numpy()[shared, 0]), rel=0.03)
    # first trait keeps its own draw on the shared part too
    np.testing.assert_array_equal(wide["a"].to_numpy()[shared], _wide(draws["a"]).loc[ids].to_numpy()[shared])
    # beta = beta_base + beta_delta still holds after mixing
    for t in "abc":
        np.testing.assert_allclose(out[t]["beta"], out[t]["beta_base"] + out[t]["beta_delta"], atol=1e-12)


def test_residual_mixing():
    rng = np.random.default_rng(0)
    raw = {t: rng.normal(size=50000) for t in "ab"}
    assert mix_cluster_residuals(raw, "ab", None)["b"] is raw["b"]
    out = mix_cluster_residuals(raw, "ab", [[1, .4], [.4, 1]])
    np.testing.assert_array_equal(out["a"], raw["a"])
    assert np.corrcoef(out["a"], out["b"])[0, 1] == pytest.approx(0.4, abs=0.02)
    assert np.var(out["b"]) == pytest.approx(1.0, rel=0.03)


# ---- run_simulation end to end ---------------------------------------------------

N_SAMPLES, N_VARIANTS = 400, 200


def _payload(tmp_path: Path, clusters=None, architectures=("a", "b", "c", "solo")) -> dict:
    sample_ids = [f"s{i}" for i in range(N_SAMPLES)]
    (tmp_path / "samples.txt").write_text("\n".join(sample_ids) + "\n")
    (tmp_path / "pops.tsv").write_text("IID\tpopulation\n" + "".join(f"{s}\tEUR\n" for s in sample_ids))
    (tmp_path / "cov.tsv").write_text("IID\tage\n" + "".join(f"{s}\t{20 + i % 40}\n" for i, s in enumerate(sample_ids)))
    (tmp_path / "coef.tsv").write_text("covariate\tbeta\nage\t0\n")
    (tmp_path / "pairs.tsv").write_text("causal_id\tmaf_eur\n" + "".join(f"chr1:{10 * (i + 1)}\t0.25\n" for i in range(N_VARIANTS)))
    payload = {
        "name": "pleio",
        "genotype_input": {"format": "hapstore", "hapstore": {
            "root_dir": "/tmp/fake", "layout": "per_pop_per_chr", "file_pattern": "{population}/chr{chrom}.hapstore.zarr"}},
        "cohorts": [{"name": "c1", "sample_ids_file": str(tmp_path / "samples.txt"), "population_map_file": str(tmp_path / "pops.tsv")}],
        "architectures": [{"name": n, "plugin": "polygenic_no_pleiotropy", "params": {"sigma_beta": 1.0, "h2": 0.3}} for n in architectures],
        "causal_selection": {
            "source_pairs_file": str(tmp_path / "pairs.tsv"), "variant_id_column": "causal_id",
            "maf_columns_by_population": {"EUR": "maf_eur"}, "anchor_population": "EUR",
            "maf_buckets": [{"name": "common", "maf_min": 0.1, "maf_max": 0.3, "fixed_count": N_VARIANTS, "subset_count": 0}]},
        "effect_model": {"maf_scaling_mode": "none", "population_order": ["EUR"], "population_corr_matrix": [[1.0]],
                         "jitter_sd_by_population": {"EUR": 0.0}},
        "covariates": {"effects_mode": "fixed", "covariate_file": str(tmp_path / "cov.tsv"), "fixed_effects_file": str(tmp_path / "coef.tsv")},
        "phenotype": {"continuous_only": True, "n_replicates": 1},
        "variance_budget": {"h2_fixed": 0.3, "h2_bg": 0.0, "covariate": 0.0, "residual": 0.7, "strict_sum_to_one": True},
        "gwas": {"global_pcs": {"mode": "none"}},
        "paths": {"output_root": str(tmp_path / "outputs")},
        "chromosomes": [1],
    }
    if clusters is not None:
        payload["pleiotropy"] = {"clusters": clusters}
    return payload


class _Adapter:
    """Independent binomial(2, 0.25) genotypes, fixed per variant ID."""

    def __init__(self):
        self.last_export_allele_by_variant = {}
        self.last_export_other_allele_by_variant = {}

    def export_genotypes_u8(self, **kwargs):
        ids = list(kwargs["variant_ids"])
        self.last_export_allele_by_variant = {v: "A" for v in ids}
        self.last_export_other_allele_by_variant = {v: "G" for v in ids}
        cols = [np.random.default_rng(zlib.crc32(v.encode())).binomial(2, 0.25, N_SAMPLES).astype(np.uint8) for v in ids]
        return np.column_stack(cols)


def _run(tmp_path, payload):
    out = tmp_path / f"run{len(list(tmp_path.glob('run*')))}"
    records = run_simulation(run_cfg=RunConfig.from_dict(payload), run_dir=out, adapter=_Adapter())
    return {r["architecture"]: r for r in records}, records


def _genotypes(ids):
    return _Adapter().export_genotypes_u8(variant_ids=ids).astype(float)


CLUSTER = {"name": "C1", "traits": ["a", "b", "c"], "shared_fraction": 0.6, "random_seed": 3,
           "effect_correlation": [[1, .8, .8], [.8, 1, .8], [.8, .8, 1]],
           "residual_correlation": [[1, .3, .3], [.3, 1, .3], [.3, .3, 1]]}


def test_run_simulation_identity_cluster_matches_independent_run(tmp_path):
    plain, _ = _run(tmp_path, _payload(tmp_path))
    ident = dict(CLUSTER, shared_fraction=1.0, effect_correlation=np.eye(3).tolist(), residual_correlation=None)
    clustered, _ = _run(tmp_path, _payload(tmp_path, [ident]))
    for t in ("a", "b", "c", "solo"):
        pd.testing.assert_frame_equal(clustered[t]["phenotype_df"], plain[t]["phenotype_df"])
        pd.testing.assert_frame_equal(clustered[t]["effects_df"].reset_index(drop=True),
                                      plain[t]["effects_df"].reset_index(drop=True))


def test_run_simulation_cluster_outputs(tmp_path):
    plain, _ = _run(tmp_path, _payload(tmp_path))
    by_trait, records = _run(tmp_path, _payload(tmp_path, [CLUSTER]))
    # unclustered architecture is untouched by the cluster
    pd.testing.assert_frame_equal(by_trait["solo"]["phenotype_df"], plain["solo"]["phenotype_df"])
    assert by_trait["solo"]["pleiotropy"] == {}
    n_shared = round(0.6 * N_VARIANTS)
    shared_ids = None
    for t in "abc":
        rec = by_trait[t]
        m = rec["membership_df"]
        assert set(m["pleiotropy_role"]) == {"shared", "private"}
        assert (m["pleiotropy_role"] == "shared").sum() == n_shared
        assert len(m) == n_shared + rec["pleiotropy"]["n_private"]
        ids = set(m["causal_id"])
        assert set(rec["effects_df"]["variant_id"]) == ids  # only this trait's causal variants persisted
        s = set(m.loc[m["pleiotropy_role"] == "shared", "causal_id"])
        shared_ids = s if shared_ids is None else shared_ids
        assert s == shared_ids
        # the persisted effects reproduce the genetic component exactly
        eff = rec["effects_df"]
        g = _genotypes(list(eff["variant_id"])) @ eff["beta"].to_numpy(dtype=float)
        np.testing.assert_allclose(rec["phenotype_df"]["GENETIC_FIXED_COMPONENT"], g, rtol=1e-6, atol=1e-6)
    privates = [set(by_trait[t]["membership_df"]["causal_id"]) - shared_ids for t in "abc"]
    assert not (privates[0] & privates[1]) and not (privates[0] & privates[2]) and not (privates[1] & privates[2])
    summary = pleiotropy_summary(records)
    assert len(summary) == 3 and set(summary["cluster"]) == {"C1"}
    assert (summary["target_effect_correlation"] == 0.8).all()
    # ~0.6 * 0.8 expected; generous bounds for 200 variants and 400 samples
    assert summary["realized_genetic_correlation"].between(0.2, 0.75).all()
    assert summary["realized_residual_correlation"].between(0.15, 0.45).all()


def test_negative_pair_and_config_validation(tmp_path):
    neg = dict(CLUSTER, effect_correlation=[[1, .5, -.5], [.5, 1, 0.], [-.5, 0., 1]], shared_fraction=1.0, residual_correlation=None)
    _, records = _run(tmp_path, _payload(tmp_path, [neg]))
    s = pleiotropy_summary(records).set_index(["trait_a", "trait_b"])["realized_genetic_correlation"]
    assert s[("a", "c")] < -0.2 and s[("a", "b")] > 0.2
    bad = [
        dict(CLUSTER, traits=["a", "b", "zz"]),
        dict(CLUSTER, shared_fraction=1.5),
        dict(CLUSTER, effect_correlation=[[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]]),
        dict(CLUSTER, effect_correlation=[[1, .8], [.8, 1]]),
        dict(CLUSTER, traits=["a", "a", "b"]),
    ]
    for cluster in bad:
        with pytest.raises(ValueError):
            RunConfig.from_dict(_payload(tmp_path, [cluster]))
    with pytest.raises(ValueError):
        RunConfig.from_dict(_payload(tmp_path, [CLUSTER, dict(CLUSTER, name="C2", traits=["c", "solo"],
                                                              effect_correlation=[[1, .1], [.1, 1]], residual_correlation=None)]))
    payload = _payload(tmp_path, [CLUSTER])
    payload["pleiotropy"]["extra"] = 1
    with pytest.raises(ValueError):
        RunConfig.from_dict(payload)
