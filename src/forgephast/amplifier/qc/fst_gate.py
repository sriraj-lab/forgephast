"""Donor-baseline and stochastic FST gate helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..config import RunConfig
from ..utils import derived_seed
from .metrics.fst import (
    hudson_fst_from_counts as matrix_hudson_fst_from_counts,
    sample_variant_indices,
)
from .policy_support import pair_key


def normalized_donor_weights(
    donor_ids: list[str],
    weights_by_donor: dict[str, float] | None,
) -> np.ndarray:
    """Return donor probabilities in the same order as the loaded genotype rows."""
    if not donor_ids:
        raise ValueError("donor_ids cannot be empty")
    if not weights_by_donor:
        return np.full(len(donor_ids), 1.0 / float(len(donor_ids)), dtype=np.float64)

    weights = np.asarray(
        [float(weights_by_donor.get(str(sample_id), 0.0)) for sample_id in donor_ids],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("donor sampling weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("donor sampling weights must sum to a positive value")
    return weights / total


def weighted_genotype_counts(
    genotypes: np.ndarray,
    donor_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert weighted donor genotypes to Hudson-FST allele counts.

    Kish effective sample size preserves the existing unweighted result when
    all donors are equally weighted while supporting configured donor mixtures.
    """
    geno = np.asarray(genotypes, dtype=np.float64)
    weights = np.asarray(donor_weights, dtype=np.float64)
    if geno.ndim != 2:
        raise ValueError("genotypes must be a two-dimensional diploid matrix")
    if weights.ndim != 1 or weights.shape[0] != geno.shape[0]:
        raise ValueError("donor_weights must have one value per genotype row")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("donor_weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("donor_weights must sum to a positive value")
    weights = weights / total

    effective_n = 1.0 / float(np.sum(np.square(weights)))
    allele_number = 2.0 * effective_n
    alt_frequency = np.sum(geno * weights[:, None], axis=0, dtype=np.float64) / 2.0
    return (
        alt_frequency * allele_number,
        np.full(geno.shape[1], allele_number, dtype=np.float64),
    )


def pairwise_fst_from_counts_map_impl(
    *,
    pairs: list[tuple[str, str]],
    counts_by_population: dict[str, tuple[np.ndarray, np.ndarray]],
    numba_threads: int | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for a, b in pairs:
        key = pair_key(a, b)
        if a not in counts_by_population or b not in counts_by_population:
            out[key] = float("nan")
            continue
        ac_a, an_a = counts_by_population[a]
        ac_b, an_b = counts_by_population[b]
        out[key] = matrix_hudson_fst_from_counts(
            ac_a,
            an_a,
            ac_b,
            an_b,
            numba_threads=numba_threads,
        )
    return out


def donor_baseline_fst_impl(
    *,
    config: RunConfig,
    chrom: int,
    pairs: list[tuple[str, str]] | None,
    load_manifest_fn: Callable[[RunConfig], Any],
    resolve_pairwise_sets_fn: Callable[[RunConfig, list[str]], list[tuple[str, str]]],
    resolve_donor_panel_source_for_population_fn: Callable[[RunConfig, int, str], tuple[str, Any]],
    load_sample_haplotypes_from_panel_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    resolve_python_qc_numba_threads_fn: Callable[[RunConfig], int],
) -> dict[str, float]:
    manifest = load_manifest_fn(config)
    donors = manifest.donors_by_population
    donor_weights = manifest.donor_weights_by_population
    pops = sorted(donors.keys())
    if pairs is None:
        pairs = resolve_pairwise_sets_fn(config, pops)
    else:
        pairs = [(str(a), str(b)) for a, b in pairs]
    if not pairs:
        return {}

    needed_pops = sorted({pop for pair in pairs for pop in pair})
    per_pop_pos: dict[str, np.ndarray] = {}
    per_pop_genotypes: dict[str, np.ndarray] = {}
    per_pop_weights: dict[str, np.ndarray] = {}
    for pop in needed_pops:
        donor_ids = list(donors[pop])
        source_kind, source_path = resolve_donor_panel_source_for_population_fn(config, chrom, pop)
        donor_pos, donor_haps = load_sample_haplotypes_from_panel_fn(
            config=config,
            source_path=source_path,
            chrom=chrom,
            sample_ids=donor_ids,
            source_kind=source_kind,
        )
        per_pop_pos[pop] = np.asarray(donor_pos, dtype=np.int64)
        per_pop_genotypes[pop] = (
            donor_haps[0::2, :].astype(np.uint8, copy=False) + donor_haps[1::2, :].astype(np.uint8, copy=False)
        )
        per_pop_weights[pop] = normalized_donor_weights(
            donor_ids,
            donor_weights.get(pop),
        )

    common_pos = per_pop_pos[needed_pops[0]]
    for pop in needed_pops[1:]:
        common_pos = np.intersect1d(common_pos, per_pop_pos[pop], assume_unique=False)

    if common_pos.size < 2:
        return {pair_key(a, b): float("nan") for a, b in pairs}

    counts_by_pop: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_variants = int(common_pos.size)
    for pop in needed_pops:
        pop_pos = per_pop_pos[pop]
        idx = np.searchsorted(pop_pos, common_pos)
        if idx.size != common_pos.size or not np.array_equal(pop_pos[idx], common_pos):
            raise RuntimeError(f"donor baseline variant alignment mismatch for population {pop!r}")
        geno = np.asarray(per_pop_genotypes[pop][:, idx], dtype=np.uint8, copy=False)
        counts_by_pop[pop] = weighted_genotype_counts(
            geno,
            per_pop_weights[pop],
        )

    return pairwise_fst_from_counts_map_impl(
        pairs=pairs,
        counts_by_population=counts_by_pop,
        numba_threads=resolve_python_qc_numba_threads_fn(config),
    )


def resolve_fst_stochastic_gate_activation_impl(
    *,
    config: RunConfig,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    donors_by_population: dict[str, list[str]],
) -> dict[str, Any]:
    mode = str(config.qc.fst_stochastic_gate_mode).strip().lower()
    threshold = float(config.qc.fst_stochastic_small_pop_fraction_threshold)
    pop_names = sorted({str(pop) for pair in pairs for pop in pair})
    sampled_cap = int(config.qc.sampled_max_diploids_per_population)

    donor_sizes: dict[str, int] = {}
    sampled_diploids: dict[str, int] = {}
    sampled_fraction_by_population: dict[str, float] = {}
    for pop in pop_names:
        donor_n = len({str(x) for x in donors_by_population.get(pop, [])})
        synth_n = len({str(x) for x in groups.get(pop, [])})
        sampled_n = min(int(synth_n), int(sampled_cap))
        donor_sizes[pop] = int(donor_n)
        sampled_diploids[pop] = int(sampled_n)
        sampled_fraction_by_population[pop] = (
            float(sampled_n) / float(donor_n) if donor_n > 0 else float("nan")
        )

    finite_fracs = [x for x in sampled_fraction_by_population.values() if np.isfinite(x)]
    auto_triggered = bool(any(x >= threshold for x in finite_fracs))
    enabled = False
    reason = "disabled_by_mode"
    if mode == "on":
        enabled = True
        reason = "forced_on"
    elif mode == "auto":
        enabled = auto_triggered
        reason = "auto_small_population_triggered" if enabled else "auto_not_triggered"

    return {
        "mode": mode,
        "enabled": bool(enabled),
        "reason": reason,
        "small_pop_fraction_threshold": float(threshold),
        "auto_triggered": bool(auto_triggered),
        "population_order": pop_names,
        "donor_sizes": donor_sizes,
        "sampled_diploids_by_population": sampled_diploids,
        "sampled_fraction_by_population": sampled_fraction_by_population,
    }


def compute_stochastic_donor_fst_gate_impl(
    *,
    config: RunConfig,
    chrom: int,
    groups: dict[str, list[str]],
    pairs: list[tuple[str, str]],
    pairwise_fst: dict[str, float],
    donors_by_population: dict[str, list[str]],
    donor_weights_by_population: dict[str, dict[str, float]],
    sampled_diploids_by_population: dict[str, int],
    numba_threads: int | None,
    resolve_donor_panel_source_for_population_fn: Callable[[RunConfig, int, str], tuple[str, Any]],
    load_sample_haplotypes_from_panel_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    reps = int(config.qc.fst_stochastic_reps)
    replace = bool(config.qc.fst_stochastic_replace)
    min_p_ge = float(config.qc.fst_stochastic_min_p_ge)
    seed = int(derived_seed(int(config.qc.seed), "fst_stochastic_gate", chrom))
    pop_names = sorted({str(pop) for pair in pairs for pop in pair})

    if reps <= 0:
        return {"available": False, "reason": "non_positive_reps", "reps": int(reps)}
    if not pop_names:
        return {"available": False, "reason": "no_populations_for_pairs"}

    donor_sizes: dict[str, int] = {}
    n_target_by_pop: dict[str, int] = {}
    per_pop_pos: dict[str, np.ndarray] = {}
    per_pop_genotypes: dict[str, np.ndarray] = {}
    per_pop_weights: dict[str, np.ndarray] = {}
    for pop in pop_names:
        donor_ids_all = sorted({str(x) for x in donors_by_population.get(pop, [])})
        donor_n = len(donor_ids_all)
        donor_sizes[pop] = int(donor_n)
        if donor_n == 0:
            return {
                "available": False,
                "reason": f"missing_donors_for_population:{pop}",
                "donor_sizes": donor_sizes,
            }

        n_target = int(sampled_diploids_by_population.get(pop, 0))
        if n_target <= 0:
            synth_n = len({str(x) for x in groups.get(pop, [])})
            n_target = min(
                int(synth_n),
                int(config.qc.sampled_max_diploids_per_population),
                donor_n,
            )
        n_target = min(max(1, int(n_target)), donor_n)
        if n_target < 2:
            return {
                "available": False,
                "reason": f"insufficient_donors_for_population:{pop}",
                "donor_sizes": donor_sizes,
                "sampled_diploids_by_population": n_target_by_pop,
            }
        if (not replace) and donor_n < n_target:
            return {
                "available": False,
                "reason": f"replace_false_requires_more_donors:{pop}",
                "donor_sizes": donor_sizes,
                "sampled_diploids_by_population": n_target_by_pop,
            }
        n_target_by_pop[pop] = int(n_target)

        source_kind, source_path = resolve_donor_panel_source_for_population_fn(config, chrom, pop)
        donor_pos, donor_haps = load_sample_haplotypes_from_panel_fn(
            config=config,
            source_path=source_path,
            chrom=chrom,
            sample_ids=donor_ids_all,
            source_kind=source_kind,
        )
        if donor_haps.shape[0] < 4 or donor_haps.shape[1] < 2:
            return {
                "available": False,
                "reason": f"insufficient_donor_haplotypes:{pop}",
                "donor_sizes": donor_sizes,
            }
        donor_genotypes = (
            donor_haps[0::2, :].astype(np.uint8, copy=False) + donor_haps[1::2, :].astype(np.uint8, copy=False)
        )
        per_pop_pos[pop] = np.asarray(donor_pos, dtype=np.int64)
        per_pop_genotypes[pop] = np.asarray(donor_genotypes, dtype=np.uint8, copy=False)
        per_pop_weights[pop] = normalized_donor_weights(
            donor_ids_all,
            donor_weights_by_population.get(pop),
        )

    common_pos = per_pop_pos[pop_names[0]]
    for pop in pop_names[1:]:
        common_pos = np.intersect1d(common_pos, per_pop_pos[pop], assume_unique=False)
    if common_pos.size < 2:
        return {
            "available": False,
            "reason": "insufficient_common_variants",
            "common_variant_count": int(common_pos.size),
            "donor_sizes": donor_sizes,
            "sampled_diploids_by_population": n_target_by_pop,
        }

    variant_idx = sample_variant_indices(
        n_variants=int(common_pos.size),
        max_variants=int(config.qc.sampled_max_variants),
        seed=int(derived_seed(int(config.qc.seed), "main_qc_variants", chrom)),
    )
    sampled_pos = common_pos if variant_idx is None else common_pos[variant_idx]

    per_pop_genotypes_common: dict[str, np.ndarray] = {}
    for pop in pop_names:
        pop_pos = per_pop_pos[pop]
        idx = np.searchsorted(pop_pos, sampled_pos)
        if idx.size != sampled_pos.size or not np.array_equal(pop_pos[idx], sampled_pos):
            return {
                "available": False,
                "reason": f"failed_variant_axis_alignment:{pop}",
                "variant_count": int(sampled_pos.size),
                "donor_sizes": donor_sizes,
                "sampled_diploids_by_population": n_target_by_pop,
            }
        per_pop_genotypes_common[pop] = np.asarray(per_pop_genotypes[pop][:, idx], dtype=np.uint8, copy=False)

    n_variants = int(sampled_pos.size)
    baseline_counts_by_pop: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pop in pop_names:
        geno = per_pop_genotypes_common[pop]
        baseline_counts_by_pop[pop] = weighted_genotype_counts(
            geno,
            per_pop_weights[pop],
        )
    baseline_pairwise_fst = pairwise_fst_from_counts_map_impl(
        pairs=pairs,
        counts_by_population=baseline_counts_by_pop,
        numba_threads=numba_threads,
    )

    observed_pair_abs_diff: dict[str, float] = {}
    for a, b in pairs:
        key = pair_key(a, b)
        obs = pairwise_fst.get(key, float("nan"))
        base = baseline_pairwise_fst.get(key, float("nan"))
        if np.isfinite(obs) and np.isfinite(base):
            observed_pair_abs_diff[key] = float(abs(obs - base))
    if not observed_pair_abs_diff:
        return {
            "available": False,
            "reason": "no_finite_observed_pairs",
            "variant_count": int(n_variants),
            "donor_sizes": donor_sizes,
            "sampled_diploids_by_population": n_target_by_pop,
            "baseline_pairwise_fst": baseline_pairwise_fst,
        }

    null_per_pair: dict[str, list[float]] = {key: [] for key in observed_pair_abs_diff}
    null_rep_means: list[float] = []
    rng = np.random.default_rng(int(seed))
    for _ in range(reps):
        rep_counts_by_pop: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for pop in pop_names:
            geno = per_pop_genotypes_common[pop]
            n_target = n_target_by_pop[pop]
            sampled_rows = rng.choice(
                geno.shape[0],
                size=int(n_target),
                replace=bool(replace),
                p=per_pop_weights[pop],
            )
            sub = geno[sampled_rows, :]
            rep_counts_by_pop[pop] = (
                np.sum(sub, axis=0, dtype=np.float64),
                np.full(n_variants, 2.0 * float(n_target), dtype=np.float64),
            )

        rep_pairwise_fst = pairwise_fst_from_counts_map_impl(
            pairs=pairs,
            counts_by_population=rep_counts_by_pop,
            numba_threads=numba_threads,
        )
        rep_abs_diffs: list[float] = []
        for key in observed_pair_abs_diff:
            base = baseline_pairwise_fst.get(key, float("nan"))
            rep = rep_pairwise_fst.get(key, float("nan"))
            delta = float(abs(rep - base)) if np.isfinite(rep) and np.isfinite(base) else float("nan")
            null_per_pair[key].append(delta)
            if np.isfinite(delta):
                rep_abs_diffs.append(delta)
        null_rep_means.append(float(np.mean(rep_abs_diffs)) if rep_abs_diffs else float("nan"))

    obs_mean = float(np.mean(np.asarray(list(observed_pair_abs_diff.values()), dtype=np.float64)))
    null_arr = np.asarray([x for x in null_rep_means if np.isfinite(x)], dtype=np.float64)
    p_ge_mean = float((np.sum(null_arr >= obs_mean) + 1.0) / (null_arr.size + 1.0)) if null_arr.size else float("nan")
    gate_pass = bool(p_ge_mean >= min_p_ge) if np.isfinite(p_ge_mean) else None

    pair_summary: dict[str, dict[str, float | int]] = {}
    for key, obs in observed_pair_abs_diff.items():
        arr = np.asarray([x for x in null_per_pair[key] if np.isfinite(x)], dtype=np.float64)
        if arr.size:
            p_ge = float((np.sum(arr >= obs) + 1.0) / (arr.size + 1.0))
            pair_summary[key] = {
                "observed_abs_diff": float(obs),
                "null_mean": float(np.mean(arr)),
                "null_sd": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
                "null_min": float(np.min(arr)),
                "null_max": float(np.max(arr)),
                "p_ge_observed": float(p_ge),
                "n_reps_used": int(arr.size),
            }
        else:
            pair_summary[key] = {
                "observed_abs_diff": float(obs),
                "null_mean": float("nan"),
                "null_sd": float("nan"),
                "null_min": float("nan"),
                "null_max": float("nan"),
                "p_ge_observed": float("nan"),
                "n_reps_used": 0,
            }

    sampled_fraction_by_population = {
        pop: (float(n_target_by_pop[pop]) / float(donor_sizes[pop])) if donor_sizes[pop] > 0 else float("nan")
        for pop in pop_names
    }
    return {
        "available": True,
        "gate_pass": gate_pass,
        "criterion": "p_ge_observed_abs_diff_mean >= fst_stochastic_min_p_ge",
        "fst_stochastic_min_p_ge": float(min_p_ge),
        "reps": int(reps),
        "replace": bool(replace),
        "seed": int(seed),
        "variant_count": int(n_variants),
        "donor_sizes": donor_sizes,
        "sampled_diploids_by_population": n_target_by_pop,
        "sampled_fraction_by_population": sampled_fraction_by_population,
        "baseline_pairwise_fst": baseline_pairwise_fst,
        "observed_pair_abs_diff": observed_pair_abs_diff,
        "summary": {
            "observed_abs_diff_mean": float(obs_mean),
            "null_abs_diff_mean_mean": float(np.mean(null_arr)) if null_arr.size else float("nan"),
            "null_abs_diff_mean_sd": float(np.std(null_arr, ddof=1)) if null_arr.size > 1 else float("nan"),
            "null_abs_diff_mean_min": float(np.min(null_arr)) if null_arr.size else float("nan"),
            "null_abs_diff_mean_max": float(np.max(null_arr)) if null_arr.size else float("nan"),
            "p_ge_observed_abs_diff_mean": float(p_ge_mean),
            "n_reps_used": int(null_arr.size),
        },
        "pairwise": pair_summary,
    }
