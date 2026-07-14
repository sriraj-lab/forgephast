"""Shared numerical summaries used by multiple QC metrics."""

from __future__ import annotations

import numpy as np


def auc_from_scores(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return float("nan")

    scores = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8)],
        axis=0,
    )
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ranks = np.empty(sorted_scores.size, dtype=np.float64)
    index = 0
    while index < sorted_scores.size:
        end = index + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[index]:
            end += 1
        ranks[index:end] = 0.5 * ((index + 1) + end)
        index = end

    positive_rank_sum = float(np.sum(ranks[sorted_labels == 1]))
    u_statistic = positive_rank_sum - (positive.size * (positive.size + 1)) / 2.0
    return float(np.clip(u_statistic / float(positive.size * negative.size), 0.0, 1.0))


def histogram_probabilities(values: np.ndarray, *, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(edges.size - 1, dtype=np.float64), np.zeros(edges.size - 1, dtype=np.int64)
    counts, _ = np.histogram(finite, bins=edges)
    counts = counts.astype(np.int64, copy=False)
    total = float(np.sum(counts))
    probabilities = (
        counts.astype(np.float64) / total
        if total > 0.0
        else np.zeros(edges.size - 1, dtype=np.float64)
    )
    return probabilities, counts


def root_mean_square_error(first: np.ndarray, second: np.ndarray) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        raise ValueError("rmse inputs must have identical shapes")
    difference = first_array - second_array
    return float(np.sqrt(np.mean(difference * difference)))


def maf_from_genotype_counts(
    observed_count: np.ndarray,
    heterozygous_count: np.ndarray,
    homozygous_alt_count: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observed_count, dtype=np.float64)
    heterozygous = np.asarray(heterozygous_count, dtype=np.float64)
    homozygous_alt = np.asarray(homozygous_alt_count, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        frequency = (2.0 * homozygous_alt + heterozygous) / (2.0 * observed)
    maf = np.minimum(frequency, 1.0 - frequency)
    maf = np.where(np.isfinite(maf), maf, np.nan)
    return np.clip(maf.astype(np.float64, copy=False), 0.0, 0.5)
