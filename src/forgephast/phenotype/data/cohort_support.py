"""Cohort metadata alignment helpers."""

from __future__ import annotations

import pandas as pd
from hapstore.ids import align_by_ids

from ..config import CohortConfig


def load_population_labels(
    cohort: CohortConfig,
    sample_ids: list[str],
    default_population: str,
) -> list[str]:
    if not cohort.population_map_file:
        return [default_population] * len(sample_ids)

    frame = pd.read_csv(cohort.population_map_file, sep="\t")
    id_column = "IID" if "IID" in frame.columns else "#IID" if "#IID" in frame.columns else None
    if id_column is None:
        raise ValueError(f"cohort {cohort.name}: population_map_file must contain IID or #IID column")

    population_column = "population" if "population" in frame.columns else None
    if population_column is None:
        candidates = [column for column in frame.columns if column != id_column]
        if not candidates:
            raise ValueError(f"cohort {cohort.name}: could not infer population column from map file")
        population_column = candidates[0]

    aligned = align_by_ids(
        frame[[id_column, population_column]].copy(),
        id_column=id_column,
        ids=sample_ids,
        context=f"cohort {cohort.name} population map alignment",
    )
    return aligned[population_column].astype(str).tolist()
