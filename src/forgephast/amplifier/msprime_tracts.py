"""msprime-backed tract simulation and donor-haplotype copying."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any

import numpy as np

from .vcfio import PanelChromData

logger = logging.getLogger(__name__)


@dataclass
class MsprimeTractSimulationResult:
    haplotypes: np.ndarray
    tract_summary: dict[str, Any]
    tract_rows: list[dict[str, Any]]
    individual_ancestry_rows: list[dict[str, Any]]


def _import_msprime() -> Any:
    try:
        import msprime  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when optional dependency is missing.
        raise RuntimeError(
            "population uses engine='msprime_tracts' but optional dependency 'msprime' is unavailable. "
            "Install with: python -m pip install -e .[msprime]"
        ) from exc
    return msprime


def _build_rate_map(
    *,
    msprime: Any,
    map_bp: np.ndarray,
    map_cm: np.ndarray,
    seq_length: float,
) -> Any:
    bp = np.asarray(map_bp, dtype=np.float64)
    cm = np.asarray(map_cm, dtype=np.float64)
    if bp.size < 2:
        raise ValueError("genetic map must contain at least two rows")

    order = np.argsort(bp)
    bp = bp[order]
    cm = cm[order]
    uniq_bp, uniq_idx = np.unique(bp, return_index=True)
    bp = uniq_bp
    cm = cm[uniq_idx]
    morgans = cm / 100.0

    if bp[0] > 0.0:
        bp = np.concatenate(([0.0], bp))
        morgans = np.concatenate(([morgans[0]], morgans))
    if bp[-1] < float(seq_length):
        bp = np.concatenate((bp, [float(seq_length)]))
        morgans = np.concatenate((morgans, [morgans[-1]]))
    else:
        keep = bp <= float(seq_length)
        bp = bp[keep]
        morgans = morgans[keep]
        if bp[-1] < float(seq_length):
            bp = np.concatenate((bp, [float(seq_length)]))
            morgans = np.concatenate((morgans, [morgans[-1]]))

    delta_bp = np.diff(bp)
    delta_m = np.diff(morgans)
    if np.any(delta_bp <= 0.0):
        raise ValueError("genetic map has non-increasing bp coordinates")
    rates = np.clip(delta_m / delta_bp, 1e-12, None)
    return msprime.RateMap(position=bp.tolist(), rate=rates.tolist())


def _assert_variant_axis_aligned(
    source_order: list[str],
    source_panels: dict[str, PanelChromData],
) -> PanelChromData:
    ref = source_panels[source_order[0]]
    for source_name in source_order[1:]:
        panel = source_panels[source_name]
        if not np.array_equal(panel.variants.pos, ref.variants.pos):
            raise ValueError(
                f"msprime source panel {source_name!r} has marker position mismatch with {source_order[0]!r}"
            )
        if panel.variants.ref != ref.variants.ref or panel.variants.alt != ref.variants.alt:
            raise ValueError(
                f"msprime source panel {source_name!r} has REF/ALT mismatch with {source_order[0]!r}"
            )
    return ref


def _stable_uniform01(*parts: Any) -> float:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return (value % (10**12)) / float(10**12)


def _stable_mod(mod: int, *parts: Any) -> int:
    if mod <= 0:
        raise ValueError(f"mod must be > 0, got {mod}")
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int(value % int(mod))


def _pick_source_weighted(
    *,
    source_order: list[str],
    source_weights: list[float],
    seed: int,
    sample_index: int,
    hap_index: int,
    root: int,
    left_bp: float,
) -> str:
    u = _stable_uniform01(seed, sample_index, hap_index, root, f"{left_bp:.6f}")
    cumulative = 0.0
    for source_name, weight in zip(source_order, source_weights):
        cumulative += float(weight)
        if u < cumulative:
            return source_name
    return source_order[-1]


def _tree_root(tree: Any, node: int) -> int:
    root = int(node)
    parent = int(tree.parent(root))
    while parent != -1:
        root = parent
        parent = int(tree.parent(root))
    return root


def _append_or_extend_tract(
    *,
    tracts: list[dict[str, Any]],
    source_name: str,
    start_idx: int,
    end_idx: int,
    start_bp: float,
    end_bp: float,
) -> None:
    if end_idx <= start_idx:
        return
    if (
        tracts
        and tracts[-1]["source_population"] == source_name
        and int(tracts[-1]["end_marker_idx"]) == int(start_idx)
    ):
        tracts[-1]["end_marker_idx"] = int(end_idx)
        tracts[-1]["end_bp"] = float(end_bp)
    else:
        tracts.append(
            {
                "start_marker_idx": int(start_idx),
                "end_marker_idx": int(end_idx),
                "start_bp": float(start_bp),
                "end_bp": float(end_bp),
                "source_population": str(source_name),
            }
        )


def _population_id_to_name_from_ts(
    *,
    ts: Any,
    source_order: list[str],
    target_name: str,
) -> dict[int, str]:
    out: dict[int, str] = {}
    for pop_id, pop in enumerate(ts.populations()):
        name: str | None = None
        md = pop.metadata
        if isinstance(md, dict) and "name" in md:
            name = str(md["name"])
        elif isinstance(md, (bytes, bytearray)) and md:
            try:
                parsed = json.loads(md.decode("utf-8"))
                if isinstance(parsed, dict) and "name" in parsed:
                    name = str(parsed["name"])
            except Exception:
                name = None
        if name is None:
            if pop_id < len(source_order):
                name = str(source_order[pop_id])
            elif pop_id == len(source_order):
                name = str(target_name)
            else:
                name = f"POP{pop_id}"
        out[int(pop_id)] = str(name)
    return out


def simulate_msprime_tract_haplotypes(
    *,
    source_order: list[str],
    source_weights: list[float],
    source_panels: dict[str, PanelChromData],
    marker_positions: np.ndarray,
    map_bp: np.ndarray,
    map_cm: np.ndarray,
    n_haps: int,
    seed: int,
    chrom: int,
    population_name: str,
    admixture_time_generations: float | None,
    end_time_generations: float | None,
    copy_workers: int = 1,
) -> MsprimeTractSimulationResult:
    if n_haps <= 0:
        raise ValueError("n_haps must be > 0")
    if n_haps % 2 != 0:
        raise ValueError("n_haps must be even for diploid replicate simulation")
    if len(source_order) != len(source_weights):
        raise ValueError("source_order and source_weights length mismatch")
    if len(source_order) <= 0:
        raise ValueError("at least one msprime source is required")
    if int(copy_workers) <= 0:
        raise ValueError("copy_workers must be > 0")

    reference_panel = _assert_variant_axis_aligned(source_order, source_panels)
    marker_positions = np.asarray(marker_positions, dtype=np.int64, order="C")
    n_markers = int(marker_positions.size)
    if n_markers <= 0:
        raise ValueError("marker_positions cannot be empty")

    end_time = 10.0 if end_time_generations is None else float(end_time_generations)
    if end_time <= 0.0:
        raise ValueError("end_time_generations must be > 0")

    msprime = _import_msprime()
    sequence_length = float(max(int(marker_positions.max()) + 1, int(map_bp.max()) + 1))
    rate_map = _build_rate_map(
        msprime=msprime,
        map_bp=np.asarray(map_bp, dtype=np.float64),
        map_cm=np.asarray(map_cm, dtype=np.float64),
        seq_length=sequence_length,
    )

    demography = msprime.Demography()
    for source in source_order:
        demography.add_population(name=str(source), initial_size=10_000)
    target_name = f"__msprime_target__{population_name}"
    demography.add_population(name=target_name, initial_size=10_000)

    if len(source_order) > 1:
        if admixture_time_generations is None:
            raise ValueError("admixture_time_generations is required when len(source_order) > 1")
        boundary_time = float(admixture_time_generations)
        if end_time < boundary_time:
            raise ValueError("end_time_generations must be >= admixture_time_generations")
        demography.add_admixture(
            time=boundary_time,
            derived=target_name,
            ancestral=[str(x) for x in source_order],
            proportions=[float(x) for x in source_weights],
        )
    else:
        boundary_time = None
        demography.add_population_split(
            time=1.0,
            derived=[target_name],
            ancestral=str(source_order[0]),
        )
    demography.sort_events()

    source_name_set = set(source_order)

    n_diploids = int(n_haps // 2)
    marker_pos_f = marker_positions.astype(np.float64, copy=False)
    tracts_by_hap: list[list[dict[str, Any]]] = [[] for _ in range(int(n_haps))]

    replicates = msprime.sim_ancestry(
        samples={target_name: 1},
        demography=demography,
        recombination_rate=rate_map,
        ploidy=2,
        model="dtwf",
        random_seed=int(seed),
        end_time=float(end_time),
        num_replicates=int(n_diploids),
    )

    for sample_i, ts in enumerate(replicates):
        sample_nodes = list(ts.samples())
        if len(sample_nodes) != 2:
            raise ValueError(
                f"Replicate {sample_i}: expected 2 haplotypes for one diploid sample, got {len(sample_nodes)}"
            )
        pop_id_to_name = _population_id_to_name_from_ts(
            ts=ts,
            source_order=source_order,
            target_name=target_name,
        )

        current_source: list[str | None] = [None, None]
        current_left: list[float] = [0.0, 0.0]
        local_tracts: list[list[tuple[str, float, float]]] = [[], []]

        for tree in ts.trees():
            left = float(tree.interval.left)
            for hap_idx, sample_node in enumerate(sample_nodes):
                root = _tree_root(tree, int(sample_node))
                pop_id = int(ts.node(root).population)
                source_name = pop_id_to_name.get(pop_id)
                if source_name == target_name or source_name not in source_name_set:
                    source_name = _pick_source_weighted(
                        source_order=source_order,
                        source_weights=source_weights,
                        seed=int(seed),
                        sample_index=int(sample_i),
                        hap_index=int(hap_idx),
                        root=int(root),
                        left_bp=float(left),
                    )
                prev = current_source[hap_idx]
                if prev is None:
                    current_source[hap_idx] = str(source_name)
                    current_left[hap_idx] = float(left)
                elif source_name != prev:
                    local_tracts[hap_idx].append((str(prev), float(current_left[hap_idx]), float(left)))
                    current_source[hap_idx] = str(source_name)
                    current_left[hap_idx] = float(left)

        seq_len = float(ts.sequence_length)
        for hap_idx in (0, 1):
            source_name = current_source[hap_idx]
            if source_name is None:
                source_name = source_order[0]
            local_tracts[hap_idx].append((str(source_name), float(current_left[hap_idx]), float(seq_len)))

            hap_global = (2 * int(sample_i)) + int(hap_idx)
            global_tracts = tracts_by_hap[hap_global]
            for tract_source, left_bp, right_bp in local_tracts[hap_idx]:
                start_idx = int(np.searchsorted(marker_pos_f, left_bp, side="left"))
                end_idx = int(np.searchsorted(marker_pos_f, right_bp, side="left"))
                _append_or_extend_tract(
                    tracts=global_tracts,
                    source_name=str(tract_source),
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    start_bp=float(left_bp),
                    end_bp=float(right_bp),
                )

        if (sample_i + 1) % 1000 == 0 or (sample_i + 1) == n_diploids:
            logger.info(
                "msprime replicate progress population=%s completed=%s/%s",
                population_name,
                sample_i + 1,
                n_diploids,
            )

    synthetic = np.zeros((int(n_haps), n_markers), dtype=np.uint8)

    def _copy_haplotype_range(
        hap_start: int,
        hap_end: int,
    ) -> tuple[int, dict[str, int], dict[str, float], int, int, list[dict[str, Any]]]:
        local_source_marker_counts = {source: 0 for source in source_order}
        local_source_bp_totals = {source: 0.0 for source in source_order}
        local_total_tracts = 0
        local_total_markers = 0
        local_tract_rows: list[dict[str, Any]] = []
        for hap_index in range(int(hap_start), int(hap_end)):
            tracts = tracts_by_hap[hap_index]
            for tract_index, tract in enumerate(tracts):
                start_idx = int(tract["start_marker_idx"])
                end_idx = int(tract["end_marker_idx"])
                source = str(tract["source_population"])
                source_panel = source_panels[source]
                donor_hap_index = _stable_mod(
                    int(source_panel.donor_haps.shape[0]),
                    int(seed),
                    str(population_name),
                    int(chrom),
                    int(hap_index),
                    int(tract_index),
                    source,
                    start_idx,
                    end_idx,
                )
                synthetic[hap_index, start_idx:end_idx] = source_panel.donor_haps[donor_hap_index, start_idx:end_idx]

                marker_count = max(0, end_idx - start_idx)
                bp_length = max(0.0, float(tract["end_bp"]) - float(tract["start_bp"]))
                local_source_marker_counts[source] += int(marker_count)
                local_source_bp_totals[source] += float(bp_length)
                local_total_markers += int(marker_count)
                local_total_tracts += 1
                local_tract_rows.append(
                    {
                        "population": str(population_name),
                        "chrom": int(chrom),
                        "haplotype_index": int(hap_index),
                        "tract_index": int(tract_index),
                        "start_marker_idx": int(start_idx),
                        "end_marker_idx": int(end_idx),
                        "start_bp": float(tract["start_bp"]),
                        "end_bp": float(tract["end_bp"]),
                        "marker_count": int(marker_count),
                        "bp_length": float(bp_length),
                        "source_population": source,
                        "donor_haplotype_index": int(donor_hap_index),
                    }
                )
        return (
            int(hap_start),
            local_source_marker_counts,
            local_source_bp_totals,
            int(local_total_tracts),
            int(local_total_markers),
            local_tract_rows,
        )

    copy_workers = max(1, min(int(copy_workers), int(n_haps)))
    chunk_size = (int(n_haps) + copy_workers - 1) // copy_workers
    ranges = [
        (start, min(int(n_haps), start + chunk_size))
        for start in range(0, int(n_haps), chunk_size)
    ]

    results: list[tuple[int, dict[str, int], dict[str, float], int, int, list[dict[str, Any]]]] = []
    if len(ranges) == 1:
        results.append(_copy_haplotype_range(ranges[0][0], ranges[0][1]))
    else:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = [
                pool.submit(_copy_haplotype_range, start, end)
                for start, end in ranges
            ]
            for future in as_completed(futures):
                results.append(future.result())

    source_marker_counts = {source: 0 for source in source_order}
    source_bp_totals = {source: 0.0 for source in source_order}
    total_tracts = 0
    total_markers = 0
    tract_rows: list[dict[str, Any]] = []
    for _, local_marker_counts, local_bp_totals, local_total_tracts, local_total_markers, local_rows in sorted(
        results,
        key=lambda x: x[0],
    ):
        for source in source_order:
            source_marker_counts[source] += int(local_marker_counts.get(source, 0))
            source_bp_totals[source] += float(local_bp_totals.get(source, 0.0))
        total_tracts += int(local_total_tracts)
        total_markers += int(local_total_markers)
        tract_rows.extend(local_rows)

    observed_proportions = {
        source: (float(source_marker_counts[source]) / float(total_markers) if total_markers > 0 else float("nan"))
        for source in source_order
    }
    configured_weights = {source: float(weight) for source, weight in zip(source_order, source_weights)}
    source_abs_diff = {
        source: abs(float(observed_proportions[source]) - float(configured_weights[source]))
        if np.isfinite(observed_proportions[source])
        else float("nan")
        for source in source_order
    }

    n_diploids = int(n_haps // 2)
    marker_counts_by_diploid = {source: np.zeros(n_diploids, dtype=np.int64) for source in source_order}
    bp_totals_by_diploid = {source: np.zeros(n_diploids, dtype=np.float64) for source in source_order}
    cm_totals_by_diploid = {source: np.zeros(n_diploids, dtype=np.float64) for source in source_order}
    total_markers_by_diploid = np.zeros(n_diploids, dtype=np.int64)
    total_bp_by_diploid = np.zeros(n_diploids, dtype=np.float64)
    total_cm_by_diploid = np.zeros(n_diploids, dtype=np.float64)
    tract_lengths_bp: list[float] = []
    tract_lengths_cm: list[float] = []
    map_bp_arr = np.asarray(map_bp, dtype=np.float64)
    map_cm_arr = np.asarray(map_cm, dtype=np.float64)
    if map_bp_arr.size > 1:
        order = np.argsort(map_bp_arr)
        map_bp_arr = map_bp_arr[order]
        map_cm_arr = map_cm_arr[order]
        uniq_bp, uniq_idx = np.unique(map_bp_arr, return_index=True)
        map_bp_arr = uniq_bp
        map_cm_arr = map_cm_arr[uniq_idx]
        map_cm_arr = np.maximum.accumulate(map_cm_arr)
    for row in tract_rows:
        diploid_index = int(row["haplotype_index"]) // 2
        source = str(row["source_population"])
        marker_count = int(row["marker_count"])
        bp_length = float(row["bp_length"])
        cm_length = float(
            np.interp(float(row["end_bp"]), map_bp_arr, map_cm_arr)
            - np.interp(float(row["start_bp"]), map_bp_arr, map_cm_arr)
        )
        if cm_length < 0.0:
            cm_length = 0.0
        row["cm_length"] = cm_length
        if 0 <= diploid_index < n_diploids and source in marker_counts_by_diploid:
            marker_counts_by_diploid[source][diploid_index] += marker_count
            bp_totals_by_diploid[source][diploid_index] += bp_length
            cm_totals_by_diploid[source][diploid_index] += cm_length
            total_markers_by_diploid[diploid_index] += marker_count
            total_bp_by_diploid[diploid_index] += bp_length
            total_cm_by_diploid[diploid_index] += cm_length
        tract_lengths_bp.append(bp_length)
        tract_lengths_cm.append(cm_length)

    individual_ancestry_rows: list[dict[str, Any]] = []
    for diploid_index in range(n_diploids):
        row: dict[str, Any] = {
            "population": str(population_name),
            "chrom": int(chrom),
            "individual_index": int(diploid_index),
            "total_markers": int(total_markers_by_diploid[diploid_index]),
            "total_bp": float(total_bp_by_diploid[diploid_index]),
            "total_cm": float(total_cm_by_diploid[diploid_index]),
        }
        denom_markers = float(total_markers_by_diploid[diploid_index])
        denom_bp = float(total_bp_by_diploid[diploid_index])
        denom_cm = float(total_cm_by_diploid[diploid_index])
        for source in source_order:
            m_count = int(marker_counts_by_diploid[source][diploid_index])
            b_total = float(bp_totals_by_diploid[source][diploid_index])
            c_total = float(cm_totals_by_diploid[source][diploid_index])
            row[f"marker_count__{source}"] = m_count
            row[f"bp_total__{source}"] = b_total
            row[f"cm_total__{source}"] = c_total
            row[f"prop_markers__{source}"] = (
                float(m_count) / denom_markers if denom_markers > 0.0 else float("nan")
            )
            row[f"prop_bp__{source}"] = (
                b_total / denom_bp if denom_bp > 0.0 else float("nan")
            )
            row[f"prop_cm__{source}"] = (
                c_total / denom_cm if denom_cm > 0.0 else float("nan")
            )
        individual_ancestry_rows.append(row)

    tract_lengths_arr = np.asarray(tract_lengths_bp, dtype=np.float64)
    tract_lengths_arr = tract_lengths_arr[np.isfinite(tract_lengths_arr) & (tract_lengths_arr >= 0.0)]
    tract_lengths_cm_arr = np.asarray(tract_lengths_cm, dtype=np.float64)
    tract_lengths_cm_arr = tract_lengths_cm_arr[np.isfinite(tract_lengths_cm_arr) & (tract_lengths_cm_arr >= 0.0)]
    if tract_lengths_arr.size > 0:
        hist_bins = max(16, min(128, int(np.sqrt(int(tract_lengths_arr.size)))))
        max_len = float(np.max(tract_lengths_arr))
        if max_len <= 0.0:
            bin_edges = np.asarray([0.0, 1.0], dtype=np.float64)
            hist_counts = np.asarray([int(tract_lengths_arr.size)], dtype=np.int64)
        else:
            bin_edges = np.linspace(0.0, max_len, int(hist_bins) + 1, dtype=np.float64)
            hist_counts, _ = np.histogram(tract_lengths_arr, bins=bin_edges)
            hist_counts = hist_counts.astype(np.int64, copy=False)
    else:
        bin_edges = np.asarray([0.0, 1.0], dtype=np.float64)
        hist_counts = np.asarray([0], dtype=np.int64)

    if tract_lengths_cm_arr.size > 0:
        hist_bins_cm = max(16, min(128, int(np.sqrt(int(tract_lengths_cm_arr.size)))))
        max_len_cm = float(np.max(tract_lengths_cm_arr))
        if max_len_cm <= 0.0:
            bin_edges_cm = np.asarray([0.0, 1.0], dtype=np.float64)
            hist_counts_cm = np.asarray([int(tract_lengths_cm_arr.size)], dtype=np.int64)
        else:
            bin_edges_cm = np.linspace(0.0, max_len_cm, int(hist_bins_cm) + 1, dtype=np.float64)
            hist_counts_cm, _ = np.histogram(tract_lengths_cm_arr, bins=bin_edges_cm)
            hist_counts_cm = hist_counts_cm.astype(np.int64, copy=False)
    else:
        bin_edges_cm = np.asarray([0.0, 1.0], dtype=np.float64)
        hist_counts_cm = np.asarray([0], dtype=np.int64)

    individual_prop_means = {}
    individual_prop_variances = {}
    for source in source_order:
        with np.errstate(divide="ignore", invalid="ignore"):
            proportions = marker_counts_by_diploid[source].astype(np.float64) / np.maximum(
                total_markers_by_diploid.astype(np.float64), 1.0
            )
        proportions = np.where(total_markers_by_diploid > 0, proportions, np.nan)
        valid = proportions[np.isfinite(proportions)]
        if valid.size > 0:
            individual_prop_means[source] = float(np.mean(valid))
            individual_prop_variances[source] = float(np.var(valid, ddof=0))
        else:
            individual_prop_means[source] = float("nan")
            individual_prop_variances[source] = float("nan")

    chrom_span_bp = float(sequence_length)
    chrom_map_morgans = (
        float(np.max(np.asarray(map_cm, dtype=np.float64)) - np.min(np.asarray(map_cm, dtype=np.float64))) / 100.0
        if np.asarray(map_cm).size > 0
        else float("nan")
    )

    tract_summary = {
        "population": str(population_name),
        "chrom": int(chrom),
        "engine": "msprime_tracts",
        "is_admixed": bool(len(source_order) > 1),
        "n_haplotypes": int(n_haps),
        "n_markers": int(n_markers),
        "tract_count": int(total_tracts),
        "mean_tract_length_markers": (float(total_markers) / float(total_tracts) if total_tracts > 0 else float("nan")),
        "mean_tract_length_bp": (
            float(sum(source_bp_totals.values())) / float(total_tracts) if total_tracts > 0 else float("nan")
        ),
        "median_tract_length_bp": (
            float(np.median(tract_lengths_arr)) if tract_lengths_arr.size > 0 else float("nan")
        ),
        "p95_tract_length_bp": (
            float(np.percentile(tract_lengths_arr, 95.0)) if tract_lengths_arr.size > 0 else float("nan")
        ),
        "mean_tract_length_cm": (
            float(np.mean(tract_lengths_cm_arr)) if tract_lengths_cm_arr.size > 0 else float("nan")
        ),
        "median_tract_length_cm": (
            float(np.median(tract_lengths_cm_arr)) if tract_lengths_cm_arr.size > 0 else float("nan")
        ),
        "p95_tract_length_cm": (
            float(np.percentile(tract_lengths_cm_arr, 95.0)) if tract_lengths_cm_arr.size > 0 else float("nan")
        ),
        "configured_source_weights": configured_weights,
        "observed_source_proportions": observed_proportions,
        "source_weight_abs_diff": source_abs_diff,
        "source_marker_counts": {k: int(v) for k, v in source_marker_counts.items()},
        "source_bp_totals": {k: float(v) for k, v in source_bp_totals.items()},
        "per_individual_source_proportion_markers_mean": individual_prop_means,
        "per_individual_source_proportion_markers_variance": individual_prop_variances,
        "tract_length_bp_histogram": {
            "bin_edges_bp": [float(x) for x in bin_edges.tolist()],
            "counts": [int(x) for x in hist_counts.tolist()],
            "n_tracts": int(tract_lengths_arr.size),
        },
        "tract_length_cm_histogram": {
            "bin_edges_cm": [float(x) for x in bin_edges_cm.tolist()],
            "counts": [int(x) for x in hist_counts_cm.tolist()],
            "n_tracts": int(tract_lengths_cm_arr.size),
        },
        "chrom_span_bp": float(chrom_span_bp),
        "chrom_map_morgans": float(chrom_map_morgans),
        "admixture_time_generations": (
            float(admixture_time_generations) if admixture_time_generations is not None else None
        ),
        "end_time_generations": float(end_time),
        "simulation_model": "dtwf_replicates",
        "copy_workers": int(copy_workers),
        "reference_panel_chrom": str(reference_panel.variants.chrom),
    }

    return MsprimeTractSimulationResult(
        haplotypes=synthetic,
        tract_summary=tract_summary,
        tract_rows=tract_rows,
        individual_ancestry_rows=individual_ancestry_rows,
    )
