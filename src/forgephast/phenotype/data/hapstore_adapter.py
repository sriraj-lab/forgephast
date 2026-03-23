"""HapStore-backed genotype adapter for simulation-stage dosages and PCs."""

from __future__ import annotations

import concurrent.futures
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..config import HapStoreInputConfig
from .global_pcs import compute_global_pcs_svd, normalize_psam_fid_column

logger = logging.getLogger(__name__)

_CHR_POS_RE = re.compile(r"^chr(?P<chrom>\d+):(?P<pos>\d+)$")


class HapStoreAdapter:
    """Adapter for reading dosages/PCs from per-population per-chromosome HapStore files."""

    def __init__(self, hapstore_cfg: HapStoreInputConfig):
        self.cfg = hapstore_cfg
        self.layout = str(getattr(hapstore_cfg, "layout", "per_pop_per_chr"))
        self.root_dir = Path(hapstore_cfg.root_dir)
        self.file_pattern = hapstore_cfg.file_pattern
        self.store_path = (
            None if hapstore_cfg.store_path is None else Path(str(hapstore_cfg.store_path))
        )
        self.missing_sentinel = np.uint8(int(hapstore_cfg.missing_sentinel))
        self.missing_fill_value = float(hapstore_cfg.missing_fill_value)
        self.io_threads = max(1, int(getattr(hapstore_cfg, "io_threads", 1) or 1))
        self.last_export_allele_by_variant: dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._store_cache: dict[tuple[str, int], object] = {}
        self._sample_index_cache: dict[tuple[str, int], dict[str, int]] = {}
        self._variant_cache: dict[tuple[str, int], tuple[np.ndarray, list[str], list[str]]] = {}

    def set_io_threads(self, io_threads: int) -> None:
        """Update per-call I/O thread budget (safe within one grid group)."""
        self.io_threads = max(1, int(io_threads))

    def validate_files(self) -> None:
        """Validate root path exists for HapStore layout."""
        if self.layout == "per_pop_per_chr":
            if not self.root_dir.exists():
                raise FileNotFoundError(f"HapStore root_dir does not exist: {self.root_dir}")
        elif self.layout == "single_store":
            if self.store_path is None:
                raise ValueError("HapStoreAdapter requires store_path for layout=single_store")
            if not self.store_path.exists():
                raise FileNotFoundError(f"HapStore store_path does not exist: {self.store_path}")
        else:
            raise ValueError(f"Unsupported HapStoreAdapter layout={self.layout!r}")

    def _cache_population_key(self, population: str) -> str:
        if self.layout == "single_store":
            return "__single_store__"
        return str(population)

    def _open_store(self, population: str, chrom: int) -> object:
        pop_key = self._cache_population_key(population)
        key = (pop_key, 0) if self.layout == "single_store" else (pop_key, int(chrom))
        with self._cache_lock:
            if key in self._store_cache:
                return self._store_cache[key]

        if self.layout == "single_store":
            if self.store_path is None:
                raise ValueError("Missing store_path for HapStore layout=single_store")
            path = self.store_path
        else:
            path = self.root_dir / self.file_pattern.format(population=population, chrom=chrom)
        if not path.exists():
            raise FileNotFoundError(f"Missing HapStore path for population={population} chrom={chrom}: {path}")
        try:
            from hapstore import HapStore  # type: ignore
        except Exception as exc:  # noqa: BLE001 - provide clear setup guidance
            raise RuntimeError(
                "HapStore support requires the hapstore package"
            ) from exc

        store = HapStore(path, mode="r")
        if not store.has_chromosome(chrom):
            raise ValueError(f"HapStore {path} does not contain chromosome {chrom}")
        with self._cache_lock:
            self._store_cache[key] = store
        return store

    def _get_sample_index_map(self, population: str, chrom: int) -> dict[str, int]:
        pop_key = self._cache_population_key(population)
        key = (pop_key, int(chrom))
        with self._cache_lock:
            if key in self._sample_index_cache:
                return self._sample_index_cache[key]
        store = self._open_store(population, chrom)
        sample_ids = [str(x) for x in store.sample_ids(chrom)]
        mapping = {iid: idx for idx, iid in enumerate(sample_ids)}
        with self._cache_lock:
            self._sample_index_cache[key] = mapping
        return mapping

    def _get_variant_arrays(self, population: str, chrom: int) -> tuple[np.ndarray, list[str], list[str]]:
        pop_key = self._cache_population_key(population)
        key = (pop_key, int(chrom))
        with self._cache_lock:
            if key in self._variant_cache:
                return self._variant_cache[key]
        store = self._open_store(population, chrom)
        pos = np.asarray(store.variant_positions(chrom), dtype=np.int64)
        info = store.variant_info(chrom)
        ref = [str(x) for x in info.get("ref", [])]
        alt = [str(x) for x in info.get("alt", [])]
        if len(ref) != len(pos) or len(alt) != len(pos):
            raise ValueError(
                f"HapStore variant metadata length mismatch for population={population} chrom={chrom}"
            )
        if pos.shape[0] > 1 and not bool(np.all(pos[1:] >= pos[:-1])):
            raise ValueError(
                f"HapStore positions are not sorted for population={population} chrom={chrom}"
            )
        out = (pos, ref, alt)
        with self._cache_lock:
            self._variant_cache[key] = out
        return out

    def _effective_io_threads(self, n_chroms: int) -> int:
        if self.layout == "single_store":
            return 1
        return min(max(1, int(self.io_threads)), max(1, int(n_chroms)))

    def _parse_variant_id(self, variant_id: str) -> tuple[int, int]:
        match = _CHR_POS_RE.match(str(variant_id))
        if not match:
            raise ValueError(
                f"HapStore mode expects variant IDs as chrN:POS; got {variant_id!r}"
            )
        return int(match.group("chrom")), int(match.group("pos"))

    def _resolve_indices_and_alt(
        self,
        *,
        population: str,
        chrom: int,
        variant_ids: list[str],
    ) -> tuple[list[int], list[str]]:
        pos_array, _, alt_array = self._get_variant_arrays(population, chrom)
        resolved_idx: list[int] = []
        resolved_alt: list[str] = []
        for variant_id in variant_ids:
            _, pos = self._parse_variant_id(variant_id)
            idx = int(np.searchsorted(pos_array, int(pos), side="left"))
            if idx < 0 or idx >= int(pos_array.shape[0]) or int(pos_array[idx]) != int(pos):
                raise ValueError(
                    f"Variant {variant_id} not found in HapStore for population={population} chrom={chrom}"
                )
            alt = str(alt_array[idx])
            if alt in {"", "."}:
                raise ValueError(
                    f"Missing ALT allele in HapStore for variant {variant_id} population={population}"
                )
            resolved_idx.append(idx)
            resolved_alt.append(alt)
        return resolved_idx, resolved_alt

    def _resolve_sample_indices(
        self,
        population: str,
        chrom: int,
        sample_ids: list[str],
    ) -> np.ndarray:
        idx_map = self._get_sample_index_map(population, chrom)
        if not sample_ids:
            raise ValueError(f"No samples provided for population={population}")

        # Duplicate IIDs would silently overwrite rows; treat as a hard error.
        unique = set(sample_ids)
        if len(unique) != len(sample_ids):
            raise ValueError(f"Duplicate sample IDs found for population={population}")

        indices = np.empty(len(sample_ids), dtype=np.int64)
        for i, iid in enumerate(sample_ids):
            if iid not in idx_map:
                raise ValueError(
                    f"IID {iid} is not present in HapStore population={population} chrom={chrom}"
                )
            indices[i] = int(idx_map[iid])
        return indices

    @staticmethod
    def _fetch_genotypes_for_indices(
        store: object,
        chrom: int,
        variant_indices: list[int],
        sample_indices: np.ndarray,
        n_store_samples: int,
        *,
        coalesce_max_gap_variants: int = 0,
        max_slice_variants: int = 0,
    ) -> np.ndarray:
        idx = np.asarray(variant_indices, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((int(sample_indices.shape[0]), 0), dtype=np.uint8)

        mask = np.zeros(int(n_store_samples), dtype=bool)
        mask[np.asarray(sample_indices, dtype=np.int64)] = True
        selected_sorted = np.flatnonzero(mask)

        order = np.argsort(idx)
        idx_sorted = idx[order]
        out_sorted = np.zeros(
            (int(selected_sorted.shape[0]), int(idx_sorted.shape[0])),
            dtype=np.uint8,
        )

        max_gap = max(0, int(coalesce_max_gap_variants))
        max_slice = max(0, int(max_slice_variants))
        run_start = 0
        n = int(idx_sorted.shape[0])
        while run_start < n:
            run_end = run_start
            left = int(idx_sorted[run_start])
            right = left + 1
            while True:
                next_i = run_end + 1
                if next_i >= n:
                    break
                gap = int(idx_sorted[next_i]) - int(idx_sorted[run_end])
                if gap > 1 + max_gap:
                    break
                prospective_right = int(idx_sorted[next_i]) + 1
                if max_slice and (prospective_right - left) > max_slice:
                    break
                run_end = next_i
                right = prospective_right

            block = np.asarray(
                store.genotypes(chrom, variant_slice=(left, right), sample_mask=mask),
                dtype=np.uint8,
            )
            cols = idx_sorted[run_start : run_end + 1] - left
            out_sorted[:, run_start : run_end + 1] = block[:, cols]
            run_start = run_end + 1

        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.shape[0])
        out_variants_ordered = out_sorted[:, inverse]

        pos = np.searchsorted(selected_sorted, np.asarray(sample_indices, dtype=np.int64))
        if not bool(np.all(selected_sorted[pos] == np.asarray(sample_indices, dtype=np.int64))):
            raise ValueError("Failed to re-order HapStore genotypes to requested sample order")
        return out_variants_ordered[pos, :]

    @staticmethod
    def _fetch_haplotypes_for_indices(
        store: object,
        chrom: int,
        variant_indices: list[int],
        sample_indices: np.ndarray,
        n_store_samples: int,
        *,
        coalesce_max_gap_variants: int = 0,
        max_slice_variants: int = 0,
    ) -> np.ndarray:
        idx = np.asarray(variant_indices, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((int(sample_indices.shape[0]), 2, 0), dtype=np.uint8)

        mask = np.zeros(int(n_store_samples), dtype=bool)
        mask[np.asarray(sample_indices, dtype=np.int64)] = True
        selected_sorted = np.flatnonzero(mask)

        order = np.argsort(idx)
        idx_sorted = idx[order]
        out_sorted = np.zeros(
            (int(selected_sorted.shape[0]), 2, int(idx_sorted.shape[0])),
            dtype=np.uint8,
        )

        max_gap = max(0, int(coalesce_max_gap_variants))
        max_slice = max(0, int(max_slice_variants))
        run_start = 0
        n = int(idx_sorted.shape[0])
        while run_start < n:
            run_end = run_start
            left = int(idx_sorted[run_start])
            right = left + 1
            while True:
                next_i = run_end + 1
                if next_i >= n:
                    break
                gap = int(idx_sorted[next_i]) - int(idx_sorted[run_end])
                if gap > 1 + max_gap:
                    break
                prospective_right = int(idx_sorted[next_i]) + 1
                if max_slice and (prospective_right - left) > max_slice:
                    break
                run_end = next_i
                right = prospective_right

            block = np.asarray(
                store.haplotypes(chrom, variant_slice=(left, right), sample_mask=mask),
                dtype=np.uint8,
            )
            if block.shape[0] != int(selected_sorted.shape[0]) * 2:
                raise ValueError("Unexpected HapStore haplotype array shape")
            block = block.reshape(int(selected_sorted.shape[0]), 2, block.shape[1])
            cols = idx_sorted[run_start : run_end + 1] - left
            out_sorted[:, :, run_start : run_end + 1] = block[:, :, cols]
            run_start = run_end + 1

        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.shape[0])
        out_variants_ordered = out_sorted[:, :, inverse]

        pos = np.searchsorted(selected_sorted, np.asarray(sample_indices, dtype=np.int64))
        if not bool(np.all(selected_sorted[pos] == np.asarray(sample_indices, dtype=np.int64))):
            raise ValueError("Failed to re-order HapStore haplotypes to requested sample order")
        return out_variants_ordered[pos, :, :]

    @staticmethod
    def _group_rows_by_population(
        sample_ids: list[str],
        sample_populations: list[str],
    ) -> dict[str, list[int]]:
        if len(sample_ids) != len(sample_populations):
            raise ValueError(
                f"sample_ids/sample_populations length mismatch: {len(sample_ids)} != {len(sample_populations)}"
            )
        grouped: dict[str, list[int]] = {}
        for row_idx, pop in enumerate(sample_populations):
            grouped.setdefault(str(pop), []).append(row_idx)
        return grouped

    def _variant_layout(
        self,
        *,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
    ) -> tuple[dict[str, list[int]], dict[int, list[str]], dict[str, int]]:
        if not variant_ids:
            raise ValueError("variant_ids cannot be empty")
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("HapStore export requires unique variant IDs")

        if self.layout == "single_store":
            pop_rows = {"__single_store__": list(range(len(sample_ids)))}
        else:
            pop_rows = self._group_rows_by_population(sample_ids, sample_populations)

        variant_by_chrom: dict[int, list[str]] = {}
        for variant_id in variant_ids:
            chrom, _ = self._parse_variant_id(variant_id)
            variant_by_chrom.setdefault(chrom, []).append(variant_id)
        col_index = {variant_id: idx for idx, variant_id in enumerate(variant_ids)}
        return pop_rows, variant_by_chrom, col_index

    def _export_variant_array(
        self,
        *,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        kind: str,
    ) -> tuple[np.ndarray, dict[str, str]]:
        pop_rows, variant_by_chrom, col_index = self._variant_layout(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            variant_ids=variant_ids,
        )
        allele_by_variant: dict[str, str] = {}

        if kind == "genotypes":
            out = np.zeros((len(sample_ids), len(variant_ids)), dtype=np.uint8)
        elif kind == "haplotypes":
            out = np.zeros((len(sample_ids), len(variant_ids), 2), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported HapStore export kind={kind!r}")

        for population, row_ids in pop_rows.items():
            cohort_iids = [sample_ids[i] for i in row_ids]
            row_indices = np.asarray(row_ids, dtype=np.int64)

            def _work(chrom_item: tuple[int, list[str]]) -> tuple[list[str], np.ndarray, list[str]]:
                chrom, chrom_variant_ids = chrom_item
                store = self._open_store(population, chrom)
                idx_map = self._get_sample_index_map(population, chrom)
                sample_indices = self._resolve_sample_indices(population, chrom, cohort_iids)
                variant_indices, alt_alleles = self._resolve_indices_and_alt(
                    population=population,
                    chrom=chrom,
                    variant_ids=chrom_variant_ids,
                )
                if kind == "genotypes":
                    block = self._fetch_genotypes_for_indices(
                        store=store,
                        chrom=chrom,
                        variant_indices=variant_indices,
                        sample_indices=sample_indices,
                        n_store_samples=len(idx_map),
                        coalesce_max_gap_variants=int(
                            getattr(self.cfg, "coalesce_max_gap_variants", 0) or 0
                        ),
                        max_slice_variants=int(getattr(self.cfg, "max_slice_variants", 0) or 0),
                    )
                else:
                    block = self._fetch_haplotypes_for_indices(
                        store=store,
                        chrom=chrom,
                        variant_indices=variant_indices,
                        sample_indices=sample_indices,
                        n_store_samples=len(idx_map),
                        coalesce_max_gap_variants=int(
                            getattr(self.cfg, "coalesce_max_gap_variants", 0) or 0
                        ),
                        max_slice_variants=int(getattr(self.cfg, "max_slice_variants", 0) or 0),
                    )
                return chrom_variant_ids, block, alt_alleles

            threads = self._effective_io_threads(len(variant_by_chrom))
            items = list(variant_by_chrom.items())
            if threads <= 1 or len(items) <= 1:
                results = [_work(item) for item in items]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=int(threads)) as ex:
                    results = list(ex.map(_work, items))

            for chrom_variant_ids, block, alt_alleles in results:
                for variant_id, alt in zip(chrom_variant_ids, alt_alleles):
                    prev = allele_by_variant.get(variant_id)
                    if prev is None:
                        allele_by_variant[variant_id] = alt
                    elif prev != alt:
                        raise ValueError(
                            f"ALT allele mismatch across populations for {variant_id}: {prev} vs {alt}"
                        )
                cols = np.asarray([col_index[v] for v in chrom_variant_ids], dtype=np.int64)
                if kind == "genotypes":
                    out[np.ix_(row_indices, cols)] = block
                else:
                    out[np.ix_(row_indices, cols, np.array([0, 1], dtype=np.int64))] = np.transpose(
                        block, (0, 2, 1)
                    )

        return out, allele_by_variant

    def export_dosages(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Export additive dosages for requested variants from HapStore."""
        del out_dir
        del chromosomes
        geno_u8, allele_by_variant = self._export_variant_array(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            variant_ids=variant_ids,
            kind="genotypes",
        )
        dosage = geno_u8.astype(np.float32)
        dosage[geno_u8 == self.missing_sentinel] = np.float32(self.missing_fill_value)
        out = pd.DataFrame(dosage, columns=variant_ids, copy=False)
        out.insert(0, "IID", sample_ids)
        out.insert(0, "FID", sample_ids)
        self.last_export_allele_by_variant = allele_by_variant
        return out

    def export_genotypes_u8(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Export additive genotypes as uint8 matrix (n_samples, n_variants).

        Values are in {0,1,2}, with missing mapped to 0.

        Note: this avoids DataFrame construction + float conversions in export_dosages and is
        intended for high-throughput internal computations.
        """
        del out_dir
        del chromosomes
        geno, allele_by_variant = self._export_variant_array(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            variant_ids=variant_ids,
            kind="genotypes",
        )
        geno = np.where(geno == self.missing_sentinel, np.uint8(0), geno).astype(np.uint8, copy=False)
        self.last_export_allele_by_variant = allele_by_variant
        return geno

    def export_haplotypes(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Export phased haplotypes for requested variants from HapStore.

        Returns a uint8 array of shape (n_samples, n_variants, 2) with values in {0,1}
        and missing values mapped to 0.
        """
        del out_dir
        del chromosomes
        hap, allele_by_variant = self._export_variant_array(
            sample_ids=sample_ids,
            sample_populations=sample_populations,
            variant_ids=variant_ids,
            kind="haplotypes",
        )
        hap = np.where(hap == self.missing_sentinel, np.uint8(0), hap).astype(np.uint8, copy=False)
        self.last_export_allele_by_variant = allele_by_variant
        return hap

    def _allocate_variant_counts(
        self,
        chrom_to_n_variants: dict[int, int],
        max_variants: int,
    ) -> dict[int, int]:
        total = int(sum(chrom_to_n_variants.values()))
        if total <= int(max_variants):
            return {chrom: int(n) for chrom, n in chrom_to_n_variants.items()}

        alloc: dict[int, int] = {chrom: 0 for chrom in chrom_to_n_variants}
        fracs: list[tuple[float, int]] = []
        for chrom, n in chrom_to_n_variants.items():
            raw = (float(n) / float(total)) * float(max_variants)
            base = min(int(n), int(np.floor(raw)))
            alloc[chrom] = base
            fracs.append((raw - base, chrom))

        assigned = int(sum(alloc.values()))
        remaining = int(max_variants) - assigned
        for _, chrom in sorted(fracs, key=lambda x: (-x[0], x[1])):
            if remaining <= 0:
                break
            if alloc[chrom] >= int(chrom_to_n_variants[chrom]):
                continue
            alloc[chrom] += 1
            remaining -= 1

        # Guarantee at least one variant per chromosome when possible.
        for chrom, n in chrom_to_n_variants.items():
            if n > 0 and alloc[chrom] == 0 and int(max_variants) >= len(chrom_to_n_variants):
                donor = max(alloc, key=lambda c: alloc[c])
                if alloc[donor] > 1:
                    alloc[donor] -= 1
                    alloc[chrom] = 1

        return alloc

    def _select_variant_indices_for_pca(
        self,
        *,
        populations: list[str],
        chromosomes: Sequence[int],
        max_variants: int,
        random_seed: int,
    ) -> dict[int, np.ndarray]:
        if not populations:
            raise ValueError("No populations provided for HapStore PCA")

        ref_pop = populations[0]
        chrom_to_n: dict[int, int] = {}
        for chrom in chromosomes:
            store = self._open_store(ref_pop, int(chrom))
            chrom_to_n[int(chrom)] = int(store.n_variants(int(chrom)))

        alloc = self._allocate_variant_counts(chrom_to_n, int(max_variants))

        selected: dict[int, np.ndarray] = {}
        for chrom in chromosomes:
            chrom = int(chrom)
            n = int(chrom_to_n[chrom])
            k = int(alloc[chrom])
            if k <= 0:
                selected[chrom] = np.zeros(0, dtype=np.int64)
                continue
            if k >= n:
                selected[chrom] = np.arange(n, dtype=np.int64)
                continue
            rng = np.random.default_rng(int(random_seed) + (chrom * 9973))
            idx = np.asarray(rng.choice(n, size=k, replace=False), dtype=np.int64)
            idx.sort()
            selected[chrom] = idx
        return selected

    def compute_global_pcs(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        n_components: int,
        out_dir: Path,
        mode_cfg: object,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Compute global PCs with SVD from HapStore dosages."""
        del out_dir
        if not chromosomes:
            raise ValueError("chromosomes must be provided for HapStore PCA compute mode")

        if self.layout == "single_store":
            pop_rows = {"__single_store__": list(range(len(sample_ids)))}
        else:
            pop_rows = self._group_rows_by_population(sample_ids, sample_populations)
        populations = list(pop_rows.keys())

        max_variants = int(getattr(mode_cfg, "svd_max_variants", 50000))
        min_maf = float(getattr(mode_cfg, "svd_min_maf", 0.01))
        random_seed = int(getattr(mode_cfg, "svd_random_seed", 42))

        selected = self._select_variant_indices_for_pca(
            populations=populations,
            chromosomes=chromosomes,
            max_variants=max_variants,
            random_seed=random_seed,
        )

        n_total = int(sum(int(v.shape[0]) for v in selected.values()))
        if n_total <= 0:
            raise ValueError("No HapStore variants selected for PCA")

        x = np.full((len(sample_ids), n_total), np.nan, dtype=np.float64)

        col_offset = 0
        for chrom in chromosomes:
            chrom = int(chrom)
            idx = selected[chrom]
            width = int(idx.shape[0])
            if width == 0:
                continue

            ref_store = self._open_store(populations[0], chrom)
            ref_pos = np.asarray(ref_store.variant_positions(chrom), dtype=np.int64)

            for population, row_ids in pop_rows.items():
                cohort_iids = [sample_ids[i] for i in row_ids]
                row_indices = np.asarray(row_ids, dtype=np.int64)

                store = self._open_store(population, chrom)
                idx_map = self._get_sample_index_map(population, chrom)
                pop_pos = np.asarray(store.variant_positions(chrom), dtype=np.int64)
                if pop_pos.shape != ref_pos.shape or not np.array_equal(pop_pos[idx], ref_pos[idx]):
                    raise ValueError(
                        f"HapStore variant position mismatch for population={population} chrom={chrom}"
                    )

                sample_indices = self._resolve_sample_indices(population, chrom, cohort_iids)
                g_u8 = self._fetch_genotypes_for_indices(
                    store=store,
                    chrom=chrom,
                    variant_indices=idx.tolist(),
                    sample_indices=sample_indices,
                    n_store_samples=len(idx_map),
                )
                g = g_u8.astype(np.float64)
                g[g_u8 == self.missing_sentinel] = np.nan
                x[np.ix_(row_indices, np.arange(col_offset, col_offset + width, dtype=np.int64))] = g

            col_offset += width

        return compute_global_pcs_svd(
            genotypes=x,
            sample_ids=sample_ids,
            n_components=int(n_components),
            max_variants=max_variants,
            min_maf=min_maf,
            random_seed=random_seed,
        )

    def export_to_pgen(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        out_prefix: Path,
        chromosomes: Sequence[int] | None = None,
        variant_ids: List[str] | None = None,
        export_cfg: object | None = None,
    ) -> str:
        """Export cohort samples from HapStore to a canonical multi-chromosome PGEN prefix."""
        if not chromosomes:
            raise ValueError("chromosomes must be provided for HapStore->PGEN export")

        try:
            from hapstore import HapStore, concat_hapstores
            from hapstore.pgen import export_pgen
        except Exception as exc:  # noqa: BLE001 - preserve setup guidance
            raise RuntimeError(
                "HapStore->PGEN export requires the hapstore package"
            ) from exc

        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pgen", ".pvar", ".psam"):
            Path(f"{out_prefix}{ext}").unlink(missing_ok=True)
        plink2_path = str(getattr(export_cfg, "plink2_path", "plink2"))
        merge_threads = int(getattr(export_cfg, "threads", 1))
        concat_workers = int(getattr(export_cfg, "concat_workers", 1))
        pgen_shard_workers = int(getattr(export_cfg, "pgen_shard_workers", 1))

        if self.layout == "single_store":
            if self.store_path is None:
                raise ValueError("HapStore->PGEN export requires store_path for layout=single_store")
            chrom_prefixes: list[Path] = []
            variant_ids_by_chrom: dict[int, list[str]] = {}
            if variant_ids is not None:
                for variant_id in variant_ids:
                    chrom_id, _ = self._parse_variant_id(str(variant_id))
                    variant_ids_by_chrom.setdefault(int(chrom_id), []).append(str(variant_id))
            for chrom in chromosomes:
                chrom = int(chrom)
                requested_ids = (
                    variant_ids_by_chrom.get(chrom) if variant_ids is not None else None
                )
                if variant_ids is not None and not requested_ids:
                    continue
                chr_prefix = out_prefix.parent / f"{out_prefix.name}.chr{chrom}"
                for ext in (".pgen", ".pvar", ".psam"):
                    Path(f"{chr_prefix}{ext}").unlink(missing_ok=True)
                export_pgen(
                    store_path=str(self.store_path),
                    out_prefix=chr_prefix,
                    chrom=chrom,
                    sample_ids=list(sample_ids),
                    variant_ids=list(requested_ids) if requested_ids is not None else None,
                    on_partial_missing="missing",
                    n_workers=max(1, pgen_shard_workers),
                    plink2_workers=max(1, merge_threads),
                    plink2_path=plink2_path,
                )
                normalize_psam_fid_column(Path(f"{chr_prefix}.psam"))
                chrom_prefixes.append(chr_prefix)

            if not chrom_prefixes:
                raise ValueError("No chromosome shards were exported to PGEN")

            if len(chrom_prefixes) == 1:
                only = chrom_prefixes[0]
                for ext in (".pgen", ".pvar", ".psam"):
                    shutil.move(f"{only}{ext}", f"{out_prefix}{ext}")
                normalize_psam_fid_column(Path(f"{out_prefix}.psam"))
                return str(out_prefix)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".pmerge.list",
                delete=False,
                encoding="utf-8",
            ) as handle:
                merge_list = Path(handle.name)
                for prefix in chrom_prefixes:
                    handle.write(f"{prefix}\n")

            cmd = [
                plink2_path,
                "--pmerge-list",
                str(merge_list),
                "--make-pgen",
                "--threads",
                str(merge_threads),
                "--out",
                str(out_prefix),
            ]
            logger.info("Merging HapStore-exported PGEN shards: %s", " ".join(cmd))
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    "PLINK2 pmerge failed\n"
                    f"CMD: {' '.join(cmd)}\n"
                    f"STDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )
            return str(out_prefix)

        pop_rows = self._group_rows_by_population(sample_ids, sample_populations)
        population_order = list(dict.fromkeys(str(pop) for pop in sample_populations))
        populations = [pop for pop in population_order if pop in pop_rows]
        if not populations:
            raise ValueError("No populations available for HapStore->PGEN export")

        def _concat_layout_overrides(input_stores: list[str], chrom: int) -> dict[str, int]:
            chrom_key = f"chr{int(chrom)}"
            compression_levels: list[int] = []
            variant_chunks: list[int] = []
            hap_chunks: list[int | None] = []
            sample_chunks: list[int | None] = []
            for path_str in input_stores:
                attrs = HapStore(path_str, mode="r").attrs(chrom_key)
                compression_levels.append(int(attrs.get("compression_level", 3)))
                variant_chunks.append(int(attrs.get("chunk_variant_size", 500)))
                hap_chunk = attrs.get("chunk_haplotype_size", None)
                sample_chunk = attrs.get("chunk_sample_size", None)
                hap_chunks.append(None if hap_chunk is None else int(hap_chunk))
                sample_chunks.append(None if sample_chunk is None else int(sample_chunk))

            overrides: dict[str, int] = {}
            if len(set(compression_levels)) > 1:
                overrides["compression_level"] = max(compression_levels)
            if len(set(variant_chunks)) > 1:
                overrides["chunk_variant_size"] = max(variant_chunks)
            if len(set(hap_chunks)) > 1:
                non_null = [int(x) for x in hap_chunks if x is not None]
                if non_null:
                    overrides["chunk_haplotype_size"] = max(non_null)
            if len(set(sample_chunks)) > 1:
                non_null = [int(x) for x in sample_chunks if x is not None]
                if non_null:
                    overrides["chunk_sample_size"] = max(non_null)
            return overrides

        variant_ids_by_chrom: dict[int, list[str]] = {}
        if variant_ids is not None:
            for variant_id in variant_ids:
                chrom, _ = self._parse_variant_id(str(variant_id))
                variant_ids_by_chrom.setdefault(chrom, []).append(str(variant_id))

        try:
            chrom_prefixes: list[Path] = []
            for chrom in chromosomes:
                chrom = int(chrom)
                requested_ids = variant_ids_by_chrom.get(chrom) if variant_ids is not None else None
                if variant_ids is not None and not requested_ids:
                    continue
                input_stores = [
                    str(self.root_dir / self.file_pattern.format(population=pop, chrom=chrom))
                    for pop in populations
                ]
                for path_str in input_stores:
                    if not Path(path_str).exists():
                        raise FileNotFoundError(
                            f"Missing HapStore input for export: {path_str}"
                        )
                merged_store_path = (
                    out_prefix.parent / f"{out_prefix.name}.merged_input.chr{chrom}.hapstore.zarr"
                )
                if merged_store_path.exists():
                    shutil.rmtree(merged_store_path, ignore_errors=True)
                concat_overrides = _concat_layout_overrides(input_stores, chrom)
                concat_hapstores(
                    input_stores=input_stores,
                    output_store=merged_store_path,
                    chrom=chrom,
                    axis="samples",
                    n_workers=max(1, concat_workers),
                    overwrite=True,
                    store_genotypes=False,
                    **concat_overrides,
                )
                chr_prefix = out_prefix.parent / f"{out_prefix.name}.chr{chrom}"
                for ext in (".pgen", ".pvar", ".psam"):
                    Path(f"{chr_prefix}{ext}").unlink(missing_ok=True)

                export_pgen(
                    store_path=merged_store_path,
                    out_prefix=chr_prefix,
                    chrom=chrom,
                    sample_ids=sample_ids,
                    variant_ids=requested_ids,
                    on_partial_missing="missing",
                    n_workers=max(1, pgen_shard_workers),
                    plink2_workers=max(1, merge_threads),
                    plink2_path=plink2_path,
                )
                normalize_psam_fid_column(Path(f"{chr_prefix}.psam"))
                shutil.rmtree(merged_store_path, ignore_errors=True)
                chrom_prefixes.append(chr_prefix)

            if not chrom_prefixes:
                raise ValueError("No chromosome shards were exported to PGEN")

            if len(chrom_prefixes) == 1:
                only = chrom_prefixes[0]
                for ext in (".pgen", ".pvar", ".psam"):
                    shutil.move(f"{only}{ext}", f"{out_prefix}{ext}")
                normalize_psam_fid_column(Path(f"{out_prefix}.psam"))
            else:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".pmerge.list",
                    delete=False,
                    encoding="utf-8",
                ) as handle:
                    merge_list = Path(handle.name)
                    for prefix in chrom_prefixes:
                        handle.write(f"{prefix}\n")

                try:
                    cmd = [
                        plink2_path,
                        "--threads",
                        str(max(1, merge_threads)),
                        "--pmerge-list",
                        str(merge_list),
                        "pfile",
                        "--indiv-sort",
                        "none",
                        "--make-pgen",
                        "--out",
                        str(out_prefix),
                    ]
                    completed = subprocess.run(cmd, capture_output=True, text=True)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            "PLINK2 --pmerge-list failed during HapStore->PGEN export\n"
                            f"CMD: {' '.join(cmd)}\n"
                            f"STDOUT:\n{completed.stdout}\n"
                            f"STDERR:\n{completed.stderr}"
                        )
                finally:
                    merge_list.unlink(missing_ok=True)

            for ext in (".pgen", ".pvar", ".psam"):
                if not Path(f"{out_prefix}{ext}").exists():
                    raise RuntimeError(
                        f"HapStore->PGEN export missing output artifact: {out_prefix}{ext}"
                    )
            normalize_psam_fid_column(Path(f"{out_prefix}.psam"))
            return str(out_prefix)
        finally:
            for chrom in chromosomes:
                chrom = int(chrom)
                merged_store_path = (
                    out_prefix.parent / f"{out_prefix.name}.merged_input.chr{chrom}.hapstore.zarr"
                )
                shutil.rmtree(merged_store_path, ignore_errors=True)
                shard_prefix = out_prefix.parent / f"{out_prefix.name}.chr{chrom}"
                for ext in (".pgen", ".pvar", ".psam"):
                    Path(f"{shard_prefix}{ext}").unlink(missing_ok=True)
