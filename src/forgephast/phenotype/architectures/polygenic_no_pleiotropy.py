"""Phase-1 polygenic trait plugin without pleiotropy."""

from __future__ import annotations

import numpy as np
import pandas as pd
try:  # pragma: no cover - optional dependency
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    njit = None
    _HAS_NUMBA = False

from .base import ArchitectureInput, ArchitecturePlugin


if _HAS_NUMBA:

    @njit(cache=True)
    def _la_genetic_numba(hap: np.ndarray, anc: np.ndarray, betas: np.ndarray) -> np.ndarray:
        n_samples = int(hap.shape[0])
        n_variants = int(hap.shape[1])
        out = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            total = 0.0
            for v in range(n_variants):
                a0 = int(anc[i, v, 0])
                a1 = int(anc[i, v, 1])
                total += float(hap[i, v, 0]) * float(betas[v, a0])
                total += float(hap[i, v, 1]) * float(betas[v, a1])
            out[i] = total
        return out


class PolygenicNoPleiotropyPlugin(ArchitecturePlugin):
    """Polygenic sum across selected variants with population-specific betas."""

    @staticmethod
    def _build_effects_detail(data: ArchitectureInput) -> pd.DataFrame:
        variant_col = data.variant_id_column
        effects_detail = data.membership_df[
            [variant_col, "maf_bucket", "membership_type", "replicate"]
        ].merge(
            data.beta_df,
            left_on=variant_col,
            right_on="variant_id",
            how="left",
        )
        if variant_col != "variant_id":
            effects_detail = effects_detail.drop(columns=["variant_id"])
            effects_detail = effects_detail.rename(columns={variant_col: "variant_id"})
        return effects_detail

    @staticmethod
    def _compute_local_ancestry_genetic_component(data: ArchitectureInput) -> np.ndarray:
        if data.haplotypes is None or data.local_ancestry_codes is None:
            raise ValueError("Local ancestry component requires haplotypes and local_ancestry_codes")
        if data.ancestry_order is None:
            raise ValueError("Local ancestry component requires ancestry_order")

        hap = np.asarray(data.haplotypes, dtype=np.uint8)
        anc = np.asarray(data.local_ancestry_codes, dtype=np.int16)
        if hap.ndim != 3 or hap.shape[2] != 2:
            raise ValueError("haplotypes must have shape (n_samples, n_variants, 2)")
        if anc.shape != hap.shape:
            raise ValueError("local_ancestry_codes must match haplotypes shape (n_samples, n_variants, 2)")

        variant_col = data.variant_id_column
        variants = data.membership_df[variant_col].astype(str).tolist()
        if len(variants) != hap.shape[1]:
            raise ValueError(
                "haplotypes variant dimension mismatch: "
                f"membership_variants={len(variants)} haplotypes_variants={hap.shape[1]}"
            )

        beta_pivot = data.beta_df.pivot(index="variant_id", columns="population", values="beta")
        missing_beta_variants = [v for v in variants if v not in beta_pivot.index]
        if missing_beta_variants:
            raise ValueError(f"Missing betas for variants: {missing_beta_variants[:5]}")

        ancestry_order = list(data.ancestry_order)
        missing_pops = [p for p in ancestry_order if p not in beta_pivot.columns]
        if missing_pops:
            raise ValueError(f"Missing beta columns for ancestries: {missing_pops}")

        betas = beta_pivot.loc[variants, ancestry_order].to_numpy(dtype=np.float32)  # (n_variants, n_anc)
        n_samples = int(hap.shape[0])
        n_variants = int(hap.shape[1])
        n_anc = int(betas.shape[1])

        if np.min(anc) < 0:
            raise ValueError("local_ancestry_codes contains negative (missing) values")
        if np.max(anc) >= n_anc:
            raise ValueError(
                f"local_ancestry_codes contains ancestry codes >= n_anc (max={int(np.max(anc))}, n_anc={n_anc})"
            )
        if data.use_numba and _HAS_NUMBA:
            return _la_genetic_numba(
                hap.astype(np.float32, copy=False),
                anc.astype(np.int64, copy=False),
                betas.astype(np.float32, copy=False),
            )

        chunk_n_samples = int(data.la_chunk_n_samples or 2048)
        chunk_n_variants = int(data.la_chunk_n_variants or 2048)

        genetic = np.zeros(n_samples, dtype=np.float64)
        hap_f = hap.astype(np.float32)
        anc_i = anc.astype(np.int64)

        for s0 in range(0, n_samples, chunk_n_samples):
            s1 = min(n_samples, s0 + chunk_n_samples)
            for v0 in range(0, n_variants, chunk_n_variants):
                v1 = min(n_variants, v0 + chunk_n_variants)
                betas_chunk = betas[v0:v1, :]  # (m, n_anc)
                choices = [betas_chunk[:, a] for a in range(n_anc)]  # each (m,)
                for hap_idx in (0, 1):
                    anc_chunk = anc_i[s0:s1, v0:v1, hap_idx]  # (s, m)
                    hap_chunk = hap_f[s0:s1, v0:v1, hap_idx]  # (s, m)
                    beta_sel = np.choose(anc_chunk, choices)  # (s, m)
                    genetic[s0:s1] += np.sum(hap_chunk * beta_sel, axis=1, dtype=np.float64)

        return genetic

    def name(self) -> str:
        """Return plugin name."""
        return "polygenic_no_pleiotropy"

    def compute_genetic_component(self, data: ArchitectureInput) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute additive genetic component by sample population."""
        variant_col = data.variant_id_column
        variants = data.membership_df[variant_col].astype(str).tolist()
        if not variants:
            raise ValueError("No variants selected in membership table")

        if data.haplotypes is not None and data.local_ancestry_codes is not None:
            genetic = self._compute_local_ancestry_genetic_component(data)
        else:
            beta_pivot = data.beta_df.pivot(index="variant_id", columns="population", values="beta")
            missing_beta_variants = [v for v in variants if v not in beta_pivot.index]
            if missing_beta_variants:
                raise ValueError(f"Missing betas for variants: {missing_beta_variants[:5]}")

            if data.genotype_matrix is not None:
                dosage_matrix = np.asarray(data.genotype_matrix, dtype=np.float32)
                if dosage_matrix.shape != (len(data.sample_ids), len(variants)):
                    raise ValueError(
                        "genotype_matrix shape mismatch: "
                        f"got {dosage_matrix.shape}, expected {(len(data.sample_ids), len(variants))}"
                    )
            else:
                for variant_id in variants:
                    if variant_id not in data.dosage_df.columns:
                        raise ValueError(f"Missing dosage column for selected variant: {variant_id}")
                dosage_matrix = data.dosage_df[variants].to_numpy(dtype=np.float32, copy=False)

            sample_count = len(data.sample_ids)
            genetic = np.zeros(sample_count, dtype=float)
            populations = np.asarray(data.sample_populations, dtype=object)
            for pop in np.unique(populations):
                if pop not in beta_pivot.columns:
                    raise ValueError(f"Missing beta population column: {pop}")
                row_idx = np.where(populations == pop)[0]
                if row_idx.size == 0:
                    continue
                betas = beta_pivot.loc[variants, pop].to_numpy(dtype=np.float32)
                genetic[row_idx] = dosage_matrix[row_idx, :] @ betas

        return genetic, self._build_effects_detail(data)
