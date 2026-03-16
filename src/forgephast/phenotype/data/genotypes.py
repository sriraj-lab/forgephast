"""Canonical genotype access utilities."""

from __future__ import annotations

import logging
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from ..utils.ids import align_by_ids

logger = logging.getLogger(__name__)


@runtime_checkable
class GenotypeAdapter(Protocol):
    """Protocol for genotype adapters used by phenotype simulation."""

    last_export_allele_by_variant: dict[str, str]

    def validate_files(self) -> None:
        """Validate required input artifacts."""

    def export_dosages(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Export additive dosage matrix for requested samples/variants."""

    def export_haplotypes(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Export phased haplotypes as shape (n_samples, n_variants, 2) array."""

    def compute_global_pcs(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        n_components: int,
        out_dir: Path,
        mode_cfg: object,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Compute global PCs for requested samples."""

    def export_to_pgen(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        out_prefix: Path,
        chromosomes: Sequence[int] | None = None,
        variant_ids: List[str] | None = None,
        export_cfg: object | None = None,
    ) -> str:
        """Export selected samples/variants to PGEN and return output prefix."""


def _format_chromosome_arg(chromosomes: Sequence[int] | None) -> str | None:
    """Format chromosome list for PLINK2 --chr flag."""
    if not chromosomes:
        return None
    return ",".join(str(int(chrom)) for chrom in chromosomes)


class CanonicalPgenAdapter:
    """Adapter for retrieving sample IDs and dosages from a PGEN prefix."""

    def __init__(
        self,
        pgen_prefix: str,
        plink2_path: str = "plink2",
        pgen_dosage_backend: str = "plink2_export",
    ):
        self.pgen_prefix = pgen_prefix
        self.plink2_path = plink2_path
        self.pgen_dosage_backend = pgen_dosage_backend
        self.last_export_allele_by_variant: dict[str, str] = {}
        self._dosage_cache: dict[str, tuple[pd.DataFrame, dict[str, str]]] = {}

    @property
    def psam_path(self) -> Path:
        """Path to .psam file."""
        return Path(f"{self.pgen_prefix}.psam")

    @property
    def pvar_path(self) -> Path:
        """Path to .pvar file."""
        return Path(f"{self.pgen_prefix}.pvar")

    def validate_files(self) -> None:
        """Ensure pgen triplet exists."""
        for ext in (".pgen", ".pvar", ".psam"):
            path = Path(f"{self.pgen_prefix}{ext}")
            if not path.exists():
                raise FileNotFoundError(f"Missing genotype input artifact: {path}")

    def load_sample_ids(self) -> List[str]:
        """Load IID order from .psam."""
        self.validate_files()
        df = pd.read_csv(self.psam_path, sep=r"\s+")
        id_col = "IID" if "IID" in df.columns else "#IID"
        if id_col not in df.columns:
            raise ValueError(f".psam missing IID column: {self.psam_path}")
        return df[id_col].astype(str).tolist()

    def _iid_to_fid_map(self) -> dict[str, str]:
        """Return IID->FID mapping, defaulting FID to 0 when absent in .psam."""
        self.validate_files()
        df = pd.read_csv(self.psam_path, sep=r"\s+")
        iid_col = "IID" if "IID" in df.columns else "#IID" if "#IID" in df.columns else None
        if iid_col is None:
            raise ValueError(f".psam missing IID/#IID column: {self.psam_path}")

        fid_col = "FID" if "FID" in df.columns else "#FID" if "#FID" in df.columns else None
        if fid_col is None:
            return {iid: "0" for iid in df[iid_col].astype(str).tolist()}

        return {
            iid: fid
            for iid, fid in zip(
                df[iid_col].astype(str).tolist(),
                df[fid_col].astype(str).tolist(),
            )
        }

    def load_variant_ids(self) -> List[str]:
        """Load variant IDs from .pvar."""
        self.validate_files()
        df = pd.read_csv(self.pvar_path, sep=r"\s+", comment="#")
        if "ID" not in df.columns:
            raise ValueError(f".pvar missing ID column: {self.pvar_path}")
        return df["ID"].astype(str).tolist()

    def _variant_alt_allele_map(self, variant_ids: List[str]) -> Dict[str, str]:
        """Return ID->ALT mapping for requested variants from .pvar."""
        self.validate_files()
        need = set(variant_ids)
        out: Dict[str, str] = {}

        with open(self.pvar_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    # Header/comment line.
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                variant_id = parts[2]
                if variant_id not in need:
                    continue
                alt = parts[4].split(",")[0]
                if alt == "." or alt == "":
                    raise ValueError(f"Missing ALT allele in .pvar for variant {variant_id}")
                out[variant_id] = alt
                if len(out) == len(need):
                    break

        missing = [variant_id for variant_id in variant_ids if variant_id not in out]
        if missing:
            raise ValueError(
                f"Missing variants in .pvar while building ALT allele map: {missing[:5]} (n={len(missing)})"
            )
        return out

    def export_dosages(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Export additive dosages for selected IDs using PLINK2 --export A."""
        del sample_populations
        self.validate_files()
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "sample_ids": list(sample_ids),
                    "variant_ids": list(variant_ids),
                    "chromosomes": list(chromosomes or []),
                    "backend": self.pgen_dosage_backend,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if cache_key in self._dosage_cache:
            cached_df, cached_alleles = self._dosage_cache[cache_key]
            self.last_export_allele_by_variant = dict(cached_alleles)
            return cached_df.copy()

        if self.pgen_dosage_backend == "pgenlib_direct":
            logger.info(
                "pgen_dosage_backend=pgenlib_direct is experimental; "
                "falling back to plink2 export backend for compatibility."
            )

        variant_file = out_dir / "extract_variants.txt"
        with open(variant_file, "w", encoding="utf-8") as handle:
            for variant_id in variant_ids:
                handle.write(f"{variant_id}\n")

        # Force dosage export to count ALT alleles, to match PLINK2/REGENIE
        # association effect-allele orientation.
        alt_allele_map = self._variant_alt_allele_map(variant_ids)
        export_allele_file = out_dir / "export_alleles.tsv"
        with open(export_allele_file, "w", encoding="utf-8") as handle:
            for variant_id in variant_ids:
                handle.write(f"{variant_id}\t{alt_allele_map[variant_id]}\n")

        iid_to_fid = self._iid_to_fid_map()
        keep_file = out_dir / "keep_samples.txt"
        with open(keep_file, "w", encoding="utf-8") as handle:
            handle.write("FID IID\n")
            for iid in sample_ids:
                if iid not in iid_to_fid:
                    raise ValueError(f"IID {iid} not found in .psam: {self.psam_path}")
                handle.write(f"{iid_to_fid[iid]} {iid}\n")

        out_prefix = out_dir / "dosages"
        cmd = [
            self.plink2_path,
            "--pfile",
            self.pgen_prefix,
            "--keep",
            str(keep_file),
            "--extract",
            str(variant_file),
            "--export-allele",
            str(export_allele_file),
        ]
        chr_arg = _format_chromosome_arg(chromosomes)
        if chr_arg:
            cmd.extend(["--chr", chr_arg])
        cmd.extend([
            "--export",
            "A",
            "--out",
            str(out_prefix),
        ])
        logger.info("Exporting genotype dosages for %d variants", len(variant_ids))
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "PLINK2 dosage export failed\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        raw_file = Path(f"{out_prefix}.raw")
        if not raw_file.exists():
            raise RuntimeError(f"Missing PLINK2 dosage export output: {raw_file}")

        raw_df = pd.read_csv(raw_file, sep=r"\s+")
        required_cols = {"FID", "IID"}
        if not required_cols.issubset(raw_df.columns):
            raise ValueError(f"Unexpected PLINK2 .raw columns: {list(raw_df.columns)}")

        dosage_cols = [c for c in raw_df.columns if c not in {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}]
        if not dosage_cols:
            raise ValueError("No dosage columns parsed from PLINK2 raw output")

        dosage_df = raw_df[["FID", "IID"] + dosage_cols].copy()
        for col in dosage_cols:
            dosage_df[col] = pd.to_numeric(dosage_df[col], errors="coerce").fillna(0.0)

        # PLINK2 may append allele suffixes; normalize to requested variant IDs when possible.
        rename_map = {}
        allele_by_variant: dict[str, str] = {}
        request_set = set(variant_ids)
        for col in dosage_cols:
            base = col.rsplit("_", 1)[0]
            if col not in request_set and base in request_set:
                rename_map[col] = base
                observed = col[len(base) + 1 :]
                expected = alt_allele_map[base]
                if observed and observed != expected:
                    logger.warning(
                        "Dosage column allele mismatch for %s: observed=%s expected_alt=%s",
                        base,
                        observed,
                        expected,
                    )
                allele_by_variant[base] = expected
            elif col in request_set:
                allele_by_variant[col] = alt_allele_map.get(col, "")
        if rename_map:
            dosage_df = dosage_df.rename(columns=rename_map)

        missing = [variant_id for variant_id in variant_ids if variant_id not in dosage_df.columns]
        if missing:
            raise ValueError(
                f"Missing requested variants after dosage export: {missing[:5]} (n={len(missing)})"
            )

        self.last_export_allele_by_variant = {
            variant_id: allele_by_variant.get(variant_id, alt_allele_map.get(variant_id, ""))
            for variant_id in variant_ids
        }

        # Keep exact requested order.
        dosage_df = dosage_df[["FID", "IID"] + variant_ids].copy()
        self._dosage_cache[cache_key] = (
            dosage_df.copy(),
            dict(self.last_export_allele_by_variant),
        )
        return dosage_df

    def export_haplotypes(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        variant_ids: List[str],
        out_dir: Path,
        chromosomes: Sequence[int] | None = None,
    ) -> np.ndarray:
        del sample_ids
        del sample_populations
        del variant_ids
        del out_dir
        del chromosomes
        raise NotImplementedError(
            "Phased haplotype export is not supported for CanonicalPgenAdapter. "
            "Local-ancestry simulation currently requires genotype_input.format=hapstore."
        )

    def compute_global_pcs(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        n_components: int,
        out_dir: Path,
        mode_cfg: object,
        chromosomes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Compute cohort-global PCs with PLINK2."""
        del sample_populations
        self.validate_files()
        if n_components <= 0:
            raise ValueError("n_components must be > 0")
        approx = bool(getattr(mode_cfg, "plink2_approx", True))
        extra_args = getattr(mode_cfg, "plink2_extra_args", None)

        out_dir.mkdir(parents=True, exist_ok=True)
        iid_to_fid = self._iid_to_fid_map()
        keep_file = out_dir / "keep_samples.txt"
        with open(keep_file, "w", encoding="utf-8") as handle:
            handle.write("FID IID\n")
            for iid in sample_ids:
                if iid not in iid_to_fid:
                    raise ValueError(f"IID {iid} not found in .psam: {self.psam_path}")
                handle.write(f"{iid_to_fid[iid]} {iid}\n")

        out_prefix = out_dir / "global_pcs"
        pca_args = [str(n_components)]
        if approx:
            pca_args.append("approx")

        cmd = [
            self.plink2_path,
            "--pfile",
            self.pgen_prefix,
            "--keep",
            str(keep_file),
        ]
        chr_arg = _format_chromosome_arg(chromosomes)
        if chr_arg:
            cmd.extend(["--chr", chr_arg])
        cmd.extend([
            "--pca",
            *pca_args,
            "--out",
            str(out_prefix),
        ])
        if extra_args:
            cmd.extend(list(extra_args))

        logger.info("Computing global PCs with PLINK2: %s", " ".join(cmd))
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "PLINK2 PCA failed\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        eigenvec_path = Path(f"{out_prefix}.eigenvec")
        if not eigenvec_path.exists():
            raise RuntimeError(f"Missing PLINK2 PCA output: {eigenvec_path}")

        pcs_df = _read_plink2_eigenvec(eigenvec_path, n_components=n_components)
        aligned = align_by_ids(
            pcs_df,
            id_column="IID",
            ids=sample_ids,
            context="global PCs alignment (PLINK2)",
        )
        # Standardize to the same keying convention as phenotype/covariate outputs.
        aligned["FID"] = aligned["IID"].astype(str)
        ordered_cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, n_components + 1)]
        return aligned[ordered_cols].copy()

    def export_to_pgen(
        self,
        sample_ids: List[str],
        sample_populations: List[str],
        out_prefix: Path,
        chromosomes: Sequence[int] | None = None,
        variant_ids: List[str] | None = None,
        export_cfg: object | None = None,
    ) -> str:
        """Canonical PGEN adapter uses already canonical input as export source.

        For non-HapStore workflows, GWAS can run directly on the configured prefix.
        """
        del sample_ids
        del sample_populations
        del out_prefix
        del chromosomes
        del variant_ids
        del export_cfg
        self.validate_files()
        return self.pgen_prefix


def _read_plink2_eigenvec(eigenvec_path: Path, n_components: int) -> pd.DataFrame:
    """Parse PLINK2 .eigenvec output with or without a header row."""
    first_line = ""
    with open(eigenvec_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                first_line = line
                break

    if not first_line:
        raise ValueError(f"Empty PLINK2 .eigenvec file: {eigenvec_path}")

    first_tokens = first_line.split()
    has_header = False
    if len(first_tokens) >= 2:
        header_start = {"#FID", "FID", "#IID", "IID"}
        has_header = (
            first_tokens[0] in header_start
            or first_tokens[1] in {"IID", "#IID"}
        )

    if has_header:
        pcs_df = pd.read_csv(eigenvec_path, sep=r"\s+")
    else:
        parsed_components = max(0, len(first_tokens) - 2)
        names = ["FID", "IID"] + [f"PC{i}" for i in range(1, parsed_components + 1)]
        pcs_df = pd.read_csv(eigenvec_path, sep=r"\s+", header=None, names=names)

    rename_map = {}
    if "#FID" in pcs_df.columns:
        rename_map["#FID"] = "FID"
    if "#IID" in pcs_df.columns:
        rename_map["#IID"] = "IID"
    if rename_map:
        pcs_df = pcs_df.rename(columns=rename_map)

    # Some PLINK2 builds write IID-only .eigenvec output when .psam lacks FID.
    if "FID" not in pcs_df.columns and "IID" in pcs_df.columns:
        pcs_df["FID"] = pcs_df["IID"].astype(str)

    required = {"FID", "IID"}
    if not required.issubset(pcs_df.columns):
        raise ValueError(f"Unexpected .eigenvec columns: {list(pcs_df.columns)}")

    out = pcs_df[["FID", "IID"]].copy()
    for idx in range(1, n_components + 1):
        col = f"PC{idx}"
        if col in pcs_df.columns:
            out[col] = pd.to_numeric(pcs_df[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    return out


def check_id_alignment(sample_ids: List[str], dosage_df: pd.DataFrame) -> None:
    """Check dosage table IID order against expected sample IDs."""
    got = dosage_df["IID"].astype(str).tolist()
    if len(got) != len(sample_ids):
        raise ValueError(f"ID alignment failed: expected {len(sample_ids)} rows, got {len(got)}")
    mismatches = [
        (i, exp, obs)
        for i, (exp, obs) in enumerate(zip(sample_ids, got))
        if exp != obs
    ]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"ID alignment mismatch at row {first[0]}: expected {first[1]} got {first[2]}"
        )


def compute_global_pcs_from_dosages(
    dosage_df: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame:
    """Compute top global PCs from dosage matrix using SVD."""
    if n_components <= 0:
        raise ValueError("n_components must be > 0")

    variant_cols = [c for c in dosage_df.columns if c not in {"FID", "IID"}]
    x = dosage_df[variant_cols].to_numpy(dtype=float)
    if x.shape[1] == 0:
        raise ValueError("Cannot compute PCs with zero variants")

    x = x - x.mean(axis=0, keepdims=True)
    # Avoid divide-by-zero on monomorphic columns.
    col_std = x.std(axis=0, keepdims=True)
    col_std[col_std == 0] = 1.0
    x = x / col_std

    u, s, _ = np.linalg.svd(x, full_matrices=False)
    max_components = min(n_components, u.shape[1])
    scores = u[:, :max_components] * s[:max_components]

    pcs_df = dosage_df[["FID", "IID"]].copy()
    for idx in range(max_components):
        pcs_df[f"PC{idx + 1}"] = scores[:, idx]

    for idx in range(max_components, n_components):
        pcs_df[f"PC{idx + 1}"] = 0.0

    return pcs_df


def build_genotype_adapter(
    *,
    genotype_format: str,
    pgen_prefix: str | None,
    plink2_path: str,
    pgen_dosage_backend: str = "plink2_export",
    hapstore_cfg: object | None = None,
) -> GenotypeAdapter:
    """Factory for simulation genotype adapter."""
    if genotype_format == "hapstore":
        if hapstore_cfg is None:
            raise ValueError("hapstore_cfg is required for genotype_format=hapstore")
        from .hapstore_adapter import HapStoreAdapter

        return HapStoreAdapter(hapstore_cfg=hapstore_cfg)

    if not pgen_prefix:
        raise ValueError("pgen_prefix is required for non-hapstore genotype formats")
    return CanonicalPgenAdapter(
        pgen_prefix,
        plink2_path=plink2_path,
        pgen_dosage_backend=pgen_dosage_backend,
    )
