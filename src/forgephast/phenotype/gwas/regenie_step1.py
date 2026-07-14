from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ..._internal.cache import canonical_hash
from ..._internal.command import run_checked_command
from .base import GwasTask

if TYPE_CHECKING:
    from ..config_schema import Plink2EngineConfig, RegenieEngineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step1VariantList:
    path: Path
    variant_count: int
    cache_hit: bool


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _pgen_identity(prefix: str) -> dict[str, dict[str, int | str]]:
    base = Path(prefix)
    paths = {
        "pgen": Path(f"{base}.pgen"),
        "psam": Path(f"{base}.psam"),
    }
    pvar = Path(f"{base}.pvar")
    paths["pvar"] = pvar if pvar.exists() else Path(f"{base}.pvar.zst")
    missing = [
        str(path)
        for path in paths.values()
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f"Missing or empty PGEN input artifact(s): {missing}")
    return {name: _file_identity(path) for name, path in paths.items()}


def _count_variant_ids(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _metadata_path(variant_list: Path) -> Path:
    return variant_list.with_suffix(variant_list.suffix + ".forgephast.json")


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def prepare_regenie_step1_variant_list(
    *,
    task: GwasTask,
    out_dir: Path,
    regenie_cfg: "RegenieEngineConfig",
    plink2_cfg: "Plink2EngineConfig | None",
) -> Step1VariantList | None:
    if str(regenie_cfg.step1_variant_mode) == "all":
        return None

    keep_path = Path(task.keep_file)
    if not keep_path.exists() or keep_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty REGENIE keep file: {keep_path}")

    plink2_path = str(getattr(plink2_cfg, "path", "plink2"))
    plink2_threads = int(getattr(plink2_cfg, "threads", regenie_cfg.threads))
    plink2_memory_mb = getattr(plink2_cfg, "memory_mb", None)
    cache_key = canonical_hash(
        {
            "schema_version": 2,
            "pgen": _pgen_identity(task.pgen_prefix),
            "keep_file": {
                **_file_identity(keep_path),
                "sha256": _file_hash(keep_path),
            },
            "chromosomes": [int(chrom) for chrom in (task.chromosomes or [])],
            "plink2_path": plink2_path,
            "min_maf": float(regenie_cfg.step1_min_maf),
            "min_mac": int(regenie_cfg.step1_min_mac),
            "max_missing": float(regenie_cfg.step1_max_missing),
            "hwe_p": float(regenie_cfg.step1_hwe_p),
            "ld_window_variants": int(regenie_cfg.step1_ld_window_variants),
            "ld_step_variants": int(regenie_cfg.step1_ld_step_variants),
            "ld_r2": float(regenie_cfg.step1_ld_r2),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    prune_prefix = out_dir / "step1_variants"
    missing_id_exclude = out_dir / "step1_variants.missing_ids.exclude"
    missing_id_exclude.write_text(".\n", encoding="utf-8")
    variant_list = Path(f"{prune_prefix}.prune.in")
    metadata_path = _metadata_path(variant_list)
    if (
        bool(regenie_cfg.step1_reuse_variant_list)
        and variant_list.exists()
        and metadata_path.exists()
    ):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            metadata = {}
        variant_count = _count_variant_ids(variant_list)
        if metadata.get("cache_key") == cache_key and variant_count > 0:
            logger.info(
                "Reusing REGENIE step-1 variant list path=%s variants=%d",
                variant_list,
                variant_count,
            )
            return Step1VariantList(variant_list, variant_count, True)

    cmd = [
        plink2_path,
        "--pfile",
        task.pgen_prefix,
        "--keep",
        task.keep_file,
        "--exclude",
        str(missing_id_exclude),
        "--rm-dup",
        "exclude-all",
    ]
    if task.chromosomes:
        cmd.extend(["--chr", ",".join(str(int(chrom)) for chrom in task.chromosomes)])
    cmd.extend(
        [
            "--maf",
            str(float(regenie_cfg.step1_min_maf)),
            "--mac",
            str(int(regenie_cfg.step1_min_mac)),
            "--geno",
            str(float(regenie_cfg.step1_max_missing)),
            "--hwe",
            str(float(regenie_cfg.step1_hwe_p)),
            "--indep-pairwise",
            str(int(regenie_cfg.step1_ld_window_variants)),
            str(int(regenie_cfg.step1_ld_step_variants)),
            str(float(regenie_cfg.step1_ld_r2)),
            "--threads",
            str(plink2_threads),
        ]
    )
    if plink2_memory_mb is not None:
        cmd.extend(["--memory", str(int(plink2_memory_mb))])
    cmd.extend(["--out", str(prune_prefix)])
    run_checked_command(cmd, label="PLINK2 REGENIE step-1 variant selection", logger=logger)

    if not variant_list.exists():
        raise RuntimeError(f"PLINK2 did not produce REGENIE step-1 variant list: {variant_list}")
    variant_count = _count_variant_ids(variant_list)
    if variant_count == 0:
        raise RuntimeError(
            "REGENIE step-1 common/QC/LD-pruned variant list is empty; "
            "relax gwas.regenie step1 filters for this cohort"
        )
    _write_metadata(
        metadata_path,
        {
            "schema_version": 2,
            "cache_key": cache_key,
            "variant_count": variant_count,
            "variant_list": str(variant_list),
        },
    )
    return Step1VariantList(variant_list, variant_count, False)


__all__ = ["Step1VariantList", "prepare_regenie_step1_variant_list"]
