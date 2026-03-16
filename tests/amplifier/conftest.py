from __future__ import annotations

from pathlib import Path

import pysam
import yaml



def _write_tiny_panel_vcf(path: Path, chrom: str = "21") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_vcf = path.with_suffix("")
    samples = ["A1", "A2", "B1", "B2", "C1", "C2"]

    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.contigs.add(chrom)
    header.formats.add("GT", number=1, type="String", description="Genotype")
    for sid in samples:
        header.add_sample(sid)

    positions = [100, 200, 300, 400, 500, 600, 700, 800]
    pop_a = [(0, 0), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 0), (0, 1)]
    pop_b = [(1, 1), (1, 0), (1, 1), (1, 1), (1, 0), (1, 1), (1, 1), (1, 0)]
    pop_c = [(0, 1), (0, 1), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)]

    with pysam.VariantFile(str(temp_vcf), "w", header=header) as out:
        for i, pos in enumerate(positions):
            rec = out.new_record(
                contig=chrom,
                start=pos - 1,
                stop=pos,
                id=f"v{i+1}",
                alleles=("A", "G"),
            )
            for sid in ["A1", "A2"]:
                rec.samples[sid]["GT"] = pop_a[i]
                rec.samples[sid].phased = True
            for sid in ["B1", "B2"]:
                rec.samples[sid]["GT"] = pop_b[i]
                rec.samples[sid].phased = True
            for sid in ["C1", "C2"]:
                rec.samples[sid]["GT"] = pop_c[i]
                rec.samples[sid].phased = True
            out.write(rec)

    pysam.tabix_compress(str(temp_vcf), str(path), force=True)
    pysam.tabix_index(str(path), preset="vcf", force=True)
    temp_vcf.unlink(missing_ok=True)



def _write_tiny_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for i, pos in enumerate([100, 200, 300, 400, 500, 600, 700, 800], start=1):
            handle.write(f"21\t.\t{i * 0.1:.4f}\t{pos}\n")



def _write_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("IID\tSUPERPOP\n")
        handle.write("A1\tPOP_A\n")
        handle.write("A2\tPOP_A\n")
        handle.write("B1\tPOP_B\n")
        handle.write("B2\tPOP_B\n")
        handle.write("C1\tPOP_C\n")
        handle.write("C2\tPOP_C\n")



def write_test_config(tmp_path: Path, name: str = "test_run", n_shards_workers: int = 2) -> Path:
    panel = tmp_path / "panel" / "chr21.vcf.gz"
    gmap = tmp_path / "maps" / "chr21.map"
    meta = tmp_path / "meta" / "metadata.tsv"

    _write_tiny_panel_vcf(panel)
    _write_tiny_map(gmap)
    _write_metadata(meta)

    config = {
        "name": name,
        "input": {
            "metadata_path": str(meta),
            "metadata_sep": "\t",
            "sample_id_column": "IID",
            "population_column": "SUPERPOP",
            "panel_vcf_pattern": str(tmp_path / "panel" / "chr{chrom}.vcf.gz"),
            "genetic_map_pattern": str(tmp_path / "maps" / "chr{chrom}.map"),
        },
        "populations": [
            {"name": "P1", "selector": {"values": ["POP_A"]}, "n_diploids": 4},
            {"name": "P2", "selector": {"values": ["POP_B"]}, "n_diploids": 4},
            {"name": "P3", "selector": {"values": ["POP_C"]}, "n_diploids": 4},
        ],
        "simulation": {
            "chromosomes": [21],
            "random_seed": 123,
            "mean_tract_cm": 1.0,
            "threads_per_task": n_shards_workers,
            "donor_io_workers": 1,
        },
        "outputs": {
            "root": str(tmp_path / "outputs"),
            "write_phased_vcf": True,
            "write_pgen": False,
            "write_combined_shard_output": True,
            "plink2_path": "plink2",
            "merge_backend": "python",
        },
        "qc": {
            "within_split_abs_max": 0.5,
            "fst_tolerance_relative": 1.0,
        },
    }

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return cfg_path
