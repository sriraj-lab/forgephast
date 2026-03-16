from __future__ import annotations

from pathlib import Path

import pysam

from forgephast.amplifier.vcfio import load_donor_haplotypes


def _write_chr_prefixed_vcf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_vcf = path.with_suffix("")

    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.contigs.add("chr21")
    header.formats.add("GT", number=1, type="String", description="Phased genotype")
    for sid in ("S1", "S2"):
        header.add_sample(sid)

    with pysam.VariantFile(str(temp_vcf), "w", header=header) as out:
        for pos, gt1, gt2 in [(100, (0, 0), (0, 1)), (200, (0, 1), (1, 1))]:
            rec = out.new_record(contig="chr21", start=pos - 1, stop=pos, id=f"v{pos}", alleles=("A", "C"))
            rec.samples["S1"]["GT"] = gt1
            rec.samples["S2"]["GT"] = gt2
            rec.samples["S1"].phased = True
            rec.samples["S2"].phased = True
            out.write(rec)

    pysam.tabix_compress(str(temp_vcf), str(path), force=True)
    pysam.tabix_index(str(path), preset="vcf", force=True)
    temp_vcf.unlink(missing_ok=True)


def test_load_donor_haplotypes_handles_chr_prefix_alias(tmp_path: Path) -> None:
    vcf_path = tmp_path / "panel.chr21.vcf.gz"
    _write_chr_prefixed_vcf(vcf_path)

    panel = load_donor_haplotypes(vcf_path, chrom=21, donor_ids=["S1", "S2"])
    assert panel.variants.chrom == "chr21"
    assert panel.donor_haps.shape == (4, 2)

