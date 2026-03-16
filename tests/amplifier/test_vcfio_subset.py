from __future__ import annotations

from pathlib import Path

import numpy as np

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.vcfio import load_donor_haplotypes, subset_panel_donors

from .conftest import write_test_config


def test_subset_panel_donors_matches_direct_load(tmp_path):
    cfg_path = write_test_config(tmp_path, name="subset_panel")
    cfg = from_yaml(cfg_path)
    manifest = build_manifest_from_config(cfg)

    panel_vcf = Path(cfg.input.panel_vcf_pattern.format(chrom=21, population="P1"))
    all_donors = sorted({sid for donors in manifest.donors_by_population.values() for sid in donors})

    panel_all = load_donor_haplotypes(vcf_path=panel_vcf, chrom=21, donor_ids=all_donors)
    p1_donors = manifest.donors_by_population["P1"]
    panel_p1_direct = load_donor_haplotypes(vcf_path=panel_vcf, chrom=21, donor_ids=p1_donors)
    panel_p1_subset = subset_panel_donors(panel_all, p1_donors)

    assert np.array_equal(panel_p1_direct.variants.pos, panel_p1_subset.variants.pos)
    assert np.array_equal(panel_p1_direct.donor_haps, panel_p1_subset.donor_haps)
    assert panel_p1_subset.donor_ids == p1_donors
