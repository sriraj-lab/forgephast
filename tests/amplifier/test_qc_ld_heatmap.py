from __future__ import annotations

import numpy as np
import yaml

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.merge import merge_chromosome
from forgephast.amplifier.qc import run_qc
from forgephast.amplifier.qc.ld_heatmap import ld_heatmap_binned_from_genotypes
from forgephast.amplifier.simulation import simulate_chromosome

from .conftest import write_test_config


def test_ld_heatmap_binned_is_deterministic():
    rng = np.random.default_rng(0)
    genotypes = rng.integers(0, 3, size=(12, 40)).astype(np.float64)
    positions_bp = (100 + 10 * np.arange(genotypes.shape[1])).astype(np.int64)
    map_bp = positions_bp.astype(np.float64)
    map_cm = (positions_bp.astype(np.float64) / 1000.0).astype(np.float64)

    out1, edges1, meta1 = ld_heatmap_binned_from_genotypes(
        genotypes=genotypes,
        variant_positions_bp=positions_bp,
        map_bp=map_bp,
        map_cm=map_cm,
        n_position_bins=20,
        max_variants=25,
        min_maf=0.01,
        max_distance_cm=0.05,
        max_pairs_total=2000,
        max_neighbors_per_anchor=32,
        seed=123,
    )
    out2, edges2, meta2 = ld_heatmap_binned_from_genotypes(
        genotypes=genotypes,
        variant_positions_bp=positions_bp,
        map_bp=map_bp,
        map_cm=map_cm,
        n_position_bins=20,
        max_variants=25,
        min_maf=0.01,
        max_distance_cm=0.05,
        max_pairs_total=2000,
        max_neighbors_per_anchor=32,
        seed=123,
    )

    assert meta1["status"] == "ok"
    assert meta2["status"] == "ok"
    assert meta1["used_pairs"] == meta2["used_pairs"]
    np.testing.assert_allclose(edges1, edges2)
    np.testing.assert_allclose(out1, out2, equal_nan=True)


def test_ld_heatmap_plot_written_in_qc(tmp_path):
    try:
        import matplotlib  # noqa: F401

        has_matplotlib = True
    except Exception:
        has_matplotlib = False

    cfg_path = write_test_config(tmp_path, name="qc_ld_heatmap")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["outputs"]["write_combined_shard_output"] = False
    payload["outputs"]["merge_combined_populations"] = False
    payload["qc"]["backend"] = "python"
    payload["qc"]["ld_decay_enabled"] = True
    payload["qc"]["ext_max_diploids_per_population"] = 4
    payload["qc"]["ext_max_variants"] = 8
    payload["qc"]["ld_decay_min_pairs_per_bin"] = 1
    payload["qc"]["ld_decay_max_distance_cm"] = 0.5
    payload["qc"]["ld_decay_bin_width_cm"] = 0.1
    payload["qc"]["ld_decay_rmse_max"] = 1.0
    payload["qc"]["ld_decay_half_decay_rel_diff_max"] = 1.0

    payload["qc"]["plots_enabled"] = True
    payload["qc"]["ld_heatmap_n_position_bins"] = 20

    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    cfg = from_yaml(cfg_path)
    build_manifest_from_config(cfg)
    simulate_chromosome(cfg, chrom=21, shard_id=0, n_shards=2)
    simulate_chromosome(cfg, chrom=21, shard_id=1, n_shards=2)
    merge_chromosome(cfg, chrom=21, n_shards=2)
    report = run_qc(cfg, chrom=21, n_shards=2)

    assert report.ld_decay is not None
    assert report.ld_decay.get("heatmap", {}).get("enabled") is True
    plots_dir = report.diagnostics_path.parent / "plots"

    if has_matplotlib:
        pngs = sorted(plots_dir.glob("ld_heatmap_*_chr21.png"))
        assert pngs, f"Expected LD heatmap PNGs under {plots_dir}"
    else:
        pop_summary = report.ld_decay.get("population_summary", {})
        heatmaps = [
            (pop, info.get("heatmap", {}))
            for pop, info in pop_summary.items()
            if isinstance(info, dict) and isinstance(info.get("heatmap"), dict)
        ]
        assert heatmaps
        assert any(hm.get("status") == "error" for _, hm in heatmaps)
