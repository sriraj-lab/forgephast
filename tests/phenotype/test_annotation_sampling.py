"""Annotation-aware causal sampling tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forgephast.phenotype.causal.annotations import (
    annotate_sampling_weights,
    load_annotation_intervals_for_config,
    resolve_variant_coordinates,
)
from forgephast.phenotype.causal.hybrid_sampler import sample_membership_for_replicate
from forgephast.phenotype.causal.selection import build_bucketed_pool
from forgephast.phenotype.config import (
    AnnotationSamplingConfig,
    CausalSelectionConfig,
    CustomAnnotationConfig,
    DefaultGeneWindowConfig,
    MafBucketConfig,
)


def _write_bed(path: Path, rows: list[tuple[str, int, int]]) -> None:
    path.write_text(
        "".join(f"{chrom}\t{start}\t{end}\n" for chrom, start, end in rows),
        encoding="utf-8",
    )


def _write_pool(path: Path, *, n_variants: int = 100) -> None:
    rows = []
    for idx in range(n_variants):
        pos = idx + 1
        rows.append(
            {
                "causal_id": f"chr1:{pos}",
                "chrom": 1,
                "pos": pos,
                "maf_eur": 0.02,
                "maf_afr": 0.025,
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_resolve_variant_coordinates_prefers_explicit_columns(tmp_path: Path) -> None:
    pool = pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20"],
            "chromosome": ["chr1", "1"],
            "position": [101, 202],
            "maf_eur": [0.02, 0.02],
        }
    )
    cfg = CausalSelectionConfig(
        source_pairs_file=str(tmp_path / "unused.tsv"),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur"},
        anchor_population="EUR",
        maf_buckets=[MafBucketConfig("b1", 0.01, 0.05, fixed_count=1, subset_count=1)],
        annotation_sampling=AnnotationSamplingConfig(
            enabled=True,
            variant_chrom_column="chromosome",
            variant_pos_column="position",
            default_gene_window=DefaultGeneWindowConfig(enabled=True),
        ),
    )

    resolved = resolve_variant_coordinates(pool, cfg)
    assert resolved["chrom"].tolist() == [1, 1]
    assert resolved["pos"].tolist() == [101, 202]
    assert resolved["pos0"].tolist() == [100, 201]


def test_resolve_variant_coordinates_falls_back_to_variant_id(tmp_path: Path) -> None:
    pool = pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr2:20"],
            "maf_eur": [0.02, 0.03],
        }
    )
    cfg = CausalSelectionConfig(
        source_pairs_file=str(tmp_path / "unused.tsv"),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur"},
        anchor_population="EUR",
        maf_buckets=[MafBucketConfig("b1", 0.01, 0.05, fixed_count=1, subset_count=1)],
        annotation_sampling=AnnotationSamplingConfig(
            enabled=True,
            default_gene_window=DefaultGeneWindowConfig(enabled=True),
        ),
    )

    resolved = resolve_variant_coordinates(pool, cfg)
    assert resolved["chrom"].tolist() == [1, 2]
    assert resolved["pos"].tolist() == [10, 20]
    assert resolved["pos0"].tolist() == [9, 19]


def test_default_gene_window_derivation_respects_strand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    genes_path = tmp_path / "genes.tsv"
    pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "gene_start": [1000, 5000],
            "gene_end": [2000, 6000],
            "strand": ["+", "-"],
            "tss": [1000, 6000],
            "gene_name": ["PLUS", "MINUS"],
        }
    ).to_csv(genes_path, sep="\t", index=False)

    monkeypatch.setattr(
        "forgephast.phenotype.causal.annotations.bundled_default_gene_annotation_path",
        lambda genome_build: genes_path,
    )

    cfg = AnnotationSamplingConfig(
        enabled=True,
        default_gene_window=DefaultGeneWindowConfig(
            enabled=True,
            genome_build="GRCh38",
            upstream_bp=100,
            weight_multiplier=5.0,
        ),
    )

    interval_sets = list(load_annotation_intervals_for_config(cfg))
    assert len(interval_sets) == 1
    starts, ends = interval_sets[0].intervals_by_chrom[1]
    assert starts.tolist() == [900, 4999]
    assert ends.tolist() == [2000, 6100]


def test_bed_overlap_uses_half_open_intervals_and_max_multiplier(tmp_path: Path) -> None:
    bed_a = tmp_path / "a.bed"
    bed_b = tmp_path / "b.bed"
    _write_bed(bed_a, [("chr1", 9, 20)])
    _write_bed(bed_b, [("chr1", 19, 30)])

    pool = pd.DataFrame(
        {
            "causal_id": ["chr1:10", "chr1:20", "chr1:21", "chr1:30", "chr1:31"],
            "maf_eur": [0.02] * 5,
        }
    )
    cfg = CausalSelectionConfig(
        source_pairs_file=str(tmp_path / "unused.tsv"),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur"},
        anchor_population="EUR",
        maf_buckets=[MafBucketConfig("b1", 0.01, 0.05, fixed_count=1, subset_count=1)],
        annotation_sampling=AnnotationSamplingConfig(
            enabled=True,
            default_gene_window=DefaultGeneWindowConfig(enabled=False),
            custom_annotations=[
                CustomAnnotationConfig(name="a", bed_file=str(bed_a), weight_multiplier=2.0),
                CustomAnnotationConfig(name="b", bed_file=str(bed_b), weight_multiplier=5.0),
            ],
        ),
    )

    annotated = annotate_sampling_weights(pool, cfg)
    assert annotated["annotation_hit"].tolist() == [True, True, True, True, False]
    assert annotated["sampling_weight"].tolist() == [2.0, 5.0, 5.0, 5.0, 1.0]
    assert annotated["annotation_labels"].tolist() == ["a", "a,b", "b", "b", ""]


def test_uniform_behavior_is_preserved_when_annotation_sampling_disabled(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    _write_pool(pool_path, n_variants=8)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur"},
        anchor_population="EUR",
        maf_buckets=[MafBucketConfig("b1", 0.01, 0.05, fixed_count=2, subset_count=2)],
        random_seed=7,
    )

    bucketed = build_bucketed_pool(cfg)
    assert set(bucketed.full_pool["sampling_weight"].tolist()) == {1.0}
    assert not bucketed.full_pool["annotation_hit"].any()
    assert set(bucketed.full_pool["annotation_labels"].tolist()) == {""}


def test_weighted_sampling_enriches_annotation_hits_and_preserves_bucket_counts(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    bed_path = tmp_path / "genes.bed"
    _write_pool(pool_path, n_variants=100)
    _write_bed(bed_path, [("chr1", 0, 20)])

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur", "AFR": "maf_afr"},
        anchor_population="EUR",
        maf_buckets=[MafBucketConfig("b1", 0.01, 0.05, fixed_count=10, subset_count=10)],
        annotation_sampling=AnnotationSamplingConfig(
            enabled=True,
            default_gene_window=DefaultGeneWindowConfig(enabled=False),
            custom_annotations=[
                CustomAnnotationConfig(
                    name="genes",
                    bed_file=str(bed_path),
                    weight_multiplier=100.0,
                )
            ],
        ),
        random_seed=123,
    )

    bucketed = build_bucketed_pool(cfg)
    membership = sample_membership_for_replicate(1, bucketed.full_pool, bucketed.fixed_set, cfg)

    counts = membership.groupby(["maf_bucket", "membership_type"]).size().to_dict()
    assert counts[("b1", "fixed")] == 10
    assert counts[("b1", "subset")] == 10
    assert {"sampling_weight", "annotation_hit", "annotation_labels"} <= set(membership.columns)
    assert float(membership["annotation_hit"].mean()) > 0.6
    assert membership["annotation_labels"].astype(str).str.contains("genes").any()


def test_build_bucketed_pool_supports_default_gene_window_config(tmp_path: Path) -> None:
    pool_path = tmp_path / "pairs.tsv"
    _write_pool(pool_path, n_variants=5)

    cfg = CausalSelectionConfig(
        source_pairs_file=str(pool_path),
        variant_id_column="causal_id",
        maf_columns_by_population={"EUR": "maf_eur", "AFR": "maf_afr"},
        anchor_population="EUR",
        maf_buckets=[MafBucketConfig("b1", 0.01, 0.05, fixed_count=1, subset_count=1)],
        annotation_sampling=AnnotationSamplingConfig(
            enabled=True,
            default_gene_window=DefaultGeneWindowConfig(enabled=True, weight_multiplier=5.0),
        ),
        random_seed=5,
    )

    bucketed = build_bucketed_pool(cfg)
    membership = sample_membership_for_replicate(1, bucketed.full_pool, bucketed.fixed_set, cfg)

    assert "sampling_weight" in bucketed.full_pool.columns
    assert "annotation_hit" in membership.columns
    assert membership.shape[0] == 2
