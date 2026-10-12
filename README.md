# forgephast

FORGEPHAST combines population amplification and phenotype simulation workflows around a HapStore-first runtime.

## Amplifier donor input contract

### Donor selector weights

`populations[].selector_weights` sets the desired aggregate sampling mass for metadata groups inside a mosaic donor pool. These are group-level mixture weights, not per-individual weights.

If group `B` represents fraction `f_B` of the selected donor individuals but is assigned selector weight `w_B`, then:

- `w_B > f_B` upsamples each `B` donor relative to unweighted donor sampling.
- `w_B < f_B` downsamples each `B` donor relative to unweighted donor sampling.
- `w_B = f_B` is equivalent to unweighted sampling for that group.

For example, if a donor pool contains `80` group-`A` individuals and `20` group-`B` individuals, unweighted sampling gives `f_A=0.8` and `f_B=0.2`. Setting `selector_weights: {A: 0.6, B: 0.4}` makes each `B` individual more likely to be copied than each `A` individual because the smaller `B` pool receives `40%` of the total donor sampling mass.

Internally, each donor in group `g` receives weight `w_g / n_g`, and that diploid donor weight is split evenly across its two haplotypes for tract-start and switch-point copying. Use `selector_weights` only when that intentional up/down-sampling is part of the simulation design.

## Amplifier simulation contract

Amplifier now has one simulation path: the shard-pool engine.

- `forgephast amplify simulate` is the canonical simulation command.
- `n_shards=1` preserves the old single-shard behavior, but still runs through the same shard-pool implementation and returns the same output artifacts.
- The separate public `simulate-pool` command/API no longer exists.
- Runtime defaults for shard count, worker count, threads-per-task, and task start method live under `simulation.n_shards`, `simulation.task_workers`, `simulation.threads_per_task`, and `simulation.task_start_method`.

### Numba acceleration and multiprocessing

Install `forgephast[accel]` for Numba acceleration and its TBB threading backend.
Numba normally prefers TBB when available. Before starting a `fork` worker pool,
FORGEPHAST checks the actual selected backend and rejects Linux OpenMP, which is
not fork-safe and can otherwise leave the pool hanging. If OpenMP is already
initialized, restart Python with `NUMBA_THREADING_LAYER=tbb` after installing
the extra, or use `simulation.task_start_method: spawn`. Setting
`NUMBA_THREADING_LAYER=workqueue` before starting Python is also supported.
TBB-backed `fork` pools must be launched from the main thread. The check does
not change the configured backend or start method automatically.

### Publication TODO: `msprime_tracts` shard parallelism

Current limitation:

- if any simulated population uses `engine: msprime_tracts`, amplifier currently switches to a shard-sequential fallback path
- in that branch, `simulation.task_workers` does not provide shard-level parallelism; shards run one at a time
- do not interpret current `msprime_tracts` runtime as evidence of working shard-pool scale-out

This is acceptable for ongoing internal repair work, but it needs to be fixed or explicitly redesigned before publication.

Amplifier now reads donor panels at runtime from HapStore only.

- If `input.panel_hapstore_pattern` resolves to an existing HapStore, amplifier uses it directly.
- If the resolved donor source is a VCF from `input.panel_vcf_pattern`, amplifier first ingests the requested donor subset into a donor HapStore cache and then reads donors from that HapStore.
- Amplifier no longer has a separate direct donor-VCF runtime path; donor reads always go through HapStore.
- The stage-0 cache command `forgephast amplify build-cache --config <config.yaml>` prewarms those donor HapStore caches, but normal simulation and QC runs will also materialize them on demand.

By default, generated donor HapStore caches live under:

- `<outputs.root>/<name>/cache/donor_panels/`

You can override that location with `input.donor_cache_dir`.

## Amplifier QC contract

Amplifier QC now runs only against HapStore-backed synthetic outputs.

- `forgephast amplify qc` reads merged combined HapStores or merged per-population HapStores.
- VCF outputs are optional artifacts for export and interoperability, but QC no longer uses a VCF-only execution path.
- `outputs.write_hapstore` is now required because amplifier runtime is HapStore-native.
- Shard-vs-shard synthetic round-robin is now a normal post-merge QC metric, not a separate pre-merge command.

In practice, if you want QC to be available, keep:

```yaml
outputs:
  write_hapstore: true
```

### Synthetic shard round-robin metric

Synthetic shard-vs-shard FST now runs only inside the main `forgephast amplify qc` flow and only from merged HapStore outputs plus the merged manifest `shard_id` column.

There is no separate `forgephast amplify qc-synthetic-premerge-round-robin` command anymore.

Optional config knobs:

```yaml
qc:
  synthetic_shard_round_robin_enabled: false
  synthetic_shard_round_robin_pairwise_mode: exact
  synthetic_shard_round_robin_max_pairs_per_population: null
  synthetic_shard_round_robin_min_cpus_per_match: 1
```

Removed pre-merge keys now fail config load:

- `qc.rr_pairwise_mode`
- `qc.synthetic_round_robin_max_pairs_per_population`
- `qc.synthetic_rr_min_cpus_per_match`

## Amplifier export contract

Amplifier export is now single-path:

- PGEN export is HapStore-direct only.
- Phased VCF export is produced from HapStore through the HapStore export path, which routes through PGEN/PLINK2 when `outputs.hapstore_vcf_plink2_location` is configured.
- Shard and merged VCF sample-axis merge paths were removed from amplifier.

Removed output keys now fail config load:

- `outputs.pgen_source_preference`
- `outputs.merge_backend`
- `outputs.bcftools_path`
- `outputs.bcftools_threads`
- `outputs.bcftools_no_index`
- `outputs.plink1_path`

## Amplifier config notes

The donor-panel config fields have different roles:

- `input.panel_hapstore_pattern`: canonical runtime donor source.
- `input.panel_vcf_pattern`: ingest source used only when a donor HapStore is not already available.
- `input.require_chrom_sharded_panel`: still applies to either pattern when configured.

Minimal example:

```yaml
name: demo
input:
  metadata_path: metadata.tsv
  sample_id_column: IID
  population_column: SUPERPOP
  genetic_map_pattern: maps/chr{chrom}.map
  panel_vcf_pattern: panels/chr{chrom}.vcf.gz
  # Optional canonical runtime source:
  # panel_hapstore_pattern: panels/chr{chrom}.hapstore.zarr
simulation:
  chromosomes: [21, 22]
outputs:
  root: runs
```

With that config, amplifier will build donor HapStore caches under `runs/demo/cache/donor_panels/` as needed and reuse them across simulation and QC.

## Phenotype effect geometry constraints

`forgephast phenotype` samples population-specific effect sizes with a shared-base-plus-jitter construction:

- `beta_pop = beta_base + delta_pop`
- `Var(beta_base) = sigma_beta^2`
- `Var(delta_pop) = jitter_sd_by_population[pop]^2`

The configured `effect_model.population_corr_matrix` is interpreted as the target correlation of the final per-population betas, not the correlation of the jitter terms alone.

For population pair `i, j`, the code builds:

- `total_sd_i = sqrt(sigma_beta^2 + jitter_i^2)`
- `Cov(beta_i, beta_j) = Corr(beta_i, beta_j) * total_sd_i * total_sd_j`
- `Cov(delta_i, delta_j) = Cov(beta_i, beta_j) - sigma_beta^2`

This derived jitter covariance must be mathematically valid before simulation can run.

### Required validity conditions

1. Pairwise bound:

- `|Cov(delta_i, delta_j)| <= jitter_i * jitter_j`

If this fails, the requested cross-pop correlation is incompatible with the amount of shared base variance and jitter variance.

2. Positive semidefiniteness:

- the full derived jitter covariance matrix must be positive semidefinite

Even if every pairwise bound passes, the full matrix can still be invalid.

### Practical interpretation

- Lower cross-pop correlations require less shared-base dominance or more jitter variance.
- Higher `sigma_beta` makes it harder to realize low target cross-pop correlations.
- Increasing `jitter_sd_by_population` gives the model more room to express lower correlations.
- If you see an error about `population_corr_matrix` incompatibility or a non-PSD derived jitter covariance, adjust one or more of:
  - `architectures[].params.sigma_beta`
  - `effect_model.jitter_sd_by_population`
  - `effect_model.population_corr_matrix`

### Important note about heritability

These geometry parameters control whether the cross-pop beta construction is valid. Final fixed-effect heritability is anchored by one pooled fixed-effect scale factor while preserving per-population realized variance differences.

`forgephast phenotype` uses `variance_budget.h2_fixed` or `architectures[].params.h2` as the pooled fixed-effect target. Per-population realized fixed-effect variance is reported as an output, not forced to match separate per-population h2 targets.

So changing `sigma_beta`, effect-size jitter settings, or the FST-derived correlation affects the geometry of sampled effects, while `variance_budget.h2_fixed` controls the pooled fixed-effect target.

### Auto population correlation from donor FST

`forgephast phenotype` can now derive the population-correlation matrix automatically from a precomputed donor-panel pairwise-FST table.

Use:

- `effect_model.population_corr_mode: auto_from_fst`
- `effect_model.population_fst_file: <pairwise_fst.tsv>`

Required FST table columns:

- `population_i`
- `population_j`
- `fst`

Behavior:

- all unordered pairs in `effect_model.population_order` must be present
- diagonal FST is implied as `0.0`
- the raw correlation transform is:
  - `corr_ij = exp(-fst_corr_lambda * fst_ij)`
- the resulting matrix is then projected to a valid PSD correlation matrix before beta sampling

This auto mode is explicit:

- when `population_corr_mode: manual`, the existing `population_corr_matrix` behavior is unchanged
- when `population_corr_mode: auto_from_fst`, `population_corr_matrix` must be omitted

`effect_model.base_jitter_sd` provides the default per-population jitter floor for populations not explicitly listed in `effect_model.jitter_sd_by_population`.

### `h2_fixed` target reporting

`variance_budget.h2_jitter_sd` is disabled and must remain `0.0`. Use `effect_model.base_jitter_sd` and `effect_model.jitter_sd_by_population` for cross-population effect-size heterogeneity.

Behavior:

- `h2_fixed` is interpreted as the pooled fixed-effect target.
- Per-population fixed-effect h2 is a realized output reflecting causal-variant MAF structure and the configured beta geometry.
- `resolved_h2_fixed_by_population` remains informational and mirrors the non-jittered target for configured source populations.
- `h2_jitter_delta_by_population` and `h2_jitter_clipped_by_population` remain in outputs for schema compatibility, but are zero/false under the supported config contract.

Target mode:

- single-population cohorts (or cohorts with exactly one non-`ADMIX` source population) resolve a concrete cohort-level target and report `target_mode: single_population`
- multi-population cohorts (e.g. `EUR + AFR`, or `EUR + AFR + ADMIX`) report `target_mode: free_per_population` and set cohort-level `genetic_fixed` / `resolved_h2_fixed` / `residual` to `NaN`; each population's realized fixed-effect variance appears in the new `realized_h2_fixed_by_population` JSON column instead
- the legacy `inherited_source_betas` mode has been removed; configs that relied on it will now run as `free_per_population` and report biologically free per-population h²

Auditable outputs now include:

- `target_mode` in `truth/component_summary.tsv` and `run_manifest.tsv` (values: `single_population` or `free_per_population`)
- `resolved_h2_fixed_by_population` — per-population TARGET (informational only, no longer drives β rescaling)
- `realized_h2_fixed_by_population` — per-population ACTUAL variance of `GENETIC_FIXED_COMPONENT` (new)
- `h2_jitter_delta_by_population`
- `h2_jitter_clipped_by_population`
- `shift_tightness_*` columns (see "Cross-population MAF tightness" below)

### Binary phenotypes from continuous liability

`forgephast phenotype` can optionally convert the final continuous liability into a binary 0/1 case-control phenotype:

```yaml
phenotype:
  trait_type: binary
  n_replicates: 1
  binary:
    prevalence: 0.1
    liability_logit_scale: 1.0
    random_seed: 42
    prevalence_by_population:
      EUR: 0.10
      AFR: 0.12
```

The transform standardizes the final liability within the target cohort, solves an intercept `alpha` so the mean predicted probability matches the configured prevalence, then draws labels from:

- `Pr(Y=1) = sigmoid(alpha + liability_logit_scale * z_liability)`

When `prevalence_by_population` is provided, `liability_logit_scale` is shared and one intercept is solved per non-admixed source population, falling back to `binary.prevalence` for source populations not listed. As an alternative to prevalence targets, advanced configs may provide fixed source-population intercepts directly:

```yaml
phenotype:
  binary:
    alpha_by_population:
      EUR: -2.0
      AFR: -0.5
```

Do not provide `prevalence_by_population` and `alpha_by_population` together; they are alternative source-alpha inputs.

For local-ancestry admixed cohorts, `ADMIX` samples are not assigned an admixed target prevalence by fallback. If a single cohort-level prevalence is used, the shared cohort `alpha` is applied to admixed samples. If source-population alphas differ, admixed samples receive per-sample intercepts from causal-architecture-weighted local ancestry exposure:

- `alpha_i = sum_p w_i,p * alpha_p`
- `w_i,p` is computed from local ancestry at selected causal haplotypes using squared expected log-odds effects, without using the sample's realized allele carriage

This makes admixed case prevalence an emergent output rather than a forced target. For admix-only cohorts or external source-prevalence priors, provide `alpha_by_population` for the source populations; an explicit `ADMIX` alpha is also accepted when a fixed admixed baseline is intended.

Local ancestry truth can be supplied either as the legacy compressed tract TSV or as a stable Zarr source object. The Zarr source object is a run-length encoded per-haplotype store with schema attrs and typed compressed arrays, not a pickled Python object.

Amplifier `msprime_tracts` populations can emit this source object:

```yaml
populations:
  - name: ADMIX
    engine: msprime_tracts
    msprime:
      sources:
        - {population: EUR, weight: 0.6}
        - {population: AFR, weight: 0.4}
      admixture_time_generations: 8.0
      write_ancestry_source_zarr: true
```

Phenotype configs can then consume it:

```yaml
cohorts:
  - name: train
    local_ancestry_source_pattern: /path/to/ADMIX.chr{chrom}.local_ancestry.zarr

local_ancestry:
  enabled: true
  mode: source_zarr
  ancestry_order: [EUR, AFR]
```

Use `mode: tracts_tsv` with `cohort.local_ancestry_tracts_pattern` for the legacy TSV path.

Binary phenotype outputs keep:

- `PHENO` as 0/1 labels (`0=control`, `1=case`)
- `LIABILITY` as the pre-binary continuous liability
- `BINARY_PROBABILITY` as the calibrated draw probability
- `BINARY_ALPHA` as the per-sample intercept used in the Bernoulli draw
- binary summary columns in `truth/component_summary.tsv` and `run_manifest.tsv`, including case/control counts, realized prevalence, `binary_alpha`, and `binary_alpha_by_population`

For GWAS, REGENIE switches from `--qt` to `--bt` when `trait_type=binary`; PLINK2 adds `--1` so the 0/1 labels are interpreted as control/case status. Round-trip validation compares binary GWAS effects on an approximate log-odds scale by multiplying liability-scale truth by `binary_liability_logit_scale / sqrt(realized_var_liability)` and reports `validation_status=validated_binary_log_odds`. This is a diagnostic for effect direction and gross calibration, not the same exact slope contract as quantitative GWAS.

### HapStore-native source-pair tables

`forgephast phenotype build-source-pairs` builds the causal/source-pair candidate table consumed by `causal_selection.source_pairs_file` directly from HapStore genotype blocks. It computes pooled and group-specific MAF columns from additive genotypes, applies a pooled MAF floor, and writes a TSV with variant ID, chromosome, position, source variant ID, `maf_pooled`, and configured group MAF columns.

```bash
forgephast phenotype build-source-pairs --config source_pairs.yaml
```

Minimal config:

```yaml
hapstore:
  layout: per_chr              # per_chr | single_store | per_pop_per_chr
  root_dir: /path/to/hapstores
  file_pattern: chr{chrom}.hapstore.zarr
  missing_sentinel: 255

samples:
  file: train_samples.tsv
  id_column: IID
  group_column: population

chromosomes: [1, 2, 3]

groups:
  - name: EUR
    column: population
    value: EUR
    maf_column: maf_eur
  - name: AFR
    column: population
    value: AFR
    maf_column: maf_afr

source_pairs:
  pooled_maf_min: 0.0005
  variant_block_size: 1024
  variant_id_column: variant_id
  pooled_maf_column: maf_pooled

output:
  source_pairs_file: inputs/source_pairs.tsv
  qc_file: inputs/source_pairs_qc.tsv
```

If `groups` is omitted, groups are derived from unique values in `samples.group_column` and written as `maf_<group>`. Subgroup columns can be expressed with filters:

```yaml
groups:
  - name: AFR_GULF
    column: source_population
    value: AFR_GULF
    filters:
      population: AFR
    maf_column: maf_afr_gulf
```

For `hapstore.layout: per_pop_per_chr`, set `samples.store_population_column` if it differs from `samples.group_column`; the file pattern must include both `{population}` and `{chrom}`. The first native mode is intentionally limited to candidate-table generation and does not construct LD proxy/tag pairs.

### Migration note: `beta_scale` column

The `beta_scale` column emitted on `truth/effect_sizes.tsv` was produced by the
deprecated per-population β rescaling step. With that step removed, the column
is no longer written. Consumers should switch to `effect_scale`, the
MAF-based scaling column still emitted by `sample_population_specific_betas`.

### Cross-population MAF tightness

`forgephast phenotype` can filter the causal pool by how tightly paired populations' MAFs agree,
using the max-over-pairs fold change `max(maf_i, maf_j) / min(maf_i, maf_j)`:

```yaml
causal_selection:
  ...
  tightness_filter:
    tier: moderate            # tight | moderate | extreme | off
    fold_change_cap: null     # optional numeric override of tier
    maf_floor: 1.0e-4         # clamp applied before division to handle zero MAF
```

Tiers map to caps:

- `tight`: fold change `<= 2.0`
- `moderate`: fold change `<= 8.0`
- `extreme` (default): fold change `<= 300.0`
- `off`: no cap (no filtering, no reporting)

A custom `fold_change_cap` (must be finite and `>= 1.0`) overrides the tier cap.
MAFs are clipped to `maf_floor` before the ratio is computed, so zero (or
near-zero) MAFs in one population produce a well-defined (large) fold change
rather than dividing by zero.

The filter runs early in `load_causal_pool` — on the broadest pool, before
bucket assignment — so tier changes can be compared on an equal footing.
Surviving variants' per-pair quantiles (50/75/90/99/max) are reported in
`shift_tightness_fold_change_quantiles_json` on both `component_summary.tsv`
and `run_manifest.tsv`, along with `shift_tightness_tier`,
`shift_tightness_fold_change_cap`, `shift_tightness_fold_change_max_kept`,
`shift_tightness_pre_filter_count`, and `shift_tightness_post_filter_count`.

## Phenotype performance TODO

The current phenotype runtime exposes useful I/O and chunking knobs:

- `genotype_input.hapstore.io_threads`
- `local_ancestry.chunk_n_samples`
- `local_ancestry.chunk_n_variants`
- grid-level `parallel_runs`

But the dominant single-case hotspot for large local-ancestry phenotypes is still the core fixed-effect computation in `polygenic_no_pleiotropy`, which is chunked but not yet truly parallelized across CPU workers.

Future optimization work for that path is intentionally deferred until after the PRS bakeoff phenotype release is stabilized. The detailed engineering plan is tracked in the thesis vault note:

- `03 Plans/2026/2026-04-10 ForgePhast Phenotype Parallelization Plan`

## Annotation-weighted causal sampling

`forgephast phenotype` can optionally enrich causal-variant sampling in functional regions while preserving the existing MAF-bucket fixed-plus-subset workflow.

When `causal_selection.annotation_sampling.enabled: true`:

- sampling remains bucketed by `causal_selection.maf_buckets`
- fixed-set and subset draws both switch from uniform to weighted sampling without replacement
- baseline weight is `1.0`
- overlapping annotations raise a variant's weight via the configured multiplier
- v1 uses `combine_rule: max` only

### Built-in default gene-window mode

The built-in preset is `GRCh38` only and derives one interval per gene from a bundled simple annotation table:

- `+` strand: `100 kb upstream of TSS` through `gene_end`
- `-` strand: `gene_start` through `100 kb upstream of TSS` on the reverse strand
- bundled source: GENCODE Human Release 49 (`GRCh38.p14`)
- packaged default subset: autosomes only, `protein_coding` genes only
- provenance: generated from `https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz` using `gene` features filtered to autosomal `protein_coding`

The package also includes a direct gene BED derived from the same source:

- `forgephast/phenotype/assets/grch38_gencode_v49_genes.bed`

Example:

```yaml
causal_selection:
  source_pairs_file: source_pairs.tsv
  variant_id_column: causal_id
  maf_columns_by_population:
    EUR: maf_eur
    AFR: maf_afr
  anchor_population: EUR
  maf_buckets:
    - name: low
      maf_min: 0.01
      maf_max: 0.05
      fixed_count: 100
      subset_count: 100
  annotation_sampling:
    enabled: true
    combine_rule: max

## Round-trip diagnostic plots

`forgephast phenotype` can now render a standard round-trip diagnostic set from existing GWAS outputs:

- truth-vs-estimated effect scatter
- Manhattan plot with true causals highlighted
- QQ plot
- top-k recovery curve
- enrichment summary bar plot

Optional group-stratified beta scatter is available when the user provides a sample-to-group TSV.

Config:

```yaml
roundtrip_plots:
  enabled: true
  formats: [png]
  dpi: 200
  sample_group_file: path/to/sample_groups.tsv
  id_column: IID
  group_column: population
```

When `roundtrip_plots.enabled: true`, plots are emitted automatically after successful GWAS + round-trip validation.

You can also rerender plots from an existing run without rerunning simulation or GWAS:

```bash
forgephast phenotype roundtrip-plots \
  --config path/to/run.yaml \
  --run-dir path/to/existing/run \
  --gwas-manifest path/to/existing/run/gwas_manifest.tsv
```

Outputs are written under:

- `<run_dir>/plots/roundtrip/`
- `<run_dir>/roundtrip_plot_manifest.tsv`
    default_gene_window:
      enabled: true
      genome_build: GRCh38
      upstream_bp: 100000
      weight_multiplier: 5.0
```

### Custom BED annotations

You can also provide one or more BED files with independent weight multipliers:

```yaml
causal_selection:
  annotation_sampling:
    enabled: true
    combine_rule: max
    variant_chrom_column: chrom
    variant_pos_column: pos
    default_gene_window:
      enabled: false
    custom_annotations:
      - name: enhancers
        bed_file: /path/to/enhancers.bed
        weight_multiplier: 3.0
      - name: promoters
        bed_file: /path/to/promoters.bed
        weight_multiplier: 2.0
```

Coordinate conventions:

- source-pool `pos` values are 1-based
- BED intervals are interpreted as 0-based half-open
- overlap is tested on the single-base point `pos - 1`

Truth outputs now include per-variant annotation metadata in `truth/causal_membership.tsv`:

- `sampling_weight`
- `annotation_hit`
- `annotation_labels`

and a grouped summary in `truth/causal_membership_summary.tsv` with counts and hit fractions by:

- `replicate`
- `architecture`
- `cohort`
- `maf_bucket`

## Phenotype GWAS prep performance guidance

For large round-trip GWAS runs, the main bottlenecks are usually genotype materialization and PCA refresh, not effect sampling.

As of `v0.4.0`, the default phenotype GWAS prep path is now:

1. per-chromosome HapStore -> PGEN export
2. per-chromosome monomorphic filtering
3. PLINK2 merge into one canonical genome-wide PGEN
4. LD-pruned cached PCA on the merged PGEN
5. one-time merged covariate preparation per `replicate x cohort`
6. PLINK2 and/or REGENIE GWAS on the cached merged PGEN

The legacy whole-store path is still available as a fallback with:

- `pgen_export.prep_mode: legacy_whole_store`

The staged-default production path is:

- `pgen_export.prep_mode: per_chrom_plink_merge`

### Recommended prep architecture

- Prefer chromosome-local preprocessing over one monolithic whole-genome prep stage.
- Filter monomorphic variants per chromosome before final genome-wide assembly.
- Export each chromosome to PGEN independently, then merge chromosome-level PGENs into the final genome-wide PGEN surface.
- Prefer PGEN-level merge/concat over temporary HapStore concat for final GWAS prep when possible.

This is the preferred direction because HapStore-to-PGEN export is reasonably efficient, while whole-panel HapStore concat is often the slower and more memory-fragile step.

### Why PGEN-first merge is preferable

- PLINK2 exposes explicit memory control for merge operations.
- PLINK2 merge is aggressively multithreaded and already optimized for this workload.
- Per-chromosome PGEN surfaces are easier to cache and reuse than a large temporary merged HapStore.
- Failures become easier to recover because a single chromosome can be regenerated without repeating the whole prep path.

### Important note about HapStore direct-copy fast paths

`forgephast` still relies on `hapstore.concat_hapstores()` for some intermediate sample-axis assembly before PGEN export. That concat step is much faster when `hapstore` can promote the work from blockwise NumPy copy to raw chunk-copy.

That direct-copy fast path only triggers when the source and output HapStores are layout-compatible. In practice, the important conditions are:

- same `chunk_variant_size`
- same `chunk_haplotype_size`
- same compressor and compression level
- same dtype/filter/fill-value contract
- no genotype sidecar write is requested during concat
- sample-axis chunk boundaries line up exactly with the source-store boundaries being copied

When those conditions are met, `hapstore` can copy chunk payloads directly instead of decoding and rewriting each `(hap_chunk, variant_chunk)` block through NumPy.

When those conditions are not met, `forgephast` correctly falls back to the slower generic concat path.

For future phenotype/GWAS-facing cohort regeneration, prefer stable shared chunking across populations if you want this fast path to apply consistently.

### Best practices for fast sample-axis concat

If you expect to use `forgephast phenotype` on a panel repeatedly, design the upstream HapStores for sample-axis concat from the start.

Recommended defaults:

- keep `chunk_variant_size` constant across all population stores for a chromosome
- keep `chunk_haplotype_size` constant across all population stores for a chromosome
- keep compressor type and compression level constant across all population stores
- prefer sample/haplotype chunk sizes that divide each source population cleanly
- for large phenotype/GWAS-facing cohorts, prefer larger variant chunks such as `chunk_variant_size: 1500`

Why this matters:

- most phenotype-prep concats are along the sample axis, not the variant axis
- matching layout and codec settings maximize eligibility for raw chunk-copy instead of decode-copy-reencode
- divisible sample/haplotype chunking reduces boundary chunks that span multiple source stores
- larger variant chunks reduce metadata churn and the number of per-block copies on wide chromosomes

In practice, the fastest `forgephast` prep path is:

- standardized chunking
- standardized compression
- sample-axis chunk boundaries aligned to source cohort sizes
- per-chromosome PGEN export and PLINK2 merge downstream

### Where the new direct-copy concat path helps most

The new `hapstore` fast path matters most in the intermediate sample-axis concat steps that happen before PGEN export.

It is most likely to help in:

- per-chromosome population stacking before phenotype/GWAS PGEN export
  - for example, combining `AFR/EUR/CSA/EAS` chromosome stores into one temporary chromosome-local panel
- repeated homogeneous-panel assembly where every population store was regenerated with the same layout contract
- any workflow where concat is along the sample axis and the source stores are already chunk-aligned

It is less relevant or unlikely to trigger in:

- final genome-wide assembly, which now prefers PLINK2 merge on chromosome-level PGENs
- variant-axis concat, where the current direct-copy optimization does not apply
- concat runs that request genotype sidecar materialization during the copy
- mixed-layout or mixed-compression stores
- source cohorts whose sample counts force many boundary chunks to span multiple stores

So the practical mental model is:

- the fast path is mainly a speedup for the remaining HapStore sample-axis assembly that still exists inside the new PGEN-first prep flow
- it is not a replacement for the downstream PLINK2 merge stage

### PCA guidance

- Do not compute ancestry PCs from the full dense association panel unless you specifically need that behavior.
- Prefer LD-pruned PCA over full-panel PCA for GWAS covariate generation.
- The default PLINK2 PCA path now supports:
  - `gwas.global_pcs.prune_before_pca`
  - `gwas.global_pcs.prune_window_kb`
  - `gwas.global_pcs.prune_step_variants`
  - `gwas.global_pcs.prune_r2`
- When the prune window is specified in kilobases, `forgephast` emits the
  PLINK2-compatible increment of `1` for `--indep-pairwise`.
- Cache the final PCA result by:
  - merged PGEN prefix
  - kept sample set
  - chromosome list
  - PCA settings
  - pruning settings

If those inputs have not changed, a rerun should reuse the existing PCA artifact instead of recomputing it.

### Other worthwhile speedups

- Precompute merged GWAS covariates once per replicate/cohort and reuse them across engines instead of rebuilding them separately for PLINK2 and REGENIE.
- Treat the merged genome-wide PGEN as the canonical reusable prep artifact for reruns.
- Prefer scheduler-level parallelism across chromosomes or replicate-level prep units over a single large threaded parent job when scaling on HPC.

### Practical rule of thumb

For large `forgephast phenotype gwas` runs, the target prep flow should be:

1. per-chromosome filter
2. per-chromosome PGEN export
3. PLINK2 merge into final genome-wide PGEN
4. LD-pruned cached PCA
5. GWAS engines

That flow is generally faster, easier to resume, and less OOM-prone than a prep path centered on temporary whole-genome HapStore concat.
