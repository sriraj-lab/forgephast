# forgephast

FORGEPHAST combines population amplification and phenotype simulation workflows around a HapStore-first runtime.

## Amplifier donor input contract

## Amplifier simulation contract

Amplifier now has one simulation path: the shard-pool engine.

- `forgephast amplify simulate` is the canonical simulation command.
- `n_shards=1` preserves the old single-shard behavior, but still runs through the same shard-pool implementation and returns the same output artifacts.
- The separate public `simulate-pool` command/API no longer exists.
- Runtime defaults for shard count, worker count, threads-per-task, and task start method live under `simulation.n_shards`, `simulation.task_workers`, `simulation.threads_per_task`, and `simulation.task_start_method`.

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

These geometry parameters control whether the cross-pop beta construction is valid. They do not directly lock the final phenotype variance.

Later in simulation, the realized fixed genetic component is rescaled to the requested target variance:

- `genetic_fixed_scale = sqrt(target_h2_fixed / Var(genetic_raw))`
- emitted per-variant truth betas are multiplied by the same scale

So changing `sigma_beta` or jitter settings affects the geometry of sampled effects, while `variance_budget.h2_fixed` still controls the final fixed genetic variance target.

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
