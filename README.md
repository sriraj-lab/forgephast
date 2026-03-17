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
