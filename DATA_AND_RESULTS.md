# Data and result layout

The repository keeps human-readable conclusions, machine-readable result files,
and raw recomputation inputs separate.

## Start here

- `docs/EXPERIMENT_LEDGER.md`: chronological experiment ledger and terminology.
- `rescue_audit/FINAL_RESCUE_REPORT.md`: audited scientific conclusions.
- `rescue_audit/CLAUDE_CROSS_AUDIT.md`: cross-audit, including corrected positive findings.
- `rescue_audit/EXPERIMENT_REGISTRY.csv`: row-level index of fitted probes and metrics.
- `rescue_audit/results/batch_exports/dataset_summary.json`: compact dataset totals.
- `rescue_audit/results/batch_exports/batch_catalog.csv`: one row per raw shard.
- `rescue_audit/results/batch_exports/trend_summary.csv`: summaries by generation step and
  prompt stratum for quick trend filtering.
- `rescue_audit/results/batch_exports/cards/`: one JSON result card per raw shard.
- `rescue_audit/results/full_inventory/asset_manifest.csv`: SHA-256 inventory of
  every data file, derived result, sealed probe model, and run log in the project.
- `rescue_audit/results/full_inventory/cards/`: lightweight schema/statistics cards
  for every NPZ/NPY asset, including historical, ablation, smoke, and model files.

## Raw recomputation inputs

The complete Task C, Task D, and Task E label/feature shards are stored under
`data/labels_taskC`, `data/labels_taskD`, and `data/labels_taskE_svamp`.  The `.npz`
files are tracked with Git LFS. Each card records its SHA-256 checksum, byte size,
array shapes/dtypes, document/state/row counts, and scalar-label statistics.

After cloning, obtain the full inputs with:

```bash
git lfs pull
```

Verify or regenerate every card and compact table with:

```bash
/home/xqin5/.conda/envs/p08_skilloverload/bin/python \
  rescue_audit/export_batch_catalog.py
```

Core analyses load these shards through `src/dataset.py`. The frozen Task D result
is `rescue_audit/results/CONFIRMATORY_taskD_frozen.json`; its sealed pre-label models
and preregistration are under `rescue_audit/sealed_taskD_primary/` and
`rescue_audit/TASKD_PREREGISTRATION.md`.

Project-trained probe weights are included, notably the three sealed Task D models
under `rescue_audit/sealed_taskD_primary/`. Foundation-model checkpoints are not
redistributed: they remain third-party Hugging Face assets and are identified by
the pinned model IDs/revisions in collection metadata and scripts.

## Provenance rule

Raw shards are append-only experimental evidence. Derived JSON/CSV reports may be
regenerated, but should not replace or silently modify raw shards. Compare the
recorded SHA-256 values before recomputation.
