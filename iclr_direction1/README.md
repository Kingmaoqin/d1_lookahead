# Direction One ICLR evidence bundle

Start here:

1. `DIRECTION1_ICLR_FINAL_EVIDENCE.md` — final scientific story and claim limits.
2. `DIRECTION1_ICLR_MASTER_RESULTS.csv` — authoritative paper-number table (one paper-usable result per row).
3. `_ALL_RESULT_CANDIDATES.csv` — broad searchable pool, including all experiment-registry rows.
4. `OLD_VS_NEW_EVIDENCE.md` — conflict and supersession ledger.
5. `build_iclr_evidence.py` — deterministic builder for both CSV files and raw task-reward alignment audit.

Audit/provenance files:

- `VREAD_INDEPENDENT_RECOMPUTE.log`
- `TASK_REWARD_ALIGNMENT_RECOMPUTE.json`
- `CLAUDE_PRIOR_REPORT_SNAPSHOT.md`
- `FILE_INVENTORY.tsv`
- `GIT_PROVENANCE.txt`

Rebuild from the repository root:

```bash
python iclr_direction1/build_iclr_evidence.py
```

The builder intentionally keeps exploratory registry entries in the candidate pool but promotes only curated, source-linked rows into the master table.
