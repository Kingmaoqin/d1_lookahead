# Task-D / Task-E long-run recovery

All collectors are append-safe at the document level.  If a process is
interrupted, rerun its exact command with `--resume`; complete documents in
existing shards are skipped.  Incomplete in-memory documents were never saved
and are recomputed from the locked seed.

## Task D, worker 0 (physical GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 /home/xqin5/.conda/envs/p08_skilloverload/bin/python -u scripts/collect_task_labels.py --n_screen 600 --n_prompts 570 --offset 700 --K 8 --n_cand 6 --rollout_batch 64 --tag taskD --resume --num_workers 2 --worker_index 0 --shard_examples 96
```

## Task D, worker 1 (physical GPU 3)

```bash
CUDA_VISIBLE_DEVICES=3 /home/xqin5/.conda/envs/p08_skilloverload/bin/python -u scripts/collect_task_labels.py --n_screen 600 --n_prompts 570 --offset 700 --K 8 --n_cand 6 --rollout_batch 64 --tag taskD --resume --num_workers 2 --worker_index 1 --shard_examples 96
```

## Task E, SVAMP (sharded over physical GPUs 0 and 2)

```bash
CUDA_VISIBLE_DEVICES=0,2 /home/xqin5/.conda/envs/p08_skilloverload/bin/python -u scripts/collect_task_labels.py --dataset svamp --n_screen 300 --n_prompts 200 --offset 0 --K 8 --n_cand 6 --rollout_batch 8 --tag taskE_svamp --resume --num_workers 1 --worker_index 0 --shard_model --reserve_gb 1.6 --shard_examples 96
```

## Completion invariants

* Task D: 570 documents, 2,280 states, 13,680 candidate rows.
* Task E: 200 documents, 800 states, 4,800 candidate rows.
* Every state has exactly six candidate rows and every document has exactly
  four states / 24 rows.
* `A_task`, cheap controls, and hidden features must all be finite.
* Task D document IDs must not overlap Task C.

After Task D validates, run `rescue_audit/taskD_apply_frozen_primary.py` once
before any Task-D refitting.  This preserves the preregistered frozen-transfer
endpoint.
