#!/bin/bash
set -u
cd /home/xqin5/diffusion_LLM/d1_lookahead
exec conda run --no-capture-output -n llm python -u scripts/collect_labels.py \
  --n_prompts 200 --K 24 --n_cand 6 --n_cand_conf 3 --horizon 16 \
  --order ancestral --offset 400 --tag freshA --backbone mdlm \
  --traj_batch 6 --rollout_batch 48
