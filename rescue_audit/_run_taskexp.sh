#!/bin/bash
set -u
cd /home/xqin5/diffusion_LLM/d1_lookahead
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec setsid conda run --no-capture-output -n p08_skilloverload python -u \
  scripts/collect_task_labels.py \
  --n_screen 260 --n_prompts 180 --offset 440 --K 8 --n_cand 6 \
  --rollout_batch 32 --tag taskC --resume
