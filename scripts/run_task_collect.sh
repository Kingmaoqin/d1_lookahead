#!/bin/bash
# setsid 彻底脱离会话，避免像上次那样被会话回收连带 SIGTERM
cd /home/xqin5/diffusion_LLM/d1_lookahead
TAG=$1; GPU=$2; OFFSET=$3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
exec conda run -n p08_skilloverload python -u scripts/collect_task_labels.py \
  --n_screen 220 --n_prompts 150 --offset $OFFSET --K 8 --n_cand 4 \
  --rollout_batch 32 --tag $TAG >> logs/collect_$TAG.log 2>&1
