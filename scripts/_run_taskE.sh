#!/bin/bash
cd /home/xqin5/diffusion_LLM/d1_lookahead
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n p08_skilloverload python -u \
  scripts/estimate_all.py --task --seeds 50 --out data/estimate_task.json \
  > logs/estimate_task.log 2>&1
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n p08_skilloverload python -u \
  scripts/task_state_readout.py --seeds 50 --out data/task_state_readout.json \
  > logs/task_state_readout.log 2>&1
