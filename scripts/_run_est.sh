#!/bin/bash
cd /home/xqin5/diffusion_LLM/d1_lookahead
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
exec conda run --no-capture-output -n p08_skilloverload python -u \
  scripts/estimate_all.py --seeds 50 --out data/estimate_all.json \
  > logs/estimate_all.log 2>&1
