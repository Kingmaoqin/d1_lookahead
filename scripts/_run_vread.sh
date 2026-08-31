#!/bin/bash
# detached wrapper: bare `nohup conda run` stays attached to the tool-call
# shell and dies with it (this killed a run at 20/220 earlier).
GPU=$1; OFF=$2; N=$3; TAG=$4
cd /home/xqin5/diffusion_LLM/d1_lookahead
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU
exec conda run --no-capture-output -n p08_skilloverload python -u \
  scripts/collect_v_readout.py --n_prompts $N --offset $OFF --tag $TAG \
  > logs/${TAG}.log 2>&1
