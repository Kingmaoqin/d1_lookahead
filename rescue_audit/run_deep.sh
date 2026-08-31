#!/bin/bash
# R1 深度探针：在若干层上跑完整 P0–P13 族（含低秩双线性 P2）。
set -u
cd /home/xqin5/diffusion_LLM/d1_lookahead
ARM=${1:-MDLM_anc}
TGT=${2:-A_pertok}
for L in 3 6 9 11; do
  echo "############ layer $L ############"
  conda run --no-capture-output -n llm python rescue_audit/r1_screen.py \
    --arm "$ARM" --targets "$TGT" --criteria within_r2 \
    --split_seeds 1 --probe_seeds 3 --epochs 400 --layers $L \
    --which P0 P1 P2k P2 P3 P4 P5 P6 P7 P7k P8 P9 P13 \
    --kron_dims 16 32 --skip_controls \
    --out rescue_audit/results/R1deep_${ARM}_${TGT}_L${L}.json
done
echo "ALL DEEP DONE"
