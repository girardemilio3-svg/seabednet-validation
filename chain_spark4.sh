#!/bin/sh
# after chain_spark3: the full-size candidate — small (34.8M) model, full corpus (100 m + 10 m), winter radar channel, 4000 steps, from scratch
# (same size/steps as the published v5_small_temporal.pt = 13.26 m), then the corridor temporal benchmark.
cd /home/fenexpertai/seabednet; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q CHAIN_DONE chain_spark3.log 2>/dev/null; do sleep 120; done
echo "START $(date)" >> chain_spark4.log
rm -f v5_small_s1full.pt
V5_CORPUS="tiles_nat:100:3857,tiles10:10:3857" V5_TEMPORAL="tiles_nat=index_out,tiles10=index_out10" V5_SIZE=small V5_STEPS=4000 V5_BATCH=8 V5_WORKERS=6 V5_LR=2e-4 V5_S1=1 V5_CKPT=v5_small_s1full.pt V5_CUDA_FRAC=0.45 python3 v5_train.py > v5_s1full.log 2>&1
echo "TRAIN_s1full $(date)" >> chain_spark4.log
V5_S1=1 TE_BW=8 V5_SIZE=small V5_CKPT=v5_small_s1full.pt TAG=temporal_s1full python3 temporal_eval.py corridor_blocks.txt > temporal_eval_s1full.log 2>&1
echo "EVAL_s1full $(date) $(grep -o '"mae_model": [0-9.]*' temporal_eval_s1full.log | head -1)" >> chain_spark4.log
echo "CHAIN_DONE $(date)" >> chain_spark4.log
