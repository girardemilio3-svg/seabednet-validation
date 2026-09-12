#!/bin/sh
# matched control for the full-size radar model: same recipe (small, full corpus, bs 8, 4000 steps, from scratch) without radar.
cd /home/fenexpertai/seabednet; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "START $(date)" >> chain_spark6.log
rm -f v5_small_ctlfull.pt
V5_CORPUS="tiles_nat:100:3857,tiles10:10:3857" V5_TEMPORAL="tiles_nat=index_out,tiles10=index_out10" V5_SIZE=small V5_STEPS=4000 V5_BATCH=8 V5_WORKERS=6 V5_LR=2e-4 V5_AUX=0 V5_CKPT=v5_small_ctlfull.pt V5_CUDA_FRAC=0.45 python3 v5_train.py > v5_ctlfull.log 2>&1
echo "TRAIN_ctlfull $(date)" >> chain_spark6.log
TE_BW=8 V5_SIZE=small V5_CKPT=v5_small_ctlfull.pt TAG=temporal_ctlfull python3 temporal_eval.py corridor_blocks.txt > temporal_eval_ctlfull.log 2>&1
echo "EVAL_ctlfull $(date) $(grep -o '"mae_model": [0-9.]*' temporal_eval_ctlfull.log | head -1)" >> chain_spark6.log
echo "CHAIN_DONE $(date)" >> chain_spark6.log
