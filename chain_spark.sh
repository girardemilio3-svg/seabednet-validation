#!/bin/sh
# From-scratch A/B/C/D on the Spark GPU (tiny model, 100 m corpus, pre-2016 soundings): no extra channels / +DEM+S2 / +winter S1 / +raw gravity with leakage-free anchor.
cd /home/fenexpertai/seabednet; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COMMON='V5_CORPUS=tiles_nat:100:3857 V5_TEMPORAL=tiles_nat=index_out V5_SIZE=tiny V5_STEPS=3000 V5_BATCH=8 V5_WORKERS=4 V5_LR=2e-4 V5_CUDA_FRAC=0.35'
for run in ctl0 aux0 s10 grav0; do
  case $run in ctl0) X="V5_AUX=0";; aux0) X="V5_AUX=1";; s10) X="V5_S1=1";; grav0) X="V5_GRAV=1 V5_PRIOR=planetary/gravity_prior_canada_nosid.npz";; esac
  rm -f v5_tiny_$run.pt; env $COMMON $X V5_CKPT=v5_tiny_$run.pt python3 v5_train.py > v5_$run.log 2>&1
  echo "TRAIN_$run $(date) $(tail -c 200 v5_$run.log | tr '\r' '\n' | grep '\[' | tail -1)" >> chain_spark.log
  env $X TE_BW=8 V5_SIZE=tiny V5_CKPT=v5_tiny_$run.pt TAG=temporal_$run python3 temporal_eval.py corridor_blocks.txt > temporal_eval_$run.log 2>&1
  echo "EVAL_$run $(date) $(grep -o '"mae_model": [0-9.]*' temporal_eval_$run.log | head -1)" >> chain_spark.log
done
echo "CHAIN_DONE $(date)" >> chain_spark.log
