#!/bin/sh
# After chain_spark (grav0): (1) winter-radar model from scratch with FULL radar coverage (s11), (2) hazard head A/B fine-tune
# with/without radar (v4ctl / v4s1) from hazard_small_v2.pt, (3) national field for v4s1, (4) Coast Guard tests live + archive, (5) offshore scan.
cd /home/fenexpertai/seabednet; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q CHAIN_DONE chain_spark.log 2>/dev/null; do sleep 120; done
until grep -q S1_DONE s1_build_hz.log 2>/dev/null; do sleep 120; done
echo "START $(date)" >> chain_spark3.log
COMMON='V5_CORPUS=tiles_nat:100:3857 V5_TEMPORAL=tiles_nat=index_out V5_SIZE=tiny V5_STEPS=3000 V5_BATCH=8 V5_WORKERS=4 V5_LR=2e-4 V5_CUDA_FRAC=0.35'
rm -f v5_tiny_s11.pt; env $COMMON V5_S1=1 V5_CKPT=v5_tiny_s11.pt python3 v5_train.py > v5_s11.log 2>&1
env V5_S1=1 TE_BW=8 V5_SIZE=tiny V5_CKPT=v5_tiny_s11.pt TAG=temporal_s11 python3 temporal_eval.py corridor_blocks.txt > temporal_eval_s11.log 2>&1
echo "EVAL_s11 $(date) $(grep -o '"mae_model": [0-9.]*' temporal_eval_s11.log | head -1)" >> chain_spark3.log
EXCL="-96.90,71.35;-91.349,69.717;-112.672,67.970;-97.537,68.5625;-94.307,63.993;-91.520,63.620;-61.5697,56.4508"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
rm -f hazard_small_v4ctl.pt hazard_small_v4s1.pt
HZ_SIZE=small HZ_STEPS=1500 HZ_BATCH=8 HZ_LR=3e-5 HZ_COVMIN=0.95 HZ_SHOALMAX=-0.5 HZ_S1=0 HZ_INIT=hazard_small_v2.pt HZ_CKPT=hazard_small_v4ctl.pt HZ_EXCLUDE="$EXCL" python3 train_hazard.py > hazard_v4ctl.log 2>&1
echo "HZ_v4ctl $(date)" >> chain_spark3.log
HZ_SIZE=small HZ_STEPS=1500 HZ_BATCH=8 HZ_LR=3e-5 HZ_COVMIN=0.95 HZ_SHOALMAX=-0.5 HZ_S1=1 HZ_INIT=hazard_small_v2.pt HZ_CKPT=hazard_small_v4s1.pt HZ_EXCLUDE="$EXCL" python3 train_hazard.py > hazard_v4s1.log 2>&1
echo "HZ_v4s1 $(date)" >> chain_spark3.log
HZ_BW=8 HZ_CKPT=hazard_small_v4s1.pt HZ_SIZE=small HZ_S1=1 HZ_BLOCKS=all_nat.txt HZ_OUT=hazard_nat_v4s1 python3 hazard_corridor.py > hazard_nat_v4s1.log 2>&1
echo "INFER_v4s1 $(date) $(ls hazard_nat_v4s1 | wc -l)" >> chain_spark3.log
HZ_DIR=hazard_nat_v4s1 TAG=_v4s1 python3 navwarn_hindcast.py > navwarn_hindcast_v4s1.log 2>&1
NW_POINTS=recon_navwarn/cancelled_danger_points.csv HZ_DIR=hazard_nat_v4s1 TAG=_v4s1_archive python3 navwarn_hindcast.py > navwarn_hindcast_v4s1_archive.log 2>&1
echo "NAVWARN_v4s1 $(date)" >> chain_spark3.log
HZ_BW=8 HZ_CKPT=hazard_small_v4ctl.pt HZ_SIZE=small HZ_S1=0 HZ_BLOCKS=all_nat.txt HZ_OUT=hazard_nat_v4ctl python3 hazard_corridor.py > hazard_nat_v4ctl.log 2>&1
HZ_DIR=hazard_nat_v4ctl TAG=_v4ctl python3 navwarn_hindcast.py > navwarn_hindcast_v4ctl.log 2>&1
NW_POINTS=recon_navwarn/cancelled_danger_points.csv HZ_DIR=hazard_nat_v4ctl TAG=_v4ctl_archive python3 navwarn_hindcast.py > navwarn_hindcast_v4ctl_archive.log 2>&1
echo "NAVWARN_v4ctl $(date)" >> chain_spark3.log
HZ_DIR=hazard_nat_v4s1 PMIN=0.6 SH_MIN=-18 SH_MAX=-3 OUT=shoal_candidates_v4s1.csv python3 shoal_list_v2.py > shoal_list_v4s1.log 2>&1
echo "CHAIN_DONE $(date)" >> chain_spark3.log
