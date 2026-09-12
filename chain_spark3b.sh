#!/bin/sh
# Hazard-head A/B with the winter radar channel (fine-tunes from hazard_small_v2.pt), national field for both, Coast Guard tests, offshore scan.
# Appends CHAIN_DONE to chain_spark3.log so chain_spark4 (full-size radar model) starts afterwards.
cd /home/fenexpertai/seabednet; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "START3b $(date)" >> chain_spark3.log
EXCL="-96.90,71.35;-91.349,69.717;-112.672,67.970;-97.537,68.5625;-94.307,63.993;-91.520,63.620;-61.5697,56.4508"
rm -f hazard_small_v4ctl.pt hazard_small_v4s1.pt
HZ_SIZE=small HZ_STEPS=1500 HZ_BATCH=8 HZ_LR=3e-5 HZ_COVMIN=0.95 HZ_SHOALMAX=-0.5 HZ_S1=0 HZ_INIT=hazard_small_v2.pt HZ_CKPT=hazard_small_v4ctl.pt HZ_EXCLUDE="$EXCL" python3 train_hazard.py > hazard_v4ctl.log 2>&1
echo "HZ_v4ctl $(date) $(tail -c 200 hazard_v4ctl.log | tr '\r' '\n' | grep '\[' | tail -1)" >> chain_spark3.log
HZ_SIZE=small HZ_STEPS=1500 HZ_BATCH=8 HZ_LR=3e-5 HZ_COVMIN=0.95 HZ_SHOALMAX=-0.5 HZ_S1=1 HZ_INIT=hazard_small_v2.pt HZ_CKPT=hazard_small_v4s1.pt HZ_EXCLUDE="$EXCL" python3 train_hazard.py > hazard_v4s1.log 2>&1
echo "HZ_v4s1 $(date) $(tail -c 200 hazard_v4s1.log | tr '\r' '\n' | grep '\[' | tail -1)" >> chain_spark3.log
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
