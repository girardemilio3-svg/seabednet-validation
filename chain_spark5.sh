#!/bin/sh
# after chain_spark4: hazard heads FROM SCRATCH, with and without winter radar (3000 steps, bs 8, LR 2e-4), field for the radar one, Coast Guard tests.
cd /home/fenexpertai/seabednet; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q CHAIN_DONE chain_spark6.log 2>/dev/null; do sleep 120; done
echo "START $(date)" >> chain_spark5.log
EXCL="-96.90,71.35;-91.349,69.717;-112.672,67.970;-97.537,68.5625;-94.307,63.993;-91.520,63.620;-61.5697,56.4508"
for run in ctl s1; do
  case $run in ctl) X="HZ_S1=0";; s1) X="HZ_S1=1";; esac
  rm -f hazard_small_v5$run.pt; env $X HZ_SIZE=small HZ_STEPS=3000 HZ_BATCH=8 HZ_LR=2e-4 HZ_COVMIN=0.95 HZ_SHOALMAX=-0.5 HZ_CKPT=hazard_small_v5$run.pt HZ_EXCLUDE="$EXCL" python3 train_hazard.py > hazard_v5$run.log 2>&1
  echo "HZ_v5$run $(date) $(tail -c 200 hazard_v5$run.log | tr '\r' '\n' | grep '\[' | tail -1)" >> chain_spark5.log
done
HZ_BW=8 HZ_CKPT=hazard_small_v5s1.pt HZ_SIZE=small HZ_S1=1 HZ_BLOCKS=all_nat.txt HZ_OUT=hazard_nat_v5s1 python3 hazard_corridor.py > hazard_nat_v5s1.log 2>&1
echo "INFER_v5s1 $(date) $(ls hazard_nat_v5s1 | wc -l)" >> chain_spark5.log
HZ_DIR=hazard_nat_v5s1 TAG=_v5s1 python3 navwarn_hindcast.py > navwarn_hindcast_v5s1.log 2>&1
NW_POINTS=recon_navwarn/cancelled_danger_points.csv HZ_DIR=hazard_nat_v5s1 TAG=_v5s1_archive python3 navwarn_hindcast.py > navwarn_hindcast_v5s1_archive.log 2>&1
echo "NAVWARN_v5s1 $(date)" >> chain_spark5.log
HZ_BW=8 HZ_CKPT=hazard_small_v5ctl.pt HZ_SIZE=small HZ_S1=0 HZ_BLOCKS=all_nat.txt HZ_OUT=hazard_nat_v5ctl python3 hazard_corridor.py > hazard_nat_v5ctl.log 2>&1
HZ_DIR=hazard_nat_v5ctl TAG=_v5ctl python3 navwarn_hindcast.py > navwarn_hindcast_v5ctl.log 2>&1
NW_POINTS=recon_navwarn/cancelled_danger_points.csv HZ_DIR=hazard_nat_v5ctl TAG=_v5ctl_archive python3 navwarn_hindcast.py > navwarn_hindcast_v5ctl_archive.log 2>&1
echo "CHAIN_DONE $(date)" >> chain_spark5.log
