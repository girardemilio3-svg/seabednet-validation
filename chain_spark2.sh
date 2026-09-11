#!/bin/sh
# after the from-scratch chain: national fields for the v3 hazard heads on the Spark GPU, then the Coast Guard tests (live + archive) and the offshore scan
cd /home/fenexpertai/seabednet; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q CHAIN_DONE chain_spark.log 2>/dev/null; do sleep 120; done
echo "SPARK1_DONE $(date)" >> chain_spark2.log
HZ_BW=8 HZ_CKPT=hazard_small_v3ctl.pt HZ_SIZE=small HZ_AUX=0 HZ_BLOCKS=all_nat.txt HZ_OUT=hazard_nat_v3ctl python3 hazard_corridor.py > hazard_nat_v3ctl.log 2>&1
echo "INFER_CTL $(date) $(ls hazard_nat_v3ctl | wc -l)" >> chain_spark2.log
HZ_BW=8 HZ_CKPT=hazard_small_v3aux.pt HZ_SIZE=small HZ_AUX=1 HZ_BLOCKS=all_nat.txt HZ_OUT=hazard_nat_v3aux python3 hazard_corridor.py > hazard_nat_v3aux.log 2>&1
echo "INFER_AUX $(date) $(ls hazard_nat_v3aux | wc -l)" >> chain_spark2.log
for k in v3ctl v3aux; do
  HZ_DIR=hazard_nat_$k TAG=_$k python3 navwarn_hindcast.py > navwarn_hindcast_$k.log 2>&1
  NW_POINTS=recon_navwarn/cancelled_danger_points.csv HZ_DIR=hazard_nat_$k TAG=_${k}_archive python3 navwarn_hindcast.py > navwarn_hindcast_${k}_archive.log 2>&1
  echo "NAVWARN_$k $(date)" >> chain_spark2.log
done
HZ_DIR=hazard_nat_v3aux PMIN=0.6 SH_MIN=-18 SH_MAX=-3 OUT=shoal_candidates_v3aux.csv python3 shoal_list_v2.py > shoal_list_v3aux.log 2>&1
echo "CHAIN_DONE $(date)" >> chain_spark2.log
