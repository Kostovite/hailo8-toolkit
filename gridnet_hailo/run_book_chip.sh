#!/bin/bash
cd /mnt/storage/lib_project/models/Dolphin/wordlid_vert
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export LD_PRELOAD=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16
$HPY scan_units.py /mnt/storage/lib_project/models/Hailo_model/nlp/T1_straight_pages 1-664 \
  /mnt/storage/lib_project/models/Hailo_model/nlp/units_T1/clean_chip --clean \
  --hef /mnt/storage/lib_project/models/Hailo_model/gridnet_hailo/L0_int8.hef > logs/scan_clean_chip.log 2>&1
echo EXIT $? >> logs/scan_clean_chip.log
