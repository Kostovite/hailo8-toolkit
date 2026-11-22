#!/bin/bash
# usage: run_q.sh NAME OPT_LEVEL GPU [ALLS]
cd /mnt/storage/lib_project/models/Hailo_model/gridnet_hailo
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export LD_PRELOAD=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD
export CUDA_VISIBLE_DEVICES=$3 TF_FORCE_GPU_ALLOW_GROWTH=true OPT_LEVEL=$2
$HPY quantize.py gridnet_vert1_parsed.har $1.har 1024 $4 > logs/q_$1.log 2>&1
echo EXIT $? >> logs/q_$1.log
