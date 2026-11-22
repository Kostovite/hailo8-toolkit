#!/bin/bash
# usage: run_e.sh HAR CTX TAG LIMIT GPU
cd /mnt/storage/lib_project/models/Hailo_model/gridnet_hailo
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export LD_PRELOAD=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD
export CUDA_VISIBLE_DEVICES=$5 TF_FORCE_GPU_ALLOW_GROWTH=true
$HPY eval_hailo.py $1 $2 $3 $4 >> logs/eval_$(basename $1 .har)_$2.log 2>&1
