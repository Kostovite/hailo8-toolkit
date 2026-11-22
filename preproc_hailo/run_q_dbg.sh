#!/bin/bash
# usage: run_q.sh MODEL NAME MODE [GPU]
cd /mnt/storage/lib_project/models/Hailo_model/preproc_hailo
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export LD_PRELOAD=/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:$LD_PRELOAD
export CUDA_VISIBLE_DEVICES=${4:-} TF_FORCE_GPU_ALLOW_GROWTH=true OPT_LEVEL=0
$HPY quantize_dbg.py $1_parsed.har $2.har calib/$( [ $1 = r3b ] && echo layout || echo ${1%2} )_1024.npy $3 > logs/q_$2.log 2>&1; echo EXIT $? >> logs/q_$2.log
