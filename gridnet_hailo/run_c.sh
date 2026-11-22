#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/gridnet_hailo
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=99
$HPY /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py L0_int8.har L0_int8.hef > logs/compile_L0_int8.log 2>&1
echo EXIT $? >> logs/compile_L0_int8.log
