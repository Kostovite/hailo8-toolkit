#!/bin/bash
# bq.sh GPU args... : run bench_qat.py in the torch env on one GPU
P=/mnt/storage/anaconda3/envs/general
G=$1; shift
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat
CUDA_VISIBLE_DEVICES=$G env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib PATH=$P/bin:/usr/bin:/bin $P/bin/python bench_qat.py "$@" 2>&1 | grep -v -iE "warn|torchvision"
