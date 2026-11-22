#!/bin/bash
# DFC level 4 (AdaRound) + 16-bit heads with uslib's patched DFC overlay (DALI GPU reader, per-layer CPU-reader retry)
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
. env.sh
export CUDA_VISIBLE_DEVICES=0,1 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH=$PWD/dfc_patched
OPT_LEVEL=4 TRAIN_GLOB="$PWD/data_v14sec/train/images/*" NMS="" nohup $HPY b2_levelN.py ../yolo26m_v14sec_parsed.har ../yolo26m_v14sec_L4a16.har 896 896 1024 out/alls_a16heads.txt > out/b2_y26_L4a16_dali_2gpu.log 2>&1 &
echo "L4 $!"
