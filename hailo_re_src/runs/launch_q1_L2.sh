#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
GPUS="0 1 2" EVAL_ALL=1 nohup ./exp.sh q1_a16_e3 --epochs 1 --eval_every 1000 --lr_k 3e-4 --lr_b 1.5e-2 --ovf 0 --k_radius 1 --b_radius 64 > runs/q1.log 2>&1 &
echo "q1 $!"
. env.sh
export CUDA_VISIBLE_DEVICES=3 TF_FORCE_GPU_ALLOW_GROWTH=true
OPT_LEVEL=2 TRAIN_GLOB="$PWD/data_v14sec/train/images/*" NMS="" nohup $HPY b2_levelN.py ../yolo26m_v14sec_parsed.har ../yolo26m_v14sec_L2a16.har 896 896 1024 out/alls_a16heads.txt > out/b2_y26_L2a16.log 2>&1 &
echo "L2 $!"
