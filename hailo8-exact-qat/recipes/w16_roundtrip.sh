#!/bin/bash
# round trip: trained 16-bit ints -> HAR -> capture -> eval (must equal the training eval)
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; P=/mnt/storage/anaconda3/envs/general
cd $R; . env.sh
cd hailoqat
env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib $P/bin/python ints2npz.py ../runs/n_w16L_3ep/ints_step6900.pt ../capture_n_L0w16top8 ../small/w16_s4600_ints.npz
CUDA_VISIBLE_DEVICES=99 $HPY writeback.py ../small/y26n_L0w16top8.har ../small/w16_s4600_ints.npz ../small/y26n_w16_s4600.har 2>&1 | grep -vE "^\[info\]" | tail -5
cd $R
PT=$R/small/y26n_v14sec/weights/best.pt EVAL_GPUS="1" ./har_pipeline.sh small/y26n_w16_s4600.har n_w16_s4600_rt 1 2>&1 | grep -v libtinfo | tail -3
echo "TRAINING EVAL: 0.7836650528049885 / 0.6647178013426911"
