#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
R=$PWD
HN=$R/L2a16.hn CAP=$R/capture_L2a16 PORT=29620 GPUS="0 1 2" EVAL_ALL=1 ./exp2.sh final1_L2_r4_3ep --epochs 3 --eval_every 1000 --lr_k 6e-4 --lr_b 3e-2 --ovf 0 --k_radius 4 --b_radius 256
echo "FINAL1 DONE"
