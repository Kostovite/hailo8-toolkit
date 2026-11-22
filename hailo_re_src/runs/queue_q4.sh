#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
while kill -0 3880691 2>/dev/null; do sleep 15; done
R=$PWD
HN=$R/L2a16.hn CAP=$R/capture_L2a16 GPUS="0 1 2" EVAL_ALL=1 ./exp.sh q4_fromL2_r2 --epochs 1 --eval_every 1000 --lr_k 6e-4 --lr_b 3e-2 --ovf 0 --k_radius 2 --b_radius 128
echo "Q4 DONE"
