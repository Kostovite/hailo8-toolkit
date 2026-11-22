#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
C="--epochs 1 --eval_every 1000 --lr_k 6e-4 --lr_b 3e-2 --ovf 0"
GPUS="0 1 2" EVAL_ALL=1 ./exp.sh q3a_r4 $C --k_radius 4 --b_radius 256
GPUS="0 1 2" EVAL_ALL=1 ./exp.sh q3b_r2_apu $C --k_radius 2 --b_radius 128 --apu_train
echo "Q3 DONE"
