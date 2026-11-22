#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
C="--max_steps 300 --eval_every 300"
./exp.sh e1_lowlr   $C --lr_k 1e-4 --lr_b 5e-3 --ovf 0
./exp.sh e2_bound   $C --lr_k 1e-4 --lr_b 5e-3 --ovf 0 --k_radius 1 --b_radius 64
./exp.sh e3_bound3x $C --lr_k 3e-4 --lr_b 1.5e-2 --ovf 0 --k_radius 1 --b_radius 64
./exp.sh e4_kdonly  $C --lr_k 1e-4 --lr_b 5e-3 --ovf 0 --gt 0
./exp.sh e5_ovf     $C --lr_k 1e-4 --lr_b 5e-3 --ovf 1e-3
echo "QUEUE1 DONE"
