#!/bin/bash
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
export HQ_EXT_DIR=/mnt/storage/tmp/claude-1005/-mnt-storage-lib-project-models/350d256b-4abf-4ee9-8419-160b9eb7ff76/scratchpad/hqnew
C="--max_steps 1000 --eval_every 1000 --lr_k 6e-4 --lr_b 3e-2 --ovf 0 --k_radius 2 --b_radius 128"
PORT=29610 GPUS="3" ./exp2.sh q5a_1gpu_base $C
PORT=29610 GPUS="3" ./exp2.sh q5b_1gpu_apusmall $C --apu_train --lr_apu_m 0.002 --lr_apu_o 0.02 --apu_m_radius 4 --apu_o_radius 8
echo "Q5 DONE"
