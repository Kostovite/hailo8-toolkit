#!/bin/bash
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/probe
export DEVICE=0000:03:00.0 CUDA_VISIBLE_DEVICES=
DIST=normal $HPY -u silent/chip_vs_emu.py emu 6 silent/key6.txt > silent/key6_emu.log 2>&1
for d in $(cat silent/key6.txt); do echo "== ${d##*/zoo/ops/}"; DIST=normal flock ../turbo/.chip_0000_03_00.0.lock $HPY silent/judge.py $d 2>&1 | grep -E "batch  |equal|Error"; done
echo DONE
