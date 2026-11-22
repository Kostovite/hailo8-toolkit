#!/bin/bash
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/probe
export DEVICE=0000:02:00.0 CUDA_VISIBLE_DEVICES=
#DIST=normal $HPY -u silent/chip_vs_emu.py emu 8 silent/key14.txt > silent/key14_emu.log 2>&1
for c in $(sed "s|.*/zoo/ops/||" silent/key14.txt); do
  echo "== $c"; DIST=normal flock ../turbo/.chip_0000_02_00.0.lock $HPY silent/judge.py ../zoo/ops/$c 2>&1 | grep -v "Warn\|\[info\]" | grep "SNR\|equal\|Error" | cut -c1-160
  [ -f ../zoo/ops/$c/emu_ref_n.npz ] || echo no-ref
done
echo DONE
