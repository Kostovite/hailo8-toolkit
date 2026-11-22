#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
while [ ! -f "$D/logs/layout_mix_calib.exit" ];do sleep 30;done
[ "$(cat "$D/logs/layout_mix_calib.exit")" = 0 ] || { echo 'STOP: calibration failed';exit 1; }
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=2
cd "$D"
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py "$D/r5_nofuse_mix16.har" "$D/r5_nofuse_mix16.hef"
"$HPY" -u /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py "$D/r5_nofuse_mix16.hef" "$D/r5_nofuse_mix16.har"
echo 'HEF_AND_GUARD_DONE: chip and accuracy tests still required.'
