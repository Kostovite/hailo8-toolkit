#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
while [ ! -f "$D/logs/layout_mix_calib.exit" ]; do sleep 30; done
[ "$(cat "$D/logs/layout_mix_calib.exit")" = 0 ] || { echo 'STOP: layout calibration failed';exit 1; }
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=3 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u "$P/emu.py" "$D/r5_nofuse_mix16.har" "$P/eval/layout_x.npy" "$D/layout_r5_mix16.npy" q
source /mnt/storage/anaconda3/etc/profile.d/conda.sh;conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2
unset LD_PRELOAD
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u "$P/layout_gate.py" score /mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/r5_ord5.pt "$D/layout_r5_mix16"
echo 'EMULATION_AND_AP_DONE: compare against layout_fp_score.log; not a QAT/chip promotion.'
