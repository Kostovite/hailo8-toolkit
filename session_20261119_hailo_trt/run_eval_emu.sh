#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
kind=$1; gpu=$2; test=orient_canvas; [ "$kind" = flat3d ] && test=flat_frames
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=$gpu TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u "$P/emu.py" "$D/${kind}_all16.har" "$P/eval/${test}.npy" "$D/${kind}_fp.npy" fp
"$HPY" -u "$P/emu.py" "$D/${kind}_all16.har" "$P/eval/${test}.npy" "$D/${kind}_q16.npy" q
