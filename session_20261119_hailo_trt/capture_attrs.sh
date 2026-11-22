#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
kind=$1; gpu=$2
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=$gpu TF_FORCE_GPU_ALLOW_GROWTH=true ALL_OPS=1 AUTO_ATTRS=1 OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u "$R/hailoqat/capture.py" "$D/${kind}_all16.har" "$D/${kind}_input.npy" "$D/capture_attrs_$kind"
