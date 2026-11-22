#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
OPT_LEVEL=0 "$HPY" -u "$P/quantize.py" "$D/orient3_parsed.har" "$D/orient3_int8.har" "$P/calib/orient_1024.npy" int8
