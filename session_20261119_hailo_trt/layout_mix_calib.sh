#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=2 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
OPT_LEVEL=0 "$HPY" -u "$D/quantize_layout_mix.py" "$D/r5_nofuse_parsed.har" "$D/r5_nofuse_mix16.har" /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/calib/layout_1024.npy mix16
