#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
export CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
source /mnt/storage/anaconda3/etc/profile.d/conda.sh; conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u /mnt/storage/lib_project/hailo8-exact-qat/tools/ops/onnx_hailo_compat.py "$D/r5_ord5_chip.onnx" "$D/r5_ord5_compat.onnx"
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=2 TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u "$P/parse.py" "$D/r5_ord5_compat.onnx" "$D/r5_ord5_parsed.har" r5ord5
OPT_LEVEL=0 "$HPY" -u "$P/quantize.py" "$D/r5_ord5_parsed.har" "$D/r5_ord5_int8.har" "$P/calib/layout_1024.npy" int8
