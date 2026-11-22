#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119/aux_models
source /mnt/storage/anaconda3/etc/profile.d/conda.sh;conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u "$D/export_gridnet_two.py" "$D/gridnet_two_chip.onnx"
python -u /mnt/storage/lib_project/hailo8-exact-qat/tools/ops/onnx_hailo_compat.py "$D/gridnet_two_chip.onnx" "$D/gridnet_two_compat.onnx"
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/parse.py "$D/gridnet_two_compat.onnx" "$D/gridnet_two_parsed.har" gridtwo
