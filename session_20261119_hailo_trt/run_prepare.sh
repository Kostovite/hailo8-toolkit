#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
cd "$D"
kind=$1; gpu=$2
export CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
source /mnt/storage/anaconda3/etc/profile.d/conda.sh
conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u "$D/export_preproc.py" "$kind" "$D/${kind}_chip.onnx"
python -u /mnt/storage/lib_project/hailo8-exact-qat/tools/ops/onnx_hailo_compat.py "$D/${kind}_chip.onnx" "$D/${kind}_compat.onnx"
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=$gpu TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
"$HPY" -u "$P/parse.py" "$D/${kind}_compat.onnx" "$D/${kind}_parsed.har" "$kind"
cal=orient; [ "$kind" = flat3d ] && cal=flat
OPT_LEVEL=0 "$HPY" -u "$P/quantize.py" "$D/${kind}_parsed.har" "$D/${kind}_all16.har" "$P/calib/${cal}_1024.npy" all16
printf 'PREP_OK\n'
