#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
export CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
source /mnt/storage/anaconda3/etc/profile.d/conda.sh
conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u "$D/export_layout.py" /mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/r5_ord5.pt "$D/r5_ord5_chip.onnx"
