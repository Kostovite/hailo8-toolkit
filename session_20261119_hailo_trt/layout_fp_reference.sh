#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
source /mnt/storage/anaconda3/etc/profile.d/conda.sh;conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=2
cd "$D"
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u "$D/layout_fp_reference.py" "$1"
