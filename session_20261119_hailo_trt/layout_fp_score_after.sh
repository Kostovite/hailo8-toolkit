#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
for k in 0 1; do
 while [ ! -f "$D/logs/layout_fp_$k.exit" ]; do sleep 20; done
 [ "$(cat "$D/logs/layout_fp_$k.exit")" = 0 ] || { echo "prediction $k failed";exit 1; }
done
source /mnt/storage/anaconda3/etc/profile.d/conda.sh;conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u "$D/layout_fp_score.py"
