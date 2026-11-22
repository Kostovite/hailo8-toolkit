#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 TORCH_CUDA_ARCH_LIST=5.2
unset LD_PRELOAD LD_LIBRARY_PATH
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u "$D/aux_models/router_exact_work/smoke.py"
