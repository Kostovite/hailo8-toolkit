#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1
unset LD_PRELOAD LD_LIBRARY_PATH
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/layout_gate.py score /mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/r5_ord5.pt "$D/layout_r5_mix16"
