#!/bin/bash
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
while [ ! -f "$D/logs/install_layout.exit" ]; do sleep 3; done
[ "$(cat "$D/logs/install_layout.exit")" = 0 ] || { echo install-failed; exit 1; }
export CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
exec "$D/layout_env/bin/python" -u "$D/export_layout.py" /mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/r5_ord5.pt "$D/r5_ord5_chip.onnx"
