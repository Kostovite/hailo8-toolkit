#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
A=$D/aux_models
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=2
cd "$A"
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py "$A/router_all16.har" "$A/router_all16.hef"
"$HPY" -u /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py "$A/router_all16.hef" "$A/router_all16.har"
flock "$D/hailo_chip.lock" "$HPY" -u "$A/router_chip_gate.py"
