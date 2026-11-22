#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=2
"$HPY" -u "$D/aux_models/layout_inspect_graph.py" "$D/r5_nofuse_mix16.har"
