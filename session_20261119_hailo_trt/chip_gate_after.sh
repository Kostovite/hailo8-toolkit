#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
kind=$1
while [ ! -f "$D/logs/chip_chain_${kind}.exit" ]; do sleep 20; done
code=$(cat "$D/logs/chip_chain_${kind}.exit"); [ "$code" = 0 ] || { echo "compile/guard/chip chain failed: $code"; exit 1; }
if [ "$kind" = flat3d ]; then
 "$D/run_current_gates.sh" flat flat3d_fp.npy flat3d_q16.npy flat3d_chip_all16.npy
else
 "$D/run_current_gates.sh" orient orient3_fp.npy orient3_q16.npy orient3_chip_all16.npy
fi
