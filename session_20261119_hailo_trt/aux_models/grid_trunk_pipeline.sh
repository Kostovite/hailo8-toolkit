#!/bin/bash
set -u
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
A=$D/aux_models
run() { local name=$1; shift; echo "$(date -Is) START $name"; "$@" >"$D/logs/${name}.log" 2>&1; local rc=$?; echo "$rc" >"$D/logs/${name}.exit"; echo "$(date -Is) END $name rc=$rc"; return "$rc"; }
run grid_trunk_export bash "$A/grid_export_trunk.sh" || exit 1
run grid_trunk_calib bash "$A/grid_quantize_trunk.sh" || exit 1
echo 'Calibration only; held-out accuracy, compilation and chip gate are required.'
