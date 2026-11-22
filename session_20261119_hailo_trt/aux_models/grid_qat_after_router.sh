#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
while [ ! -f "$D/logs/router_qat.exit" ];do sleep 30;done
while [ ! -f "$D/aux_models/grid_g5train_320.npz" ] || [ ! -f "$D/aux_models/grid_g5val_320.npz" ];do sleep 20;done
export PYTHONPATH="$D/trt_build/deps_compat:$D/aux_models:$D/aux_models/wordlid:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
cd "$D/aux_models"
"$D/venv/bin/python" -u grid_qat.py
