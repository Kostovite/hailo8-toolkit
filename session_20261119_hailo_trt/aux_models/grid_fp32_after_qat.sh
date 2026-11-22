#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
while [ ! -f "$D/logs/grid_qat.exit" ];do sleep 20;done
export PYTHONPATH="$D/trt_build/deps_compat:$D/aux_models:$D/aux_models/wordlid:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
cd "$D/aux_models"
"$D/venv/bin/python" -X faulthandler -u grid_fp32_export.py
"$D/venv/bin/python" -u "$D/trt_build/build_flat_engine.py" "$D/aux_models/grid_fp32.onnx" "$D/aux_models/grid_fp32.engine"
