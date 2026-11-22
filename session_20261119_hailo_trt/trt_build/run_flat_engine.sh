#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
while [ ! -f "$D/logs/orient_trt_test.exit" ];do sleep 5;done
export PYTHONPATH="$D/trt_build/deps_compat:$D/flat_qat:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
cd "$D/trt_build"
"$D/venv/bin/python" -X faulthandler -u export_flat.py
"$D/venv/bin/python" -u build_flat_engine.py "$D/artifacts/trt/flat3d_qat500.onnx" "$D/artifacts/trt/flat3d_qat500.engine"
