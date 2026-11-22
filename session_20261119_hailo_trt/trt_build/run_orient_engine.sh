#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
while [ ! -f "$D/logs/trt_deps.exit" ];do sleep 3;done
[ "$(cat "$D/logs/trt_deps.exit")" = 0 ] || exit 1
cd "$D/trt_build"
export PYTHONPATH="$D/trt_build/deps:$D/orient_full:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
"$D/venv/bin/python" -u export_orient.py
"$D/venv/bin/python" -u build_engine.py "$D/artifacts/trt/orient3_qat500_features.onnx" "$D/artifacts/trt/orient3_qat500_features.engine"
