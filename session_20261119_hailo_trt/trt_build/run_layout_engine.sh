#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
export PYTHONPATH="$D/layout_qat/deps_correct:$D/trt_build/deps_compat:$D/layout_qat:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
cd "$D/trt_build"
"$D/venv/bin/python" -X faulthandler -u export_layout.py
"$D/venv/bin/python" -u build_flat_engine.py "$D/artifacts/trt/layout_r5_mixed_qat250.onnx" "$D/artifacts/trt/layout_r5_mixed_qat250.engine"
