#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
export PYTHONPATH="$D/trt_build/deps_compat:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
cd "$D/trt_build"
"$D/venv/bin/python" -u rewrite_flat_prelu.py
"$D/venv/bin/python" -u build_flat_engine.py "$D/artifacts/trt/flat3d_qat500_prelu.onnx" "$D/artifacts/trt/flat3d_qat500_prelu.engine"
