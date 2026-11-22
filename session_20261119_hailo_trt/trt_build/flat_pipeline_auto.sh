#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
export PYTHONPATH="$D/trt_build/deps_compat:$D/flat_qat:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
cd "$D/trt_build"
"$D/venv/bin/python" -u flat_head_fp32.py
"$D/venv/bin/python" -u build_flat_engine.py "$D/artifacts/trt/flat3d_qat500_headfp32.onnx" "$D/artifacts/trt/flat3d_qat500_headfp32.engine"
"$D/venv/bin/python" -u test_flat_headfp32.py
printf 'PASS\n' > "$D/artifacts/trt/flat3d_headfp32_VALIDATED"
