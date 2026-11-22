#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
while [ ! -f "$D/logs/orient_trt_cpu.exit" ];do sleep 5;done
[ "$(cat "$D/logs/orient_trt_cpu.exit")" = 0 ] || { echo 'engine build failed';exit 1; }
export PYTHONPATH="$D/trt_build/deps_compat:$D/orient_full:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
cd "$D/trt_build"
"$D/venv/bin/python" -u test_orient_engine.py
