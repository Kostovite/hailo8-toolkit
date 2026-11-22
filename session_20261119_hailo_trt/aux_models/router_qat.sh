#!/bin/bash
set -euo pipefail
D=$HOME/codex_trt_qat_20261119
export PYTHONPATH="$D/trt_build/deps_compat:$D/aux_models:$D/aux_models/wordlid:$D/orient_full:$HOME/codex_docparse_20261119/vendor" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3
cd "$D/aux_models"
"$D/venv/bin/python" -u router_qat_pipeline.py
