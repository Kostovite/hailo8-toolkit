#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
source /mnt/storage/anaconda3/etc/profile.d/conda.sh; conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u "$D/add_qkv_identity_gemm.py" "$D/r5_ord5_compat.onnx" "$D/r5_ord5_compat_id.onnx"
python - "$D/r5_ord5_compat_id.onnx" <<'PY'
import onnx,sys
m=onnx.load(sys.argv[1]);n=[x.name for x in m.graph.node if '/IdMatMul' in x.name];print('QKV_IDENTITY_COUNT',len(n),n);assert len(n)==3
PY
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u "$P/parse.py" "$D/r5_ord5_compat_id.onnx" "$D/r5_ord5_id_parsed.har" r5ord5
