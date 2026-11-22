#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119/aux_models
source /mnt/storage/anaconda3/etc/profile.d/conda.sh;conda activate general
export PATH="$CONDA_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u "$D/export_router.py" /mnt/storage/lib_project/models/Dolphin "$D/router_dynamic.onnx"
# Hailo requires a static batch; export separate fixed-batch copy.
python - "$D/router_dynamic.onnx" "$D/router_fixed.onnx" <<'PY'
import onnx,sys
m=onnx.load(sys.argv[1]);
for t in list(m.graph.input)+list(m.graph.output):t.type.tensor_type.shape.dim[0].dim_value=1
onnx.checker.check_model(m);onnx.save(m,sys.argv[2])
PY
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/parse.py "$D/router_fixed.onnx" "$D/router_parsed.har" routercur
