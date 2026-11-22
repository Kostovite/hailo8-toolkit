#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119/aux_models
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=2 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py "$D/router_all16.har" "$D/router_val_hwc.npy" "$D/router_fp.npy" fp
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py "$D/router_all16.har" "$D/router_val_hwc.npy" "$D/router_q16.npy" q
"$HPY" - "$D" <<'PY'
import numpy as np,sys,os
D=sys.argv[1]
for i,name in enumerate(('lang','kind','body')):
 a=np.load(D+'/router_fp_%d.npy'%i)[:100].reshape(100,-1);b=np.load(D+'/router_q16_%d.npy'%i)[:100].reshape(100,-1);print(name,'fp/q agreement',np.mean(a.argmax(1)==b.argmax(1)),'mean_abs',np.abs(a-b).mean(),'max',np.abs(a-b).max(),flush=True)
PY
