#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119; A=$D/aux_models
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=2 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$A"
"$HPY" - <<'PY'
import numpy as np
x=np.load('grid_g5val_320.npz')['images'][:16];np.save('grid_trunk_val_nhwc.npy',x[...,None]);print('heldout',x.shape,flush=True)
PY
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py gridnet_trunk_all16.har grid_trunk_val_nhwc.npy grid_trunk_val_fp.npy fp
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py gridnet_trunk_all16.har grid_trunk_val_nhwc.npy grid_trunk_val_q.npy q
"$HPY" - <<'PY'
import numpy as np
for i,name in enumerate(('p2','p4')):
 a=np.load('grid_trunk_val_fp_%d.npy'%i,mmap_mode='r');b=np.load('grid_trunk_val_q_%d.npy'%i,mmap_mode='r');err=den=0
 for j in range(len(a)):
  x=a[j].astype(np.float32);y=b[j].astype(np.float32);err+=np.abs(x-y).mean();den+=np.abs(x).mean()
 print('HELDOUT',name,'n',len(a),'relative_feature_mae',err/max(den,1e-9),flush=True)
print('Feature check only: table metrics and actual chip still required.',flush=True)
PY
