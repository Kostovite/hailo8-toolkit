#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
A=$D/aux_models; R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true ALL_OPS=1 AUTO_ATTRS=1 OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
"$HPY" - "$A" <<'PY'
import numpy as np,json,sys
from hailo_sdk_client import ClientRunner
p=sys.argv[1];a=np.load(p+'/router_more16_train.npy')[0];np.save(p+'/router_exact_input.npy',a.reshape(1024,1,1));h=ClientRunner(har=p+'/router_all16.har').get_hn();json.dump(json.loads(h) if isinstance(h,str) else h,open(p+'/router_exact.hn','w'))
PY
cd "$R"
"$HPY" -u hailoqat/capture.py "$A/router_all16.har" "$A/router_exact_input.npy" "$A/router_exact_capture"
"$HPY" -u hailoqat/bias_lattice.py "$A/router_all16.har" "$A/router_exact_capture"
