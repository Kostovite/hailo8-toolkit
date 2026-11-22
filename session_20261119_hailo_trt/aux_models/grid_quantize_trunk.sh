#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119/aux_models
P=/mnt/storage/lib_project/models/Hailo_model/gridnet_hailo
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2 OPT_LEVEL=0 PYTHONPATH="$P:${PYTHONPATH:-}"
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" - "$D/gridnet_trunk_parsed.har" "$D/grid_trunk_all16.alls" <<'PY'
from hailo_sdk_client import ClientRunner
import sys
hn=ClientRunner(har=sys.argv[1]).get_hn_model();layers=list(hn.stable_toposort()); types=('conv','dense','ew_add','elementwise')
ls=[x.name.split('/')[-1] for x in layers if any(t in str(x.op).lower().replace('layertype.','') for t in types)]
assert ls;print('16bit layers',len(ls),ls,flush=True)
open(sys.argv[2],'w').write('quantization_param(['+', '.join(ls)+'], precision_mode=a16_w16)\n')
PY
"$HPY" -u "$P/quantize.py" "$D/gridnet_trunk_parsed.har" "$D/gridnet_trunk_all16.har" 2048 "$D/grid_trunk_all16.alls"
