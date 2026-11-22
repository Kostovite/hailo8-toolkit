#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
kind=$1; gpu=$2; cal=orient; [ "$kind" = flat3d ] && cal=flat
mkdir -p "$D/capture_$kind"
/mnt/storage/anaconda3/envs/general/bin/python - "$P/calib/${cal}_1024.npy" "$D/${kind}_input.npy" <<'PY'
import sys,numpy as np
x=np.load(sys.argv[1],mmap_mode='r')[0]
np.save(sys.argv[2],x.transpose(2,0,1).astype(np.float32))
PY
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=$gpu TF_FORCE_GPU_ALLOW_GROWTH=true ALL_OPS=1 OMP_NUM_THREADS=2
"$HPY" - "$D/${kind}_all16.har" "$D/${kind}.hn" <<'PY'
import json,sys
from hailo_sdk_client import ClientRunner
h=ClientRunner(har=sys.argv[1]).get_hn();json.dump(json.loads(h) if isinstance(h,str) else h,open(sys.argv[2],'w'))
PY
cd "$R"
"$HPY" -u "$R/hailoqat/capture.py" "$D/${kind}_all16.har" "$D/${kind}_input.npy" "$D/capture_$kind"
