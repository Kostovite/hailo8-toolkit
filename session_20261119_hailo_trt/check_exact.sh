#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
kind=$1; gpu=$2
while [ ! -f "$D/logs/capture_${kind}.exit" ]; do sleep 15; done
[ "$(cat "$D/logs/capture_${kind}.exit")" = 0 ] || { echo capture-failed; exit 1; }
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
CUDA_VISIBLE_DEVICES=99 "$HPY" - "$D/capture_$kind" <<'PY'
import enum,pickle,sys,numpy as np
d=pickle.load(open(sys.argv[1]+'/_all_ops.pkl','rb'))
def conv(v):
 if isinstance(v,dict): return {str(k):conv(x) for k,x in v.items()}
 if isinstance(v,list): return [conv(x) for x in v]
 if isinstance(v,tuple): return tuple(conv(x) for x in v)
 if isinstance(v,enum.Enum): return conv(v.value)
 if isinstance(v,bool): return bool(v)
 if isinstance(v,int): return int(v)
 if isinstance(v,float): return float(v)
 if isinstance(v,str) or v is None: return v
 if isinstance(v,np.ndarray): return np.array([conv(x) for x in v.ravel()],dtype=object).reshape(v.shape) if v.dtype==object else np.array(v)
 if isinstance(v,np.generic): return v.item()
 try:return np.asarray(v)
 except Exception:return str(v)
pickle.dump({str(k):conv(v) for k,v in d.items()},open(sys.argv[1]+'/_all_ops_np.pkl','wb'));print('converted',len(d))
PY
/mnt/storage/anaconda3/envs/general/bin/python - "$D/${kind}_input.npy" "$D/${kind}_input_nhwc.npy" <<'PYCODE'
import numpy as np,sys
x=np.load(sys.argv[1]); np.save(sys.argv[2],x.transpose(1,2,0)[None])
PYCODE
cd "$R/hailoqat"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 env -u LD_PRELOAD LD_LIBRARY_PATH=/mnt/storage/anaconda3/envs/general/lib /mnt/storage/anaconda3/envs/general/bin/python -u graph26.py "$D/$kind.hn" "$D/capture_$kind" "$D/${kind}_input_nhwc.npy" cuda --cont --json="$D/${kind}_exact.json"
/mnt/storage/anaconda3/envs/general/bin/python - "$D/${kind}_exact.json" <<'PYCODE'
import json,sys
r=json.load(open(sys.argv[1])); bad=[x for x in r["layers"] if x["status"]!="exact"]; print("failed_exact_layers",len(bad)); sys.exit(bool(bad))
PYCODE
