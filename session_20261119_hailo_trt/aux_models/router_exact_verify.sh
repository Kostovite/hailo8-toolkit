#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119; A=$D/aux_models; R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=99
"$HPY" - "$A/router_exact_capture" "$A/router_exact_input.npy" "$A/router_exact_input_nhwc.npy" <<'PY'
import enum,pickle,sys,numpy as np
p,src,dst=sys.argv[1:];d=pickle.load(open(p+'/_all_ops.pkl','rb'))
def c(v):
 if isinstance(v,dict):return {str(k):c(x) for k,x in v.items()}
 if isinstance(v,list):return [c(x) for x in v]
 if isinstance(v,tuple):return tuple(c(x) for x in v)
 if isinstance(v,enum.Enum):return c(v.value)
 if isinstance(v,(bool,int,float,str)) or v is None:return v
 if isinstance(v,np.ndarray):return np.array([c(x) for x in v.ravel()],dtype=object).reshape(v.shape) if v.dtype==object else np.array(v)
 if isinstance(v,np.generic):return v.item()
 try:return np.asarray(v)
 except Exception:return str(v)
pickle.dump({str(k):c(v) for k,v in d.items()},open(p+'/_all_ops_np.pkl','wb'));np.save(dst,np.load(src).transpose(1,2,0)[None]);print('converted',len(d),flush=True)
PY
cd "$R/hailoqat"
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 env -u LD_PRELOAD LD_LIBRARY_PATH=/mnt/storage/anaconda3/envs/general/lib /mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u graph26.py "$A/router_exact.hn" "$A/router_exact_capture" "$A/router_exact_input_nhwc.npy" cuda --cont --json="$A/router_exact_report.json"
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python - "$A/router_exact_report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));bad=[x for x in r['layers'] if x['status']!='exact'];print('NONEXACT',len(bad),bad,flush=True);raise SystemExit(bool(bad))
PY
