#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
A=$D/aux_models; cd /mnt/storage/lib_project/models
export SSHPASS=$(awk -F'`' '/\| password /{print $2;exit}' TRAINING_SERVER.md)
R=user@gpu-box
while :; do
  state=$(sshpass -e ssh -o ConnectTimeout=8 "$R" 'cat ~/codex_trt_qat_20261119/logs/router_more16_extract.exit 2>/dev/null || true' 2>/dev/null || true)
  [ -z "$state" ] || break
  sleep 60
done
[ "$state" = 0 ] || { echo "Extraction failed: $state"; exit 1; }
sshpass -e rsync -a -e 'ssh -o ConnectTimeout=8' "$R:codex_trt_qat_20261119/aux_models/router_more16/train.npy" "$A/router_more16_train.npy"
sshpass -e rsync -a -e 'ssh -o ConnectTimeout=8' "$R:codex_trt_qat_20261119/aux_models/router_more16/val.npy" "$A/router_more16_val.npy"
unset SSHPASS
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$A"
"$HPY" -u router_more16_qat.py
"$HPY" - <<'PY'
import numpy as np
x=np.load('router_more16_val.npy');n=(len(x)//8)*8;assert n>=96
np.save('router_more16_val_hwc.npy',x[:n].astype(np.float32).reshape(-1,1,1,1024));print('heldout evaluation',n,flush=True)
PY
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py router_more16_qat.har router_more16_val_hwc.npy router_more16_fp.npy fp
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/emu.py router_more16_qat.har router_more16_val_hwc.npy router_more16_q.npy q
"$HPY" - <<'PY'
import numpy as np
ok=True
for i,name in enumerate(('lang','kind','body')):
 a=np.load('router_more16_fp_%d.npy'%i).reshape(len(np.load('router_more16_val_hwc.npy')),-1);b=np.load('router_more16_q_%d.npy'%i).reshape(len(a),-1)
 match=float(np.mean(a.argmax(1)==b.argmax(1)));print('HELDOUT',name,'n',len(a),'agreement',match,'mae',np.abs(a-b).mean(),flush=True);ok &= match>=.99
print('EMULATION_GATE', 'PASS' if ok else 'FAIL', '99% per head; chip test still required',flush=True)
if not ok: raise SystemExit(2)
PY
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py router_more16_qat.har router_more16_qat.hef
"$HPY" -u /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py router_more16_qat.hef router_more16_qat.har
echo 'Compiled and emulation-gated; actual chip and original-FP comparison still required.'
