#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
kind=$1
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=2
cd "$D"
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py "$D/${kind}_all16.har" "$D/${kind}_all16.hef"
"$HPY" -u /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py "$D/${kind}_all16.hef" "$D/${kind}_all16.har"
frames="$P/eval/orient_canvas.npy"; [ "$kind" = flat3d ] && frames="$P/eval/flat_frames.npy"
flock "$D/hailo_chip.lock" "$HPY" -u "$P/chip.py" "$D/${kind}_all16.hef" "$frames" "$D/${kind}_chip_all16.npy"
"$HPY" - "$D/${kind}_chip_all16.npy" "$D/${kind}_fp.npy" "$D/${kind}_q16.npy" <<'PY'
import numpy as np,sys
chip=np.load(sys.argv[1],mmap_mode='r');print('chip shape',chip.shape)
for p in sys.argv[2:]:
 a=np.load(p,mmap_mode='r');assert chip.shape==a.shape,(chip.shape,a.shape)
 total=0.; mx=0.; n=0
 for i in range(len(chip)):
  d=np.abs(chip[i]-a[i]);total+=float(d.sum(dtype=np.float64));mx=max(mx,float(d.max()));n+=d.size
 print('chip vs',p,'mean_abs',total/n,'max_abs',mx,flush=True)
PY
