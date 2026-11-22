#!/bin/bash
# zoo/cov_exact.sh NET: re-capture (all op attributes) + graph26 --cont with the generic atomic-chain executor
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; W=$R/zoo/work/$1; P=/mnt/storage/anaconda3/envs/general
cd $R; . env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=99
IMG=/mnt/storage/lib_project/models/Hailo_model/datasets/coco/val2017/000000000139.jpg
[ -f $W/quant.har ] || { echo "$1 no quant.har"; exit 1; }
rm -rf $W/cap; rm -rf $W/cap2
(cd hailoqat && ALL_OPS=1 AUTO_ATTRS=1 $HPY capture.py $W/quant.har $IMG $W/cap2 > $W/capture2.log 2>&1) || { echo "$1 capture failed"; exit 1; }
$HPY - $W/cap2 <<'PY' > /dev/null 2>&1
import enum, pickle, sys
import numpy as np
d = pickle.load(open(f"{sys.argv[1]}/_all_ops.pkl", "rb"))
def conv(v):
    if isinstance(v, dict): return {str(k): conv(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return type(v)(conv(x) for x in v)
    if isinstance(v, enum.Enum): return conv(v.value)
    if isinstance(v, (bool, int, float, str)) or v is None: return v
    if isinstance(v, np.ndarray):
        return np.array([conv(x) for x in v.ravel()], dtype=object).reshape(v.shape) if v.dtype == object else np.array(v)
    if isinstance(v, np.generic): return v.item()
    try: return np.asarray(v)
    except Exception: return str(v)
pickle.dump({str(k): conv(v) for k, v in d.items()}, open(f"{sys.argv[1]}/_all_ops_np.pkl", "wb"))
PY
rm -f $W/cap2/_all_ops.pkl
(cd hailoqat && env -u LD_PRELOAD OMP_NUM_THREADS=8 LD_LIBRARY_PATH=$P/lib $P/bin/python graph26.py $W/$1.hn $W/cap2 $IMG cpu --cont --json=$W/cov2.json > $W/check2.log 2>&1)
echo "$1 $(tail -2 $W/check2.log | head -1)"
