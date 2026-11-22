#!/bin/bash
# zoo/cov_one.sh NET: exactness coverage of one Model Zoo network on our integer emulator (CPU only).
#   hailomz parse -> zoo model script + optimization_level=0, 64 COCO calib images -> HSim bit-exact capture of one
#   input (ALL_OPS) -> graph26.py --cont: every layer checked against HSim -> work/NET/cov.json
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; Z=$R/zoo; N=$1; W=$Z/work/$N
P=/mnt/storage/anaconda3/envs/general
cd $R; . env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=99 HMZ_DATA=$Z/data TF_NUM_INTEROP_THREADS=4 TF_NUM_INTRAOP_THREADS=8
mkdir -p $W; cd $W
CALIB=/mnt/storage/lib_project/models/Hailo_model/datasets/coco/val2017
IMG=$CALIB/000000000139.jpg
st() { echo "$1" > status; }
if [ ! -s parsed.har ]; then st parse; $HPY $Z/hmz.py parse $N --hw-arch hailo8 > parse.log 2>&1 || { st parse_FAIL; exit 1; }; mv $N.har parsed.har; fi
if [ ! -s quant.har ]; then
  st optimize
  A=$Z/repo_v2.18/hailo_model_zoo/cfg/alls/generic/$N.alls
  ( [ -f $A ] && grep -vE "model_optimization_flavor|post_quantization_optimization|pre_quantization_optimization\(.*(adaround|finetune)" $A
    echo; echo "model_optimization_flavor(optimization_level=0, compression_level=0)"
    echo "model_optimization_config(calibration, calibset_size=64)" ) |
    sed "s#\(\.\./\)*postprocess_config#$Z/repo_v2.18/hailo_model_zoo/cfg/postprocess_config#" > l0.alls
  $HPY $Z/hmz.py optimize $N --har parsed.har --model-script l0.alls --calib-path $CALIB > opt.log 2>&1 || { st optimize_FAIL; exit 1; }; mv $N.har quant.har
fi
st capture
$HPY - quant.har $N.hn <<'PY' 2>&1 | { grep -vE "^\[info\]" || true; }
import json, sys
from hailo_sdk_client import ClientRunner
hn = ClientRunner(har=sys.argv[1]).get_hn()
json.dump(json.loads(hn) if isinstance(hn, str) else hn, open(sys.argv[2], "w"))
PY
rm -rf cap
(cd $R/hailoqat && ALL_OPS=1 $HPY capture.py $W/quant.har $IMG $W/cap > $W/capture.log 2>&1) || { st capture_FAIL; exit 1; }
$HPY - cap <<'PY' 2>&1 | { grep -vE "^\[info\]" || true; }
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
st check
(cd $R/hailoqat && env -u LD_PRELOAD OMP_NUM_THREADS=8 LD_LIBRARY_PATH=$P/lib $P/bin/python graph26.py $W/$N.hn $W/cap $IMG cpu --cont --json=$W/cov.json > $W/check.log 2>&1) || { st check_FAIL; exit 1; }
tail -2 check.log
[ "$KEEP_CAP" = 1 ] || rm -rf cap  # captures are large; cov.json + check.log keep the result
st done
