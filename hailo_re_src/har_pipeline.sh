#!/bin/bash
# har_pipeline.sh HAR TAG [GPU]: score any quantized HAR on exact Hailo-8 arithmetic.
#   1. export TAG.hn   2. bit-exact capture with HSim on GPU (ALL_OPS) -> capture_TAG
#   3. numpy-only op pickle   4. graph26 exactness check on the capture page
#   5. bias_lattice annotation (a16_w16 layers)   6. qatfast full-val eval (evalgpu.sh)
# GPU: the device for steps 2/4 (default 3); EVAL_GPUS: devices for step 5 (default "0 1 2 3")
set -e
HAR=$(realpath $1); TAG=$2; GPU=${3:-3}
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
P=/mnt/storage/anaconda3/envs/general
cd $R
. env.sh
I=$R/data_v14sec/val/images/HCMUS_2020_DoNgocKieuChinh_1_p001.jpg
CUDA_VISIBLE_DEVICES=99 $HPY - "$HAR" "$R/$TAG.hn" <<'EOF' 2>&1 | grep -vE "^\[info\]"
import json, sys
from hailo_sdk_client import ClientRunner
hn = ClientRunner(har=sys.argv[1]).get_hn()
json.dump(json.loads(hn) if isinstance(hn, str) else hn, open(sys.argv[2], "w"))
print("hn ->", sys.argv[2])
EOF
rm -rf $R/capture_$TAG
(cd hailoqat && CUDA_VISIBLE_DEVICES=$GPU ALL_OPS=1 TF_FORCE_GPU_ALLOW_GROWTH=true $HPY capture.py $HAR $I $R/capture_$TAG \
  > $R/out/capture_$TAG.log 2>&1)
grep -E "captured" $R/out/capture_$TAG.log
CUDA_VISIBLE_DEVICES=99 $HPY - "$R/capture_$TAG" <<'EOF' 2>&1 | grep -vE "^\[info\]"
import enum, pickle, sys
import numpy as np
d = pickle.load(open(f"{sys.argv[1]}/_all_ops.pkl", "rb"))
def conv(v):  # hailo enums subclass str/int and would need the SDK to unpickle
    if isinstance(v, dict): return {str(k): conv(x) for k, x in v.items()}
    if isinstance(v, list): return [conv(x) for x in v]
    if isinstance(v, tuple): return tuple(conv(x) for x in v)
    if isinstance(v, enum.Enum): return conv(v.value)
    if isinstance(v, bool): return bool(v)
    if isinstance(v, int): return int(v)
    if isinstance(v, float): return float(v)
    if isinstance(v, str): return str(v)
    if v is None: return None
    if isinstance(v, np.ndarray):
        return np.array([conv(x) for x in v.ravel()], dtype=object).reshape(v.shape) if v.dtype == object else np.array(v)
    if isinstance(v, np.generic): return v.item()
    try: return np.asarray(v)
    except Exception: return str(v)
pickle.dump({str(k): conv(v) for k, v in d.items()}, open(f"{sys.argv[1]}/_all_ops_np.pkl", "wb"))
print("ops pickle converted", len(d))
EOF
# DROP_RAW_OPS=1: the raw SDK pickle (6.6 GB on yolo26m) is only read by the conversion above
[ "${DROP_RAW_OPS:-0}" = 1 ] && [ -s "$R/capture_$TAG/_all_ops_np.pkl" ] && rm -f "$R/capture_$TAG/_all_ops.pkl"
(cd hailoqat && CUDA_VISIBLE_DEVICES=$GPU env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib $P/bin/python graph26.py $R/$TAG.hn $R/capture_$TAG $I 2>&1 \
  | grep -E "checked layers exact")
CUDA_VISIBLE_DEVICES=99 $HPY hailoqat/bias_lattice.py $HAR $R/capture_$TAG 2>&1 | tail -1  # a16_w16 bias lattice (no-op for int8)
GPUS="${EVAL_GPUS:-0 1 2 3}" $R/evalgpu.sh $TAG hailo $R/$TAG.hn $R/capture_$TAG --pt ${PT:-$R/models/v14sec/best.pt} --bs 4 | tail -1
