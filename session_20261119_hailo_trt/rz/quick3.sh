#!/bin/bash
# usage: quick2.sh NAME ONNX GPU 8|16   -> parse, (gen8 alls), 64-img calib, compile
set -uo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119; Z=$D/rz; P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
V=$1; exec >$Z/$V.log 2>&1
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=$3 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=8 OPT_LEVEL=0
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
mkdir -p $Z/$V; cd $Z/$V
echo "== parse $(date +%T)"; "$HPY" -u $P/parse.py $2 $Z/${V}_parsed.har r5ord5 || { echo parse > $Z/$V.exit; exit 1; }
: > $Z/$V.alls; [ "$4" = 8 ] || [ "$4" = 8big ] && { "$HPY" -u $Z/gen8.py $Z/${V}_parsed.har $Z/$V.alls $([ "$4" = 8big ] && echo _encoder_Resize_2) || { echo gen8 > $Z/$V.exit; exit 1; }; }
echo "== quant $(date +%T)"; "$HPY" -u $Z/quant_mix16.py $Z/${V}_parsed.har $Z/${V}_q64.har $Z/calib64.npy mix16 $Z/$V.alls || { echo quant > $Z/$V.exit; exit 1; }
echo "== compile $(date +%T)"; "$HPY" -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py $Z/${V}_q64.har $Z/${V}_q64.hef || { echo compile > $Z/$V.exit; exit 1; }
echo 0 > $Z/$V.exit; echo "== done $(date +%T)"
