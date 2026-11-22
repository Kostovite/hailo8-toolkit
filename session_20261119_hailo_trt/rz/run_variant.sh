#!/bin/bash
# usage: run_variant.sh NAME GPU OPT_LEVEL
set -uo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119; P=/mnt/storage/lib_project/models/Hailo_model/preproc_hailo
V=$1; G=$2; O=$3; Z=$D/rz; LOG=$Z/$V.log; exec >"$LOG" 2>&1
st(){ echo "== $1 $(date +%T)"; }
fail(){ echo "STAGE_FAIL $1"; echo "$1" > $Z/$V.exit; exit 1; }
(
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=$G TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=8 OPT_LEVEL=$O
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
mkdir -p $Z/$V && cd $Z/$V
st quant; "$HPY" -u $Z/quant_mix16.py $D/r5_nofuse_parsed.har $Z/$V.har $P/calib/layout_1024.npy mix16 $Z/$V.alls || fail quant
st compile; "$HPY" -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py $Z/$V.har $Z/$V.hef || echo COMPILE_FAILED_CONTINUE_EMU
[ -f $Z/$V.hef ] && { st guard; "$HPY" -u /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py $Z/$V.hef $Z/$V.har || echo GUARD_FAILED; }
st emu; "$HPY" -u $P/emu.py $Z/$V.har $P/eval/layout_x.npy $Z/layout_$V.npy q || fail emu
) || exit 1
st score
( unset LD_PRELOAD LD_LIBRARY_PATH; export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 PYTHONNOUSERSITE=1
/mnt/storage/lib_project/models/layout_teacher/venv_hf/bin/python -u $P/layout_gate.py score /mnt/storage/lib_project/models/layout_teacher/staged_v2_20261114/r5_ord5.pt $Z/layout_$V ) || fail score
[ -f $Z/$V.hef ] && echo 0 > $Z/$V.exit || echo nohef > $Z/$V.exit
st done
