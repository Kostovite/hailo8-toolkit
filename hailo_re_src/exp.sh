#!/bin/bash
# exp.sh NAME [train_qat args...]: 4-GPU QAT run (teacher cache) then exact full-val eval of the final ints on 4 GPUs.
# Result line appended to out/eval.jsonl with tag exp_NAME; args recorded in runs/NAME/args.txt
# Env: GPUS (default "0 1 2 3"), HN / CAP (default the 16-bit-head level-0 HAR), EVAL_ALL=1 evaluates every saved ckpt
set -e
GPUS=${GPUS:-0 1 2 3}; NG=$(echo $GPUS | wc -w); CVD=$(echo $GPUS | tr ' ' ',')
NAME=$1; shift
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re
P=/mnt/storage/anaconda3/envs/general
HN=${HN:-$R/yolo26m_v14sec_L0a16.hn}; CAP=${CAP:-$R/capture_y26a16}
mkdir -p $R/runs/$NAME; echo "$@" > $R/runs/$NAME/args.txt
cd $R/hailoqat
CUDA_VISIBLE_DEVICES=$CVD env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib PATH=$P/bin:/usr/bin:/bin NCCL_P2P_DISABLE=1 \
  $P/bin/torchrun --nproc_per_node=$NG train_qat.py --out ../runs/$NAME --bs 2 --teacher_cache ../teacher_v14sec --hn $HN --cap $CAP "$@" \
  > ../runs/$NAME/stdout.log 2>&1 || { echo "EXP $NAME TRAIN FAILED"; tail -5 ../runs/$NAME/stdout.log; exit 1; }
if [ -n "$EVAL_ALL" ]; then CK=$(ls ../runs/$NAME/ints_step*.pt | sort -V); else CK=$(ls ../runs/$NAME/ints_step*.pt | sort -V | tail -1); fi
for c in $CK; do
  s=$(basename $c .pt | sed 's/ints_step//')
  A=""; [ -f ../runs/$NAME/apu_step$s.pt ] && A="--apu $R/runs/$NAME/apu_step$s.pt"
  GPUS="$GPUS" $R/evalgpu.sh exp_${NAME}_s$s hailo $HN $CAP --pt ../models/v14sec/best.pt --bs 4 --ints $c $A | sed "s/^/EXP $NAME s$s /"
done
