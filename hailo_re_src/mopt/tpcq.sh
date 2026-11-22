#!/bin/bash
# mopt/tpcq.sh: YOLO26m fast mode = channel-split graph (mTPC) at 80% resource caps (4 contexts), exact QAT.
#   1. mTPC_L0a16.har (level 0 + 16-bit heads) -> har_pipeline (capture, exactness, val eval) -> bench L0 eval
#   2. QAT (q3a recipe, 2 epochs, ckpt every 1150 steps) on GPUS -> bench eval of every ckpt -> pick on bench_sel
#   3. ints2npz -> writeback -> round-trip har_pipeline + bench eval
#   4. compile replaying the p_res80 allocator solution (part/tpc80_replay.alls) -> hef_dump
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; M=$R/mopt; P=/mnt/storage/anaconda3/envs/general
GPUS=${GPUS:-0 1}; G0=${GPUS%% *}; N=mTPC
PY="env -u LD_PRELOAD PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=$P/lib $P/bin/python"
cd $R; . env.sh
export VAL_BENCH=$R/data_v14sec/bench/images

if [ ! -d $R/capture_$N ]; then
  EVAL_GPUS="$GPUS" ./har_pipeline.sh $M/${N}_L0a16.har $N $G0 2>&1 | grep -E "captured|exact|tag" | sed "s/^/TPCQ L0 /"
fi
grep -q "\"mb_${N}_L0\"" out/eval.jsonl || VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${N}_L0 hailo $R/$N.hn $R/capture_$N \
  --pt $R/models/v14sec/best.pt --bs 4 | tail -1 | sed "s/^/TPCQ bench L0 /"

RUN=mTPC_2ep
if [ ! -f runs/$RUN/ints_step9200.pt ]; then
  HN=$R/$N.hn CAP=$R/capture_$N GPUS="$GPUS" PORT=29533 ./exp2.sh $RUN --epochs 2 --eval_every 1150 --lr_k 6e-4 --lr_b 3e-2 \
    --ovf 0 --k_radius 4 --b_radius 256 | sed "s/^/TPCQ /"
fi
TAGS="mb_${N}_L0"
for c in $(ls runs/$RUN/ints_step*.pt | sort -V); do
  s=$(basename $c .pt | sed 's/ints_step//')
  grep -q "\"mb_${RUN}_s$s\"" out/eval.jsonl || VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${RUN}_s$s hailo $R/$N.hn $R/capture_$N \
    --pt $R/models/v14sec/best.pt --bs 4 --ints $R/$c | tail -1 | cut -c1-150 | sed "s/^/TPCQ bench /"
  TAGS="$TAGS mb_${RUN}_s$s"
done
(cd hailoqat && VAL_DIR=$VAL_BENCH $PY split_map.py bench_fp bench_q3a_s2000_har $TAGS 2>&1 | tail -20) | tee $M/${N}_split.txt | sed "s/^/TPCQ split /"
BEST=$(grep "^mb_${RUN}_s" $M/${N}_split.txt | awk '{split($3,a,"/"); print a[2], $1}' | sort -gr | head -1 | awk '{print $2}' | sed 's/.*_s//')
echo "TPCQ best on bench_sel: step $BEST"

cd hailoqat
$PY ints2npz.py ../runs/$RUN/ints_step$BEST.pt ../capture_$N $M/${N}_s${BEST}_ints.npz
CUDA_VISIBLE_DEVICES=99 $HPY writeback.py $M/${N}_L0a16.har $M/${N}_s${BEST}_ints.npz $M/${N}_qat.har 2>&1 | grep -vE "^\[info\]" | tail -4 | sed "s/^/TPCQ wb /"
cd $R
EVAL_GPUS="$GPUS" ./har_pipeline.sh $M/${N}_qat.har ${N}_qat $G0 2>&1 | grep -E "exact|tag" | sed "s/^/TPCQ rt val /"
VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${N}_qat_rt hailo $R/${N}_qat.hn $R/capture_${N}_qat --pt $R/models/v14sec/best.pt --bs 4 | tail -1 | sed "s/^/TPCQ rt bench /"
(cd hailoqat && VAL_DIR=$VAL_BENCH $PY split_map.py bench_fp bench_q3a_s2000_har mb_${N}_qat_rt 2>&1 | tail -3) | sed "s/^/TPCQ rt split /"
CUDA_VISIBLE_DEVICES=99 $HPY compile_hef.py $M/${N}_qat.har $M/${N}_qat80.hef $M/part/tpc80_replay.alls > $M/${N}_qat80_compile.log 2>&1
$P/bin/python hefre/hef_dump.py $M/${N}_qat80.hef --json $M/${N}_qat80_hef.json | head -12 | sed "s/^/TPCQ hef /"
echo "TPCQ DONE"
