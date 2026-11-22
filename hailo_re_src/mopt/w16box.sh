#!/bin/bash
# mopt/w16box.sh: w16m.sh + the whole box path in a16_w16 (box convs, box output convs, neck outputs feeding them;
# 2026-10-26: the FP -> chip gap is box precision at IoU 0.90/0.95). Same recipe; only the precision set differs.
#   1. pick 8 conv/dw layers from sens_m_L0.json (skip heads and tensors that cross q3a context cuts)
#   2. DFC level 0 -> har_pipeline (capture, exactness, val eval) -> bias_lattice
#   3. QAT (q3a recipe, 2 epochs, ckpt every 1150 steps) on GPUS -> bench eval of every ckpt -> pick on bench_sel
#   4. ints2npz -> writeback -> round-trip har_pipeline + bench eval -> compile -> hef_dump
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; M=$R/mopt; P=/mnt/storage/anaconda3/envs/general
GPUS=${GPUS:-0 1}; G0=${GPUS%% *}; N=m_w16box
PY="env -u LD_PRELOAD PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=$P/lib $P/bin/python"
cd $R; . env.sh
export DROP_RAW_OPS=1  # /mnt/storage is full: keep only the converted op pickle
export VAL_BENCH=$R/data_v14sec/bench/images

cat > $M/alls_${N}.txt <<ALLS
quantization_param([conv74, conv90, conv104], precision_mode=a8_w8_a16)
quantization_param([conv70, conv86, conv100, conv97, conv69, conv_feature_splitter11_2, conv67, conv60, conv85, conv99, conv71, conv87, conv101, conv83, conv98], precision_mode=a16_w16)
ALLS
echo "W16BOX alls: \$(cat $M/alls_${N}.txt | tr '\n' ' ')"

if [ ! -f $M/${N}.har ]; then
  CUDA_VISIBLE_DEVICES=$G0 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH=$R/dfc_patched TRAIN_GLOB="$R/data_v14sec/train/images/*" NMS="" \
    OPT_LEVEL=0 $HPY b2_levelN.py $R/../yolo26m_v14sec_parsed.har $M/${N}.har 896 896 1024 $M/alls_${N}.txt > $M/${N}_quant.log 2>&1
fi
echo "W16BOX quantized"
EVAL_GPUS="$GPUS" ./har_pipeline.sh $M/${N}.har $N $G0 2>&1 | grep -E "captured|exact|tag" | sed "s/^/W16BOX L0 /"
CUDA_VISIBLE_DEVICES=99 $HPY hailoqat/bias_lattice.py $M/${N}.har $R/capture_$N 2>&1 | grep -vE "^\[info\]" | tail -2 | sed "s/^/W16BOX /"
VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${N}_L0 hailo $R/$N.hn $R/capture_$N --pt $R/models/v14sec/best.pt --bs 4 | tail -1 | sed "s/^/W16BOX bench L0 /"

RUN=m_w16box_2ep
if [ ! -f runs/$RUN/ints_step9200.pt ]; then
  HN=$R/$N.hn CAP=$R/capture_$N GPUS="$GPUS" PORT=29533 ./exp2.sh $RUN --epochs 2 --eval_every 1150 --lr_k 6e-4 --lr_b 3e-2 \
    --ovf 0 --k_radius 4 --b_radius 256 | sed "s/^/W16BOX /"
fi
TAGS="mb_${N}_L0"
for c in $(ls runs/$RUN/ints_step*.pt | sort -V); do
  s=$(basename $c .pt | sed 's/ints_step//')
  grep -q "\"mb_${RUN}_s$s\"" out/eval.jsonl || VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${RUN}_s$s hailo $R/$N.hn $R/capture_$N \
    --pt $R/models/v14sec/best.pt --bs 4 --ints $R/$c | tail -1 | cut -c1-150 | sed "s/^/W16BOX bench /"
  TAGS="$TAGS mb_${RUN}_s$s"
done
(cd hailoqat && VAL_DIR=$VAL_BENCH $PY split_map.py bench_fp bench_q3a_s2000_har $TAGS 2>&1 | tail -20) | tee $M/${N}_split.txt | sed "s/^/W16BOX split /"
BEST=$(grep "^mb_${RUN}_s" $M/${N}_split.txt | awk '{split($3,a,"/"); print a[2], $1}' | sort -gr | head -1 | awk '{print $2}' | sed 's/.*_s//')
echo "W16BOX best on bench_sel: step $BEST"

cd hailoqat
$PY ints2npz.py ../runs/$RUN/ints_step$BEST.pt ../capture_$N $M/${N}_s${BEST}_ints.npz
CUDA_VISIBLE_DEVICES=99 $HPY writeback.py $M/${N}.har $M/${N}_s${BEST}_ints.npz $M/${N}_qat.har 2>&1 | grep -vE "^\[info\]" | tail -4 | sed "s/^/W16BOX wb /"
cd $R
# the level-0 capture is only needed for QAT and the checkpoint evals, both done: free its ~10 GB first
[ -s "$M/${N}_qat.har" ] && [ -d "$R/capture_${N}" ] && rm -r -- "$R/capture_${N}"
EVAL_GPUS="$GPUS" ./har_pipeline.sh $M/${N}_qat.har ${N}_qat $G0 2>&1 | grep -E "exact|tag" | sed "s/^/W16BOX rt val /"
VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${N}_qat_rt hailo $R/${N}_qat.hn $R/capture_${N}_qat --pt $R/models/v14sec/best.pt --bs 4 | tail -1 | sed "s/^/W16BOX rt bench /"
(cd hailoqat && VAL_DIR=$VAL_BENCH $PY split_map.py bench_fp bench_q3a_s2000_har mb_${N}_qat_rt 2>&1 | tail -3) | sed "s/^/W16BOX rt split /"
CUDA_VISIBLE_DEVICES=99 $HPY compile_hef.py $M/${N}_qat.har $M/${N}_qat.hef > $M/${N}_compile.log 2>&1
$P/bin/python hefre/hef_dump.py $M/${N}_qat.hef --json $M/${N}_qat_hef.json | head -12 | sed "s/^/W16BOX hef /"
echo "W16BOX DONE"
