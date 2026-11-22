#!/bin/bash
# mopt/w16m.sh: YOLO26m accuracy mode = 16-bit heads (a8_w8_a16) + top-8 sensitivity layers in a16_w16, exact QAT.
#   1. pick 8 conv/dw layers from sens_m_L0.json (skip heads and tensors that cross q3a context cuts)
#   2. DFC level 0 -> har_pipeline (capture, exactness, val eval) -> bias_lattice
#   3. QAT (q3a recipe, 2 epochs, ckpt every 1150 steps) on GPUS -> bench eval of every ckpt -> pick on bench_sel
#   4. ints2npz -> writeback -> round-trip har_pipeline + bench eval -> compile -> hef_dump
set -e
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; M=$R/mopt; P=/mnt/storage/anaconda3/envs/general
GPUS=${GPUS:-0 1}; G0=${GPUS%% *}; N=m_w16top8
PY="env -u LD_PRELOAD PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=$P/lib $P/bin/python"
cd $R; . env.sh
export VAL_BENCH=$R/data_v14sec/bench/images

$PY - <<PY
import json
s = json.load(open("$M/sens_m_L0.json"))
cut = {"conv21", "conv22", "conv31", "dw1", "dw6", "conv49", "conv83", "conv84", "conv85", "conv98",
       "concat12", "conv_feature_splitter7_1", "conv_feature_splitter7_2"}
pick = []
for r in s["rows"]:
    n = r["layer"].split("/")[-1]
    if r["a16"] or r["type"] not in ("conv", "dw") or n in cut:
        continue
    pick.append(n)
    if len(pick) == 8:
        break
tot = s["total_int8"]
share = sum(r["score"] for r in s["rows"] if r["layer"].split("/")[-1] in pick) / tot
open("$M/alls_${N}.txt", "w").write(
    "quantization_param([conv71, conv74, conv87, conv90, conv101, conv104], precision_mode=a8_w8_a16)\n"
    f"quantization_param([{', '.join(pick)}], precision_mode=a16_w16)\n")
print("W16M pick", pick, f"= {share:.1%} of the int8 rounding-noise score")
PY

if [ ! -f $M/${N}.har ]; then
  CUDA_VISIBLE_DEVICES=$G0 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH=$R/dfc_patched TRAIN_GLOB="$R/data_v14sec/train/images/*" NMS="" \
    OPT_LEVEL=0 $HPY b2_levelN.py $R/../yolo26m_v14sec_parsed.har $M/${N}.har 896 896 1024 $M/alls_${N}.txt > $M/${N}_quant.log 2>&1
fi
echo "W16M quantized"
EVAL_GPUS="$GPUS" ./har_pipeline.sh $M/${N}.har $N $G0 2>&1 | grep -E "captured|exact|tag" | sed "s/^/W16M L0 /"
CUDA_VISIBLE_DEVICES=99 $HPY hailoqat/bias_lattice.py $M/${N}.har $R/capture_$N 2>&1 | grep -vE "^\[info\]" | tail -2 | sed "s/^/W16M /"
VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${N}_L0 hailo $R/$N.hn $R/capture_$N --pt $R/models/v14sec/best.pt --bs 4 | tail -1 | sed "s/^/W16M bench L0 /"

RUN=m_w16L_2ep
if [ ! -f runs/$RUN/ints_step9200.pt ]; then
  HN=$R/$N.hn CAP=$R/capture_$N GPUS="$GPUS" PORT=29531 ./exp2.sh $RUN --epochs 2 --eval_every 1150 --lr_k 6e-4 --lr_b 3e-2 \
    --ovf 0 --k_radius 4 --b_radius 256 | sed "s/^/W16M /"
fi
TAGS="mb_${N}_L0"
for c in $(ls runs/$RUN/ints_step*.pt | sort -V); do
  s=$(basename $c .pt | sed 's/ints_step//')
  grep -q "\"mb_${RUN}_s$s\"" out/eval.jsonl || VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${RUN}_s$s hailo $R/$N.hn $R/capture_$N \
    --pt $R/models/v14sec/best.pt --bs 4 --ints $R/$c | tail -1 | cut -c1-150 | sed "s/^/W16M bench /"
  TAGS="$TAGS mb_${RUN}_s$s"
done
(cd hailoqat && VAL_DIR=$VAL_BENCH $PY split_map.py bench_fp bench_q3a_s2000_har $TAGS 2>&1 | tail -20) | tee $M/${N}_split.txt | sed "s/^/W16M split /"
BEST=$(grep "^mb_${RUN}_s" $M/${N}_split.txt | awk '{split($3,a,"/"); print a[2], $1}' | sort -gr | head -1 | awk '{print $2}' | sed 's/.*_s//')
echo "W16M best on bench_sel: step $BEST"

cd hailoqat
$PY ints2npz.py ../runs/$RUN/ints_step$BEST.pt ../capture_$N $M/${N}_s${BEST}_ints.npz
CUDA_VISIBLE_DEVICES=99 $HPY writeback.py $M/${N}.har $M/${N}_s${BEST}_ints.npz $M/${N}_qat.har 2>&1 | grep -vE "^\[info\]" | tail -4 | sed "s/^/W16M wb /"
cd $R
EVAL_GPUS="$GPUS" ./har_pipeline.sh $M/${N}_qat.har ${N}_qat $G0 2>&1 | grep -E "exact|tag" | sed "s/^/W16M rt val /"
VAL_DIR=$VAL_BENCH GPUS="$GPUS" ./evalgpu.sh mb_${N}_qat_rt hailo $R/${N}_qat.hn $R/capture_${N}_qat --pt $R/models/v14sec/best.pt --bs 4 | tail -1 | sed "s/^/W16M rt bench /"
(cd hailoqat && VAL_DIR=$VAL_BENCH $PY split_map.py bench_fp bench_q3a_s2000_har mb_${N}_qat_rt 2>&1 | tail -3) | sed "s/^/W16M rt split /"
CUDA_VISIBLE_DEVICES=99 $HPY compile_hef.py $M/${N}_qat.har $M/${N}_qat.hef > $M/${N}_compile.log 2>&1
$P/bin/python hefre/hef_dump.py $M/${N}_qat.hef --json $M/${N}_qat_hef.json | head -12 | sed "s/^/W16M hef /"
echo "W16M DONE"
