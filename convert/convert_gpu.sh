#!/bin/bash
# GPU variant of convert.sh: quantized-emulator gate on the GPU exact executor (lib/gpu_emu.sh), FP parse check on
# FP_ROWS (default 32) rows only, Hailo compile serialized by a lock (the only long CPU step), compiled at
# compiler_optimization_level=max for the fastest chip binary (COMPILE_OPT=default for a quicker stock compile).
# usage: convert_gpu.sh RECIPE CKPT OUTDIR [GPUS=0,1,2,3] [FROM_STAGE]
# Stages (each logged to OUTDIR/logs/<stage>.log, skipped when OUTDIR/.done_<stage> exists):
#   export   torch ckpt -> model.onnx (+ host contract files in OUTDIR/host/), FP parity printed by the exporter
#   prep     ONNX rewrites for Hailo (compat front end / recipe rewrites), float parity checked
#   data     calib.npy (training-side data only) + eval_x.npy and torch reference ref_<i>.npy
#   parse    model.onnx -> parsed.har
#   quant    parsed.har -> q.har (recipe precision plan)
#   emu      emulator FP + quantized on eval_x, sharded over GPUS
#   gate_fp  emulator FP == torch reference (catches parse/rewrite errors)
#   gate_q   recipe task metric on the quantized emulator  (FAIL stops here)
#   compile  q.har -> model.hef, then hailo8_guard
#   chip     real Hailo-8 on eval_x (chip lock held)
#   gate_chip recipe task metric on the chip output
set -uo pipefail
CVD=$(cd "$(dirname "$0")" && pwd); source $CVD/lib/env.sh
R=$1; CKPT=$2; OUT=$(mkdir -p "$3" && cd "$3" && pwd); GPUS=${4:-0,1,2,3}; FROM=${5:-}
source $CVD/recipes/$R.sh; mkdir -p $OUT/logs $OUT/host
QMODE=${QMODE:-all16}; NORM=${NORM:-none}; OPT=${OPT:-0}; INPUT16=${INPUT16:-0}; CHIP_FMT=${CHIP_FMT:-uint8}
G0=${GPUS%%,*}
STAGES="export prep data parse quant emu gate_fp gate_q compile chip gate_chip"
[ -n "$FROM" ] && { on=0; for s in $STAGES; do [ $s = $FROM ] && on=1; [ $on = 1 ] && rm -f $OUT/.done_$s; done; }
echo "recipe=$R ckpt=$CKPT out=$OUT gpus=$GPUS qmode=$QMODE norm=$NORM opt=$OPT input16=$INPUT16" | tee -a $OUT/logs/run.log
run(){ local s=$1; shift
  [ -f $OUT/.done_$s ] && { echo "skip $s"; return 0; }
  echo "== $s $(date '+%F %T')" | tee -a $OUT/logs/run.log
  ( "$@" ) > $OUT/logs/$s.log 2>&1; local rc=$?
  echo "   $s exit $rc" | tee -a $OUT/logs/run.log; tail -n 4 $OUT/logs/$s.log | sed 's/^/   | /' | cut -c1-200
  [ $rc = 0 ] && touch $OUT/.done_$s || { echo "STOP at $s (log $OUT/logs/$s.log)" | tee -a $OUT/logs/run.log; exit $rc; }; }
s_parse(){ henv; export CUDA_VISIBLE_DEVICES=99; "$HPY" -u $PP/parse.py $OUT/model.onnx $OUT/parsed.har $R; }
s_quant(){ if declare -F r_alls >/dev/null; then ( r_alls ) || return 1; fi; henv; export CUDA_VISIBLE_DEVICES=$G0 INPUT16; x=(); [ -n "${EXTRA_ALLS:-}" ] && x=("$EXTRA_ALLS")
           "$HPY" -u $CVD/lib/quantize.py $OUT/parsed.har $OUT/q.har $OUT/calib.npy $QMODE $NORM $OPT "${x[@]}"; }
s_emu(){ EMU_ROWS=${FP_ROWS:-32} bash $CVD/lib/emu_shard.sh $OUT/q.har $OUT/eval_x.npy $OUT/emu_fp fp $G0 && bash $CVD/lib/gpu_emu.sh $OUT/q.har $OUT/eval_x.npy $OUT/emu_q $G0; }
s_gate_fp(){ python3 $CVD/lib/compare.py $OUT/ref $OUT/emu_fp ${FP_TOL:-1e-3}; }
s_compile(){ henv; export OMP_NUM_THREADS=16; flock $CVD/runs/.compile.lock bash $CVD/lib/compile_fast.sh $R $OUT/q.har $OUT/model.hef && "$HPY" -u $QAT/tools/fixes/hailo8_guard.py $OUT/model.hef $OUT/q.har; }   # replay cache, see lib/compile_fast.sh
s_chip(){ henv; flock $CHIP_LOCK "$HPY" -u $CVD/lib/chip.py $OUT/model.hef $OUT/eval_x.npy $OUT/chip $CHIP_FMT; }
run export  r_export
run prep    r_prep
run data    r_data
run parse   s_parse
run quant   s_quant
run emu     s_emu
run gate_fp s_gate_fp
run gate_q  r_gate emu_q
run compile s_compile
run chip    s_chip
run gate_chip r_gate chip
echo "ALL GATES PASSED: $OUT/model.hef   host contract: $OUT/host/" | tee -a $OUT/logs/run.log
