#!/bin/bash
# usage: compile_fast.sh RECIPE Q_HAR OUT_HEF        (DFC env; CPU + one short chip benchmark per candidate)
# 1. architecture signature of Q_HAR -> compile_cache/RECIPE_SIG.auto.alls
# 2. cache hit: REPLAY the stored allocator solution (skips the placement search; same FPS)
# 3. miss / replay failure: FAST PORTFOLIO = the stock compiler (with the CP-SAT memo hook, hailo_re/turbo) at default
#    level, in parallel: default caps + resources_param at each of FASTC_UTILS. Every HEF that compiles is benchmarked on
#    the chip (hailortcli, chip lock); the fastest wins (ties -> the first in order) and its auto.alls is cached.
#    Replaces compiler_optimization_level=max, whose 120% -> 75% descent runs one ~14 min timeout after another
#    (orient3: all 19 steps fail; router: lands on the default caps). COMPILE_OPT=max restores the stock level-max compile.
# Prints COMPILE_MODE=replay|fastc|max and the wall time.
set -uo pipefail; D=$(cd "$(dirname "$0")" && pwd); source $D/env.sh; henv; export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
HOOK=$HRE/turbo/bin/libhailo_turbo.so
[ "${HAILO_FAST:-1}" != 0 ] && [ -s $HOOK ] && export LD_PRELOAD=$HOOK${LD_PRELOAD:+:$LD_PRELOAD} HAILO_CPCACHE=${HAILO_CPCACHE:-$CV/compile_cache/cpsat}
R=$1; Q=$2; HEF=$3; CACHE=$CV/compile_cache; mkdir -p $CACHE
SIG=$("$HPY" $D/arch_sig.py $Q 2>/dev/null | tail -1); A=$CACHE/${R}_${SIG}.auto.alls; echo "arch signature $SIG"
t0=$(date +%s)
if [ -s $A ]; then
  echo "== replay $A"
  if timeout ${REPLAY_TIMEOUT:-5400} "$HPY" -u $HRE/compile_hef.py $Q $HEF $A; then echo "COMPILE_MODE=replay $(( $(date +%s)-t0 )) s"; exit 0; fi
  echo "replay failed -> compile"
fi
if [ "${COMPILE_OPT:-fastc}" = max ]; then
  "$HPY" -u $HRE/compile_hef.py $Q $HEF $D/compile_max.alls || exit 1
  "$HPY" $D/extract_auto_alls.py ${HEF%.hef}_compiled.har $A && echo "cached allocator solution -> $A"
  echo "COMPILE_MODE=max $(( $(date +%s)-t0 )) s"; exit 0
fi
W=${HEF%.hef}_fastc; rm -rf $W; mkdir -p $W; V=(d); for u in ${FASTC_UTILS:-0.7 0.8 0.9}; do V+=("u$u"); done
for v in "${V[@]}"; do
  mkdir -p $W/$v; : > $W/$v/script.alls; u=${v#u}
  [ $v != d ] && echo "resources_param(max_control_utilization=$u, max_compute_utilization=$u, max_compute_16bit_utilization=$u, max_memory_utilization=$u, max_input_aligner_utilization=$u, max_apu_utilization=$u)" > $W/$v/script.alls
  ( cd $W/$v && systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-24G} nice -n 5 "$HPY" -u $HRE/compile_hef.py $Q $W/$v/model.hef $W/$v/script.alls > $W/$v/compile.log 2>&1 ) &
done
wait
best=""; bf=0
for v in "${V[@]}"; do
  [ -s $W/$v/model.hef ] || { echo "  $v: compile failed ($(grep -E '\[error\]' $W/$v/compile.log | tail -1 | cut -c1-100))"; continue; }
  f=$(flock $CHIP_LOCK hailortcli benchmark $W/$v/model.hef -t ${BENCH_S:-5} 2>&1 | tr '\r' '\n' | grep -E "hw_only" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  echo "  $v: fps ${f:-n/a}"
  if [ -z "$best" ] || awk -v a="${f:-0}" -v b="$bf" 'BEGIN{exit !(a > b * 1.005)}'; then best=$v; bf=${f:-0}; fi
done
[ -n "$best" ] || { echo "COMPILE_MODE=fastc: every candidate failed"; exit 1; }
cp $W/$best/model.hef $HEF; cp $W/$best/model_compiled.har ${HEF%.hef}_compiled.har
"$HPY" $D/extract_auto_alls.py ${HEF%.hef}_compiled.har $A && echo "cached allocator solution -> $A"
echo "COMPILE_MODE=fastc best=$best fps=$bf $(( $(date +%s)-t0 )) s"
