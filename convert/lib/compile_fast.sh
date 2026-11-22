#!/bin/bash
# usage: compile_fast.sh RECIPE Q_HAR OUT_HEF        (DFC env; CPU + one short chip benchmark per candidate)
# 1. architecture signature of Q_HAR -> compile_cache/RECIPE_SIG.auto.alls
# 2. cache hit: REPLAY the stored allocator solution (skips the placement search; same FPS)
# 3. miss / replay failure: FAST PORTFOLIO = the stock compiler (with the CP-SAT memo hook, hailo_re/turbo) at default
#    level, stock solver AND modern multi-worker CP-SAT server candidates (FASTC_SRV_UTILS / FASTC_SRV_DEFAULT=0 to drop);
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
# candidates: stock solver at default caps + FASTC_UTILS, and the MODERN multi-worker CP-SAT server (turbo/cpserver.py)
# at default caps + FASTC_SRV_UTILS. Separate memo caches per solver (a server answer must never stand in for stock).
SOCK=$CV/compile_cache/cpserver.sock
if [ -n "${FASTC_SRV_UTILS-0.8}${FASTC_SRV_DEFAULT:-1}" ] && ! "$HRE/ortools_env/bin/python" -c "import socket,sys; socket.socket(socket.AF_UNIX).connect(sys.argv[1])" $SOCK 2>/dev/null; then
  setsid nohup "$HRE/ortools_env/bin/python" $HRE/turbo/cpserver.py $SOCK --workers ${CPS_WORKERS:-8} --limit ${CPS_LIMIT:-60} >> $CV/compile_cache/cpserver.log 2>&1 &
  sleep 5
fi
W=${HEF%.hef}_fastc; rm -rf $W; mkdir -p $W; V=(d); for u in ${FASTC_UTILS:-0.7 0.8 0.9}; do V+=("u$u"); done
[ "${FASTC_SRV_DEFAULT:-1}" = 1 ] && V+=(srv_d); for u in ${FASTC_SRV_UTILS-0.8}; do V+=("srv_u$u"); done
for v in "${V[@]}"; do
  mkdir -p $W/$v; : > $W/$v/script.alls; b=${v#srv_}; u=${b#u}
  [ $b != d ] && echo "resources_param(max_control_utilization=$u, max_compute_utilization=$u, max_compute_16bit_utilization=$u, max_memory_utilization=$u, max_input_aligner_utilization=$u, max_apu_utilization=$u)" > $W/$v/script.alls
  if [ $v != $b ]; then e=(HAILO_CPSERVER=$SOCK HAILO_CPCACHE=$CV/compile_cache/cpsat_srv); else e=(HAILO_CPSERVER= HAILO_CPCACHE=$CV/compile_cache/cpsat); fi
  # private SDK home per candidate: the DFC rotates its log files in $VIRTUAL_ENV/etc/hailo/dfc and concurrent
  # compilers racing that rename ABORT ("rotating_file_sink: failed renaming" -> "unexpected crash")
  mkdir -p $W/$v/sdk_home/etc/hailo/dfc; e+=(VIRTUAL_ENV=$W/$v/sdk_home CONDA_PREFIX=$W/$v/sdk_home)
  ( cd $W/$v && env "${e[@]}" systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-24G} nice -n 5 "$HPY" -u $HRE/compile_hef.py $Q $W/$v/model.hef $W/$v/script.alls > $W/$v/compile.log 2>&1 ) &
done
wait
best=""; bf=0
for v in "${V[@]}"; do
  [ -s $W/$v/model.hef ] || { echo "  $v: compile failed ($(grep -E '\[error\]' $W/$v/compile.log | tail -1 | cut -c1-100))"; continue; }
  # known silent-wrong chip patterns G1-G6 (needs the COMPILED har: G3/G4 read the allocator's own script)
  if ! "$HPY" -u $QAT/tools/fixes/hailo8_guard.py $W/$v/model.hef $W/$v/model_compiled.har "" $Q > $W/$v/guard.log 2>&1; then
    echo "  $v: REJECTED by hailo8_guard: $(grep -E '^ *G[0-9]' $W/$v/guard.log | head -2 | tr '\n' ' ')"; continue; fi
  f=$(flock $CHIP_LOCK hailortcli benchmark $W/$v/model.hef -t ${BENCH_S:-5} 2>&1 | tr '\r' '\n' | grep -E "hw_only" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  echo "  $v: fps ${f:-n/a}"
  if [ -z "$best" ] || awk -v a="${f:-0}" -v b="$bf" 'BEGIN{exit !(a > b * 1.005)}'; then best=$v; bf=${f:-0}; fi
done
[ -n "$best" ] || { echo "COMPILE_MODE=fastc: every candidate failed"; exit 1; }
cp $W/$best/model.hef $HEF; cp $W/$best/model_compiled.har ${HEF%.hef}_compiled.har
"$HPY" $D/extract_auto_alls.py ${HEF%.hef}_compiled.har $A && echo "cached allocator solution -> $A"
echo "COMPILE_MODE=fastc best=$best fps=$bf $(( $(date +%s)-t0 )) s"
