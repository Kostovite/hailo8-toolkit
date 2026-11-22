#!/bin/bash
# usage: compile_fast.sh RECIPE Q_HAR OUT_HEF        (DFC env; CPU)
# 1. architecture signature of Q_HAR -> compile_cache/RECIPE_SIG.auto.alls
# 2. cache hit: compile by REPLAYING the previous allocator solution (skips the placement search; same FPS)
#    cache miss or replay failure: full compile (COMPILE_OPT=max default), then store its auto.alls in the cache
# Prints COMPILE_MODE=replay|full and the wall time.
set -uo pipefail; D=$(cd "$(dirname "$0")" && pwd); source $D/env.sh; henv; export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
R=$1; Q=$2; HEF=$3; CACHE=$CV/compile_cache; mkdir -p $CACHE
SIG=$("$HPY" $D/arch_sig.py $Q 2>/dev/null | tail -1); A=$CACHE/${R}_${SIG}.auto.alls; echo "arch signature $SIG"
t0=$(date +%s)
if [ -s $A ]; then
  echo "== replay $A"
  if timeout ${REPLAY_TIMEOUT:-5400} "$HPY" -u $HRE/compile_hef.py $Q $HEF $A; then echo "COMPILE_MODE=replay $(( $(date +%s)-t0 )) s"; exit 0; fi
  echo "replay failed -> full compile"
fi
x=(); [ "${COMPILE_OPT:-max}" = max ] && x=($D/compile_max.alls)
"$HPY" -u $HRE/compile_hef.py $Q $HEF "${x[@]}" || exit 1
"$HPY" $D/extract_auto_alls.py ${HEF%.hef}_compiled.har $A && echo "cached allocator solution -> $A"
echo "COMPILE_MODE=full $(( $(date +%s)-t0 )) s"
