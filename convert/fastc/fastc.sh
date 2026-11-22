#!/bin/bash
# fastc.sh RUN_DIR VARIANT [UTIL] : our fast compile of RUN_DIR/q.har = stock compiler + CP-SAT memo cache (turbo2),
# default level, optional resources_param at UTIL for every resource; then the chip check. One result line ->
# RUN_DIR/fastc/results.txt. Run several variants at once (each single-threaded in the search, RAM-capped).
RUN=$1; V=$2; U=${3:-}; C=/mnt/storage/lib_project/models/Hailo_model/convert; source $C/lib/env.sh
W=$RUN/fastc/$V; mkdir -p $W; A=$W/script.alls; : > $A
[ -n "$U" ] && echo "resources_param(max_control_utilization=$U, max_compute_utilization=$U, max_compute_16bit_utilization=$U, max_memory_utilization=$U, max_input_aligner_utilization=$U, max_apu_utilization=$U)" > $A
s=$(date +%s)
( henv; export LD_PRELOAD=/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/libhailo_turbo2.so${LD_PRELOAD:+:$LD_PRELOAD}
  export HAILO_CPCACHE=$C/compile_cache/cpsat HAILO_CPCACHE_STATS=$W/cpcache_stats
  mkdir -p $W/sdk_home/etc/hailo/dfc; export VIRTUAL_ENV=$W/sdk_home CONDA_PREFIX=$W/sdk_home  # no shared log rotation race
  cd $W && systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-24G} nice -n 5 "$HPY" -u $HRE/compile_hef.py $RUN/q.har $W/model.hef $A ) > $W/compile.log 2>&1
rc=$?; dt=$(( $(date +%s) - s ))
if [ $rc -ne 0 ] || [ ! -s $W/model.hef ]; then echo "$V util=${U:-default} COMPILE_FAILED ${dt}s $(grep -E '\[error\]' $W/compile.log | tail -1 | cut -c1-150)" >> $RUN/fastc/results.txt; exit 1; fi
g=$( (henv; "$HPY" -u $QAT/tools/fixes/hailo8_guard.py $W/model.hef $W/model_compiled.har "" $RUN/q.har) 2>&1 | tail -1)
echo "$V util=${U:-default} compiled ${dt}s | $g | $($C/fastc/chipcheck.sh $RUN $W/model.hef $V)" >> $RUN/fastc/results.txt
