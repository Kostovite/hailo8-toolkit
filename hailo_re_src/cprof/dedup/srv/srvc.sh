# srvc.sh RUN_DIR TAG [UTIL]: compile through the modern solve server (fresh cache), then chip check
RUN=$1; TAG=$2; U=${3:-}; H=/mnt/storage/lib_project/models/Hailo_model/hailo_re; C=/mnt/storage/lib_project/models/Hailo_model/convert
W=$RUN/fastc/$TAG; mkdir -p $W; : > $W/script.alls
[ -n "$U" ] && echo "resources_param(max_control_utilization=$U, max_compute_utilization=$U, max_compute_16bit_utilization=$U, max_memory_utilization=$U, max_input_aligner_utilization=$U, max_apu_utilization=$U)" > $W/script.alls
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh >/dev/null 2>&1
export HAILO_CPCACHE=$W/cache HAILO_CPSERVER=$C/compile_cache/cpserver.sock HAILO_CPCACHE_STATS=$W/cpcache_stats
mkdir -p $W/sdk_home/etc/hailo/dfc; export VIRTUAL_ENV=$W/sdk_home CONDA_PREFIX=$W/sdk_home  # private SDK home (DFC log-rotation race)
s=$(date +%s); cd $W && systemd-run --user --scope -q -p MemoryMax=24G $HPY -u $H/compile_hef.py $RUN/q.har $W/model.hef $W/script.alls > $W/compile.log 2>&1; dt=$(( $(date +%s)-s ))
if [ -s $W/model.hef ]; then g=$($HPY -u /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py $W/model.hef $W/model_compiled.har "" $RUN/q.har 2>&1 | tail -1); echo "$TAG util=${U:-default} compiled ${dt}s | $g | $($C/fastc/chipcheck.sh $RUN $W/model.hef $TAG)" >> $RUN/fastc/results.txt
else echo "$TAG util=${U:-default} COMPILE_FAILED ${dt}s $(grep -E '\[error\]' $W/compile.log | tail -1 | cut -c1-150)" >> $RUN/fastc/results.txt; fi
