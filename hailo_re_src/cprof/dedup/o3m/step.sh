# step.sh X : ONE level-max step of the orient3 compile at utilization X; killed when the compiler moves to the next step
X=$1; W=/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/o3m; D=$W/s$X; mkdir -p $D; cd $D
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh >/dev/null 2>&1
export LD_PRELOAD=/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/libhailo_turbo2.so${LD_PRELOAD:+:$LD_PRELOAD} HAILO_UTIL_ONLY=$X HAILO_CPCACHE=$W/cache HAILO_CPCACHE_STATS=$D/stats
s=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=8G $HPY -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py /mnt/storage/lib_project/models/Hailo_model/convert/runs/orient3_g1/q.har $D/o.hef /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/max.alls > log 2>&1 &
P=$!
r=RUNNING
while kill -0 $P 2>/dev/null; do
  if [ $(grep -c "Resources optimization params" log) -ge 2 ]; then r=FAILED_STEP; pkill -TERM -P $P; for c in $(pgrep -f "o3m/s$X/o.hef"); do kill $c 2>/dev/null; done; break; fi
  sleep 5
done
wait $P 2>/dev/null
grep -q "Successful Mapping" log && [ -s o.hef ] && r=OK
echo "$X $r wall $(( $(date +%s) - s )) $(grep -m1 'Successful Mapping' log | sed 's/.*(//;s/)//')" >> $W/results.txt
