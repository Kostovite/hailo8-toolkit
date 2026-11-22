# c.sh TAG [cache env...]: compile router q.har at level max
T=$1; shift
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh >/dev/null 2>&1
mkdir -p /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/rt/$T; cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/rt/$T
for kv in "$@"; do export "$kv"; done
[ -n "$HAILO_CPCACHE" ] && export LD_PRELOAD=/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/libhailo_turbo2.so${LD_PRELOAD:+:$LD_PRELOAD} HAILO_TURBO_LOG=1 HAILO_CPCACHE_STATS=$PWD/stats
s=$(date +%s.%N); $HPY -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py /mnt/storage/lib_project/models/Hailo_model/convert/runs/router_t1/q.har $PWD/r.hef /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/max.alls > log 2>&1
echo "$T wall $(echo "$(date +%s.%N) - $s" | bc) md5 $(md5sum r.hef | cut -c1-12) $(cat stats 2>/dev/null)"
