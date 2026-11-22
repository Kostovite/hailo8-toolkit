# one.sh VARIANT: default-level orient3 compile with a fixed resources_param + CP-SAT cache
V=$1; W=/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/o3p
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh >/dev/null 2>&1
export LD_PRELOAD=/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/libhailo_turbo2.so${LD_PRELOAD:+:$LD_PRELOAD} HAILO_CPCACHE=$W/cache HAILO_CPCACHE_STATS=$W/$V/stats
mkdir -p $W/$V; cd $W/$V; s=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=16G $HPY -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py /mnt/storage/lib_project/models/Hailo_model/convert/runs/orient3_g1/q.har $W/$V/o.hef $W/$V.alls > log 2>&1
echo "$V EXIT $? wall $(( $(date +%s) - s ))" >> $W/results.txt
