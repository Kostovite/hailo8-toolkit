source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh >/dev/null 2>&1
export LD_PRELOAD=/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/libhailo_turbo2.so${LD_PRELOAD:+:$LD_PRELOAD} HAILO_TURBO_LOG=1 HAILO_CPCACHE=/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/cache_o3 HAILO_CPCACHE_STATS=/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/o3c/stats
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/o3c; s=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=16G $HPY -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py /mnt/storage/lib_project/models/Hailo_model/convert/runs/orient3_g1/q.har /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/o3c/o.hef /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/max.alls > log 2>&1
echo "EXIT $? wall $(($(date +%s)-s))" >> log
