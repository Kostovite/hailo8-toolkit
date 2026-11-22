source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh >/dev/null 2>&1
export LD_PRELOAD=/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo/libhailo_turbo.so${LD_PRELOAD:+:$LD_PRELOAD} HAILO_GFLAGS="cp_model_dump_file=/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/orient3/m.pb" HAILO_DUMP_SEQ=/mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/orient3/m.pb
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/orient3
timeout 1500 systemd-run --user --scope -q -p MemoryMax=16G $HPY -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py /mnt/storage/lib_project/models/Hailo_model/convert/runs/orient3_g1/q.har /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/orient3/o.hef /mnt/storage/lib_project/models/Hailo_model/hailo_re/cprof/dedup/max.alls > log 2>&1
echo EXIT $? >> log
