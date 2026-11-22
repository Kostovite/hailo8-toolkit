#!/bin/bash
# usage: run_c.sh NAME  (NAME.har -> NAME.hef, then guard + 5 s chip benchmark)
cd /mnt/storage/lib_project/models/Hailo_model/preproc_hailo
. /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh; export CUDA_VISIBLE_DEVICES=99
$HPY /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py $1.har $1.hef > logs/compile_$1.log 2>&1; echo EXIT $? >> logs/compile_$1.log
[ -f $1.hef ] && $HPY /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py $1.hef $1.har > logs/guard_$1.log 2>&1
[ -f $1.hef ] && hailortcli benchmark $1.hef -t 5 > logs/bench_$1.log 2>&1
echo DONE >> logs/compile_$1.log
