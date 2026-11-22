#!/bin/bash
# port/run1.sh TAG ALLS_TEXT HAR: one portfolio member, CPU only, timestamped log
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; cd $R && . ./env.sh >/dev/null 2>&1; cd $R/cprof/port
printf "$2\n" > $1.alls
( time $HPY -u $R/compile_hef.py $3 $1.hef $1.alls ) > $1.log 2>&1
echo "$1 done"
