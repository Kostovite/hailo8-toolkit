#!/bin/bash
# cprof/run_compile.sh TAG HAR [ALLS]: fresh compile with timestamped log + thread sampler (CPU only)
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; cd $R && . ./env.sh >/dev/null 2>&1; cd $R/cprof
TAG=$1; HAR=$2; ALLS=${3:-}
( $HPY -u $R/compile_hef.py $HAR $R/cprof/$TAG.hef $ALLS 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s)" "$l"; done > $TAG.log ) &
sleep 2; P=$(pgrep -n -f "compile_hef.py $HAR $R/cprof/$TAG.hef" | head -1)
echo "pid $P"; $HPY sample_threads.py $P $TAG.threads.tsv 5 &
wait
