#!/bin/bash
# run.sh CASES.txt OUT_DIR [PAR] [ALLS]: every case its own process, PAR at a time (skips cases already done)
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/probe && . ../env.sh >/dev/null 2>&1
mkdir -p $2; export HPY OUT=$2 ALLS=$4
grep -v '^#' $1 | xargs -P ${3:-24} -I{} sh -c 'd="$OUT/$(echo "{}" | sed -e "s|^@.*/||" -e "s/[^A-Za-z0-9_=,.-]/_/g")"; [ -f "$d/result.json" ] || $HPY -u sweep.py run "{}" "$OUT" $ALLS 2>/dev/null | grep "^{"' > $2/run.log
echo done $(grep -c '"ok"' $2/run.log) ok of $(wc -l < $2/run.log)
