#!/bin/bash
# stock-compile each @DIR case (sweep.py), then emulator refs (CPU), then chip vs emulator. $1 cases, $2 out dir
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
P=/mnt/storage/lib_project/models/Hailo_model/hailo_re/probe
mkdir -p $2
for c in $(cat $1); do (cd $P && $HPY -u sweep.py run $c $2/$(basename $c) > $2/$(basename $c).log 2>&1) & done; wait
for c in $(cat $1); do n=$(basename $c); [ -f $2/$n/$n/m.hef ] && echo $2/$n/$n; done > $2/dirs.txt
$HPY -u $P/silent/chip_vs_emu.py emu 12 $2/dirs.txt > $2/emu.log 2>&1
DEVICE=${DEVICE:-0000:03:00.0} $HPY -u $P/silent/chip_vs_emu.py chip $2/res.jsonl $2/dirs.txt 2>&1 | grep -v Warn
for c in $(cat $1); do n=$(basename $c); [ -f $2/$n/$n/m.hef ] || echo "NOHEF $n $(grep -o '"error": "[^"]*' $2/$n/$n/result.json 2>/dev/null | cut -c1-150)"; done
