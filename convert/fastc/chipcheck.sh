#!/bin/bash
# chipcheck.sh RUN_DIR HEF TAG : chip outputs on the eval set + the recipe's chip gate + hailortcli FPS (one line out)
RUN=$1; HEF=$2; TAG=$3; C=/mnt/storage/lib_project/models/Hailo_model/convert; source $C/lib/env.sh
R=$(grep -m1 -o "recipe=[a-z0-9]*" $RUN/logs/run.log | cut -d= -f2); CKPT=$(grep -m1 -o "ckpt=[^ ]*" $RUN/logs/run.log | cut -d= -f2)
OUT=$RUN; G0=${G0:-0}; source $C/recipes/$R.sh; CHIP_FMT=${CHIP_FMT:-uint8}
mkdir -p $RUN/fastc; L=$RUN/fastc/$TAG.log
( henv; flock $CHIP_LOCK "$HPY" -u $C/lib/chip.py $HEF $RUN/eval_x.npy $RUN/chip_$TAG $CHIP_FMT ) > $L 2>&1 || { echo "$TAG CHIP_RUN_FAILED $(tail -1 $L)"; exit 1; }
( r_gate chip_$TAG ) >> $L 2>&1
FPS=$( (henv; flock $CHIP_LOCK hailortcli benchmark $HEF -t 5) 2>&1 | tr '\r' '\n' | grep -iE "fps.*=|hw_only|streaming" | tail -3 | tr '\n' ' ')
echo "$TAG $(grep -E "^RESULT|^GATE" $L | tr '\n' ' ') | $FPS"
