#!/bin/sh
# matrix3/run.sh PREC CASE: one probe (skips cases already done)
[ -s matrix3/$1/$2/result.json ] && exit 0
PREC=$1 timeout 1800 $HPY probe.py run $2 matrix3/$1 > /dev/null 2>&1
rm -f matrix3/$1/$2/q.har
