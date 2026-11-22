#!/bin/bash
# zoo/cov_all.sh [JOBS]: cov_one.sh over nets.txt, JOBS in parallel; one line per network in cov_all.log
Z=/mnt/storage/lib_project/models/Hailo_model/hailo_re/zoo; cd $Z
run() { s=$(date +%s); out=$($Z/cov_one.sh $1 2>&1 | tail -2 | head -1); echo "$1	$(cat $Z/work/$1/status 2>/dev/null)	$(( $(date +%s) - s ))s	$out"; }
export -f run; export Z
grep -v '^$' nets.txt | xargs -P ${1:-6} -I{} bash -c 'run {}' >> cov_all.log 2>&1
echo COV_ALL_DONE >> cov_all.log
