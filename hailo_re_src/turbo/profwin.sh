#!/bin/bash
# turbo/profwin.sh HAR DELAY_S NSAMPLES TAG: compile HAR, after DELAY_S take NSAMPLES gdb stacks of the compiler's busiest
# threads -> TAG.stacks (raw) ; prints per-thread CPU at sampling time
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; cd $R && . ./env.sh >/dev/null 2>&1; cd $R/turbo
HAR=$1; D=$2; N=$3; TAG=$4; G=$R/tools_env/bin/gdb
$HPY -u t/compile_ts.py $HAR > t/$TAG.log 2>&1 &
PY=$!
sleep $D
P=$(pgrep -P $PY -f "build/compiler" | head -1)  # this run's compiler (PIDs wrap: never pick "the newest")
echo "compiler pid $P"; grep -m1 "build/compiler" /proc/$P/maps | cut -d- -f1 > t/$TAG.base
for i in $(seq 1 $N); do
  # busiest threads now: utime+stime over 1 s
  T=$(for t in /proc/$P/task/*; do echo "$(awk '{print $14+$15}' $t/stat 2>/dev/null) ${t##*/}"; done | sort -n > /tmp/a_$$; sleep 1; for t in /proc/$P/task/*; do echo "$(awk '{print $14+$15}' $t/stat 2>/dev/null) ${t##*/}"; done | sort -n > /tmp/b_$$; join -1 2 -2 2 <(sort -k2 /tmp/a_$$) <(sort -k2 /tmp/b_$$) | awk '{print $3-$2, $1}' | sort -rn | head -3 | awk '$1>20{print $2}')
  echo "busy threads: $T" >> t/$TAG.stacks
  for tid in $T; do $G -batch -nx -p $P -ex "set pagination off" -ex "thread find $tid" -ex "info threads" 2>/dev/null | grep -E "^\*? +[0-9]+ +Thread .*LWP $tid" | awk '{print ($1=="*")?$2:$1}' | head -1 > /tmp/tn_$$; TN=$(cat /tmp/tn_$$); [ -n "$TN" ] && $G -batch -nx -p $P -ex "set pagination off" -ex "thread $TN" -ex "bt 40" 2>/dev/null | grep "^#" >> t/$TAG.stacks; echo "---- $tid" >> t/$TAG.stacks; done
  sleep 2
done
wait
rm -f /tmp/a_$$ /tmp/b_$$ /tmp/tn_$$
