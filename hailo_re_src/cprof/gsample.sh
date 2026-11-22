#!/bin/bash
# gsample.sh PID N OUT: N gdb snapshots of the thread that is busy NOW (solve threads are short-lived), as offsets into
# the compiler binary; name them with cprof/fnmap.py. Works where perf is blocked (perf_event_paranoid=3).
P=$1; N=$2; OUT=$3; GDB=/mnt/storage/lib_project/models/Hailo_model/hailo_re/tools_env/bin/gdb
BASE=$(awk '/build\/compiler/{split($1,a,"-"); print a[1]; exit}' /proc/$P/maps)
: > $OUT
snap(){ for t in /proc/$P/task/*; do s=$(cat $t/stat 2>/dev/null) || continue; echo "${t##*/} $(echo "$s" | sed 's/.*) //' | awk '{print $12+$13}')"; done; }
for i in $(seq $N); do
  snap > a.$P; sleep 0.5; snap > b.$P
  T=$(join <(sort a.$P) <(sort b.$P) | awk '{print $3-$2, $1}' | sort -rn | head -1 | awk '{print $2}')
  $GDB -p $P -batch -nx -ex "set pagination off" -ex "thread apply all bt 80" 2>/dev/null \
   | awk -v lwp="LWP $T)" -v base=$((16#$BASE)) '/^Thread /{on=index($0,lwp)>0} on&&/^#/{a=$2; if(a ~ /^0x/){printf "%x ", strtonum(a)-base}else{printf "%s ", $4}} END{print ""}' >> $OUT
  sleep 1
done
