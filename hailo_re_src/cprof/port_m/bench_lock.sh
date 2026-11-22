#!/bin/bash
# bench_lock.sh HEF...: b1 and b4 hw FPS of each HEF, two at a time (one per chip), holding the hcc chip lock
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re
exec 9>turbo/.chips.lock; flock 9
fps() { hailortcli benchmark "$1" -t 5 --batch-size $3 --device-id $2 2>&1 | awk -F= '/hw_only/{print $2+0; exit}'; }
set -- "$@"
while [ $# -gt 0 ]; do
  a=$1; b=$2; shift; [ $# -gt 0 ] && shift
  for B in 1 4; do
    fps $a 0000:02:00.0 $B > .m0.$$ & [ -n "$b" ] && fps $b 0000:03:00.0 $B > .m1.$$; wait
    echo -e "$a\tb$B\t$(cat .m0.$$)"; [ -n "$b" ] && echo -e "$b\tb$B\t$(cat .m1.$$)"
  done
done
rm -f .m[01].$$
