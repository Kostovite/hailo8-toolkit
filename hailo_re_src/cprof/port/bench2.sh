#!/bin/bash
# bench2.sh REF.hef A.hef B.hef ...: benchmark HEFs on BOTH Hailo-8s at once (one HEF per chip), after checking the
# chips do not disturb each other (they share the chipset uplink): REF alone on each chip, then on both together.
# Falls back to one chip if the concurrent numbers differ from the solo ones by more than TOL percent.
TOL=${TOL:-0.5}
DEVS=(0000:02:00.0 0000:03:00.0)
fps() { hailortcli benchmark "$1" -t 5 --device-id "$2" 2>&1 | grep -A4 '^Summary' | awk -F= '/hw_only/{print $2+0}'; }
ref=$1; shift
s0=$(fps $ref ${DEVS[0]}); s1=$(fps $ref ${DEVS[1]})
fps $ref ${DEVS[0]} > .b0.$$ & fps $ref ${DEVS[1]} > .b1.$$; wait
c0=$(cat .b0.$$); c1=$(cat .b1.$$); rm -f .b[01].$$
echo "calibration $ref: solo $s0 / $s1   concurrent $c0 / $c1"
ok=$(awk -v a=$s0 -v b=$s1 -v c=$c0 -v d=$c1 -v t=$TOL 'BEGIN{
  x=(a-c)/a*100; y=(b-d)/b*100; if(x<0)x=-x; if(y<0)y=-y; z=(a-b)/a*100; if(z<0)z=-z; print (x<=t && y<=t && z<=t)}')
if [ "$ok" != 1 ]; then echo "chips disagree or interfere: one chip at a time"; fi
printf "hef\thw_fps\tdevice\n"
if [ "$ok" = 1 ]; then
  while [ $# -gt 0 ]; do
    a=$1; b=$2; shift; [ $# -gt 0 ] && shift
    fps $a ${DEVS[0]} > .b0.$$ &
    [ -n "$b" ] && fps $b ${DEVS[1]} > .b1.$$
    wait
    printf "%s\t%s\t%s\n" "$a" "$(cat .b0.$$)" 0
    [ -n "$b" ] && printf "%s\t%s\t%s\n" "$b" "$(cat .b1.$$)" 1
    rm -f .b[01].$$
  done
else
  for h in "$@"; do printf "%s\t%s\t%s\n" "$h" "$(fps $h ${DEVS[0]})" 0; done
fi
