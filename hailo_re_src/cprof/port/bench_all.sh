#!/bin/bash
# wait for the portfolio, then benchmark every HEF one at a time (chip is the measurement device: never in parallel)
cd "$(dirname "$0")"
while [ $(grep -c ' done$' port.log) -lt $(wc -l < variants.txt) ]; do sleep 30; done
printf "tag\twall\titers\thw_fps\tlat_ms\n" > results.tsv
printf "base\t~600s\t-\t%s\n" "$(hailortcli benchmark ../nano/nano_base.hef -t 5 2>&1 | grep -A4 '^Summary' | awk -F= '/hw_only/{f=$2}/hw\)/{l=$2}END{print f"\t"l}')" >> results.tsv
for t in $(cat variants.txt | cut -d'|' -f1); do
  wall=$(grep -m1 '^real' $t.log | awk '{print $2}')
  it=$(grep -oE 'Partitioner finished after [0-9]+' $t.log | tail -1 | awk '{print $4}')
  if [ -f $t.hef ]; then
    r=$(hailortcli benchmark $t.hef -t 5 2>&1 | grep -A4 '^Summary' | awk -F= '/hw_only/{f=$2}/hw\)/{l=$2}END{print f"\t"l}')
  else r="FAIL\t-"; fi
  printf "%s\t%s\t%s\t%b\n" "$t" "$wall" "$it" "$r" >> results.tsv
done
column -t results.tsv
