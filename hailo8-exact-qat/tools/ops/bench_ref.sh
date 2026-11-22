#!/bin/bash
# zoo/bench_ref.sh: download Hailo's precompiled Hailo-8 HEF of every public network (hailo8_public.csv) and measure
# FPS on ONE idle chip at batch 1 and 8 (the published conditions). Appends to zoo/ref_fps.tsv; skips done networks.
Z=/mnt/storage/lib_project/models/Hailo_model/hailo_re/zoo; cd $Z
P=/mnt/storage/anaconda3/envs/general/bin/python
touch ref_fps.tsv
$P -c "import csv; [print(r['network'], r['hef_url']) for r in csv.DictReader(open('hailo8_public.csv')) if r['hef_url']]" | sort -u |
while read net url; do
  grep -q "^$net	" ref_fps.tsv && continue
  [ -s hef_ref/$net.hef ] || curl -sL -m 600 -o hef_ref/$net.hef "$url"
  if hailo-smi | grep -q "pid "; then echo "chip busy, stopping"; exit 1; fi  # never share a measurement device
  f1=$(timeout 120 hailortcli run hef_ref/$net.hef --batch-size 1 -t 5 2>&1 | grep -oE 'FPS: [0-9.]+' | head -1 | cut -d' ' -f2)
  f8=$(timeout 120 hailortcli run hef_ref/$net.hef --batch-size 8 -t 5 2>&1 | grep -oE 'FPS: [0-9.]+' | head -1 | cut -d' ' -f2)
  printf "%s\t%s\t%s\t%s\n" "$net" "${f1:-ERR}" "${f8:-ERR}" "$(du -m hef_ref/$net.hef | cut -f1)MB" | tee -a ref_fps.tsv
done
echo BENCH_REF_DONE
