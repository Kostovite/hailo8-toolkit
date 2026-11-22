#!/bin/bash
# dilated conv (T_Conv_dil3) at C24 H30 x W: compile, emulator, chip -> which W are wrong
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; W0=$R/probe/runs/dil3/w; GP=/mnt/storage/anaconda3/envs/general
export DIST=normal CUDA_VISIBLE_DEVICES=; cd $R/probe
for w in 48 50 52 56 60 64; do
  mkdir -p $W0/src/W$w
  (cd $R/zoo/ops && TC_C=24 TC_H=30 TC_W=$w env -u LD_PRELOAD LD_LIBRARY_PATH=$GP/lib $GP/bin/python torch_cases.py $W0/src/W$w > /dev/null 2>&1)
  echo $W0/src/W$w/T_Conv_dil3
done > $W0/src.txt
xargs -P 6 -I{} sh -c 'w=$(basename $(dirname {})); mkdir -p '$W0'/out/$w; '$HPY' -u sweep.py run @{} '$W0'/out/$w > /dev/null 2>&1' < $W0/src.txt
ls -d $W0/out/*/T_Conv_dil3 | while read d; do [ -s $d/m.hef ] && echo $d; done > $W0/dirs.txt
$HPY -u silent/chip_vs_emu.py emu 6 $W0/dirs.txt > $W0/emu.log 2>&1
DEVICE=0000:03:00.0 $HPY -u silent/chip_vs_emu.py chip $W0/res.jsonl $W0/dirs.txt > $W0/chip.log 2>&1
for d in $(cat $W0/dirs.txt); do echo "$(basename $(dirname $d)) $(grep -o 'defuse(conv1[^)]*)' $d/auto.alls) $(grep -o '"frac": [0-9.]*' <(grep $d $W0/res.jsonl))"; done
echo WDONE
