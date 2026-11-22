#!/bin/bash
# Shape sweep: every torch op case at several (C, H, W); chip vs SDK_BIT_EXACT with in-distribution inputs.
# Memory cap: PAR quantize/compile processes (~3 GB each).
source /mnt/storage/lib_project/models/Hailo_model/hailo_re/env.sh
R=/mnt/storage/lib_project/models/Hailo_model/hailo_re; S=$R/probe/scan; cd $R/probe
PAR=${PAR:-20}; GP=/mnt/storage/anaconda3/envs/general
export DIST=normal CUDA_VISIBLE_DEVICES=
for shp in ${SHAPES:-"8 8 8" "32 40 40" "64 16 16" "128 32 32" "256 12 12" "24 30 50"}; do
  set -- $shp; tag=C$1_H$2_W$3; SRC=$S/src/$tag; OUT=$S/out/$tag; mkdir -p $SRC $OUT
  echo "== $tag export $(date +%T)"
  (cd $R/zoo/ops && TC_C=$1 TC_H=$2 TC_W=$3 env -u LD_PRELOAD LD_LIBRARY_PATH=$GP/lib $GP/bin/python torch_cases.py $SRC > $S/$tag.export.log 2>&1)
  ls -d $SRC/T_* | while read d; do [ -f $d/m.onnx ] && echo $d; done > $S/$tag.src.txt
  echo "== $tag compile $(wc -l < $S/$tag.src.txt) cases $(date +%T)"
  xargs -P $PAR -I{} sh -c "$HPY -u sweep.py run @{} $OUT > /dev/null 2>&1" < $S/$tag.src.txt
  ls -d $OUT/T_* | while read d; do [ -s $d/m.hef ] && echo $d; done > $S/$tag.dirs.txt
  grep -h '"stage"' $OUT/T_*/result.json 2>/dev/null | grep -o '"stage": "[a-z]*"' | sort | uniq -c > $S/$tag.stages.txt
  echo "== $tag emu $(wc -l < $S/$tag.dirs.txt) HEFs $(date +%T)"
  $HPY -u silent/chip_vs_emu.py emu 16 $S/$tag.dirs.txt > $S/$tag.emu.log 2>&1
  split -n l/2 $S/$tag.dirs.txt $S/$tag.half.
  DEVICE=0000:02:00.0 $HPY -u silent/chip_vs_emu.py chip $S/$tag.res0.jsonl $S/$tag.half.aa > $S/$tag.chip0.log 2>&1 &
  DEVICE=0000:03:00.0 $HPY -u silent/chip_vs_emu.py chip $S/$tag.res1.jsonl $S/$tag.half.ab > $S/$tag.chip1.log 2>&1 &
  wait
  echo "== $tag done $(date +%T)"
done
echo ALLDONE
