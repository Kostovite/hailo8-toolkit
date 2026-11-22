#!/bin/bash
# usage: gpu_suite_all.sh GPU CASE_LIST_FILE  (lines: NAME HN CAP INPUT) -> results/gpu_suite/NAME.json + summary line
Q=/mnt/storage/lib_project/hailo8-exact-qat; G=/mnt/storage/anaconda3/envs/general; OUT=$Q/results/gpu_suite; mkdir -p $OUT
while read -r n hn cap inp; do
  [ -z "$n" ] && continue; [ -f $OUT/$n.json ] && continue
  CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True env -u LD_PRELOAD LD_LIBRARY_PATH=$G/lib OMP_NUM_THREADS=4 \
    timeout 3600 $G/bin/python $Q/tools/ops/gpu_suite.py $n $hn $cap $inp $OUT/$n.json > $OUT/$n.log 2>&1
  echo "$n rc=$? $(tail -n 1 $OUT/$n.log | cut -c1-300)" >> $OUT/summary_gpu$1.txt
done < $2
