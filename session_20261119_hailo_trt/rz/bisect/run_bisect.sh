#!/bin/bash
# usage: run_bisect.sh HAR K TAG   -> compile each chunk alone, summary in TAG_summary.txt
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh >/dev/null; export CUDA_VISIBLE_DEVICES=99 OMP_NUM_THREADS=4
cd "$(dirname "$0")"; T=/mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo
"$HPY" chunks.py $1 $2 $3 > $3_chunks.txt 2>&1 || exit 1
for i in $(seq 0 $(($2-1))); do ( "$HPY" -u $T/compile_sub.py $1 $3_$i.json $PWD/$3_c$i > $3_c$i.log 2>&1 ) & done; wait
for i in $(seq 0 $(($2-1))); do echo "chunk $i $(sed -n ${i}p $3_chunks.txt | head -0) $(python3 -c "import json;d=json.load(open('$3_c$i.json'));print(d.get('ok'),str(d.get('error',''))[:150])" 2>/dev/null)"; done > $3_summary.txt
