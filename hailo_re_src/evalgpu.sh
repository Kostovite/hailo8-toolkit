#!/bin/bash
# Exact-integer (or FP) val mAP sharded over GPUS (default "0 1 2 3"). Usage: [GPUS="0 1"] evalgpu.sh TAG <evaluate.py args...>
# Writes out/stats_TAG_{k}.npz and appends the merged result to out/eval.jsonl
set -e
TAG=$1; shift
GPUS=(${GPUS:-0 1 2 3})
N=${#GPUS[@]}
P=/mnt/storage/anaconda3/envs/general
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat
ST=()
for i in "${!GPUS[@]}"; do
  rm -f ../out/stats_${TAG}_$i.npz
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} env -u LD_PRELOAD PYTHONNOUSERSITE=1 LD_LIBRARY_PATH=$P/lib PATH=$P/bin:/usr/bin:/bin \
    $P/bin/python evaluate.py "$@" --shard $i/$N --stats ../out/stats_${TAG}_$i.npz > ../out/eval_${TAG}_$i.log 2>&1 &
  ST+=(../out/stats_${TAG}_$i.npz)
done
wait
for f in "${ST[@]}"; do [ -f "$f" ] || { echo "EVAL $TAG FAILED: missing $f (see out/eval_${TAG}_*.log)"; exit 1; }; done
R=$(env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib $P/bin/python evaluate.py --combine "${ST[@]}" 2>/dev/null | tail -1)
echo "{\"tag\": \"$TAG\", \"result\": $R}" | tee -a ../out/eval.jsonl
