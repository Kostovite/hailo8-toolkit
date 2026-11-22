#!/bin/bash
# Exact-integer (or FP) val mAP on GPUs 0+1. Usage: eval2gpu.sh TAG <evaluate.py args...>
# Writes out/stats_TAG_{0,1}.npz and appends the merged result to out/eval.jsonl
set -e
TAG=$1; shift
P=/mnt/storage/anaconda3/envs/general
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/hailoqat
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib PATH=$P/bin:/usr/bin:/bin \
    $P/bin/python evaluate.py "$@" --shard $g/2 --stats ../out/stats_${TAG}_$g.npz > ../out/eval_${TAG}_$g.log 2>&1 &
done
wait
R=$(env -u LD_PRELOAD LD_LIBRARY_PATH=$P/lib $P/bin/python evaluate.py --combine ../out/stats_${TAG}_0.npz ../out/stats_${TAG}_1.npz 2>/dev/null | tail -1)
echo "{\"tag\": \"$TAG\", \"result\": $R}" | tee -a ../out/eval.jsonl
