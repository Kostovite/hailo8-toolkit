#!/bin/bash
# usage: emu_shard.sh HAR IN_NPY OUT_PREFIX fp|q GPUS(e.g. 0,1,2,3)  -> OUT_PREFIX_<i>.npy (all rows, merged)
set -uo pipefail; source "$(dirname "$0")/env.sh"
HAR=$1; IN=$2; PRE=$3; M=$4; IFS=, read -ra G <<< "$5"
N=$(python3 -c "import numpy as np;print(len(np.load('$IN',mmap_mode='r')))"); [ -n "${EMU_ROWS:-}" ] && [ "$EMU_ROWS" -lt "$N" ] && N=$EMU_ROWS; K=${#G[@]}; pids=()
for j in "${!G[@]}"; do s=$((N*j/K)); e=$((N*(j+1)/K))
  ( henv; export CUDA_VISIBLE_DEVICES=${G[$j]}; "$HPY" -u $CV/lib/emu.py "$HAR" "$IN" "${PRE}.s$j" "$M" $s $e ) > "${PRE}.s$j.log" 2>&1 & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { echo "emu shard failed: see ${PRE}.s*.log"; exit 1; }
python3 - "$PRE" $K <<'P'
import sys, glob, numpy as np
pre, K = sys.argv[1], int(sys.argv[2]); i = 0
while glob.glob(f"{pre}.s0_{i}.npy"):
    parts = [np.load(f"{pre}.s{j}_{i}.npy", mmap_mode="r") for j in range(K)]
    o = np.lib.format.open_memmap(f"{pre}_{i}.npy", "w+", parts[0].dtype, (sum(map(len, parts)),) + parts[0].shape[1:]); a = 0
    for p in parts: o[a:a + len(p)] = p; a += len(p)
    o.flush(); print("merged", f"{pre}_{i}.npy", o.shape); i += 1
P
rm -f "${PRE}".s*_*.npy
