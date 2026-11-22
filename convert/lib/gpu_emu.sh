#!/bin/bash
# usage: gpu_emu.sh HAR EVAL_X OUT_PREFIX GPU   (DFC env capture once per HAR -> torch env exact executor on the GPU)
# Steps (= hailo8-exact-qat tools/har_pipeline.sh): HSim ALL_OPS capture of eval row 0, numpy op pickle, bias lattice,
# graph26 --cont coverage (every layer fed HSim's inputs) must be exact or <= 1 LSB on < 0.1 % of elements, then QATNet
# over all rows in batches.
set -uo pipefail; source "$(dirname "$0")/env.sh"
HAR=$1; EV=$2; PRE=$3; GPU=$4; CAP=$(dirname $PRE)/gcap; G=/mnt/storage/anaconda3/envs/general
# recapture when there is no capture or the HAR is newer than it
if [ ! -f $CAP/_all_ops_np.pkl ] || [ $HAR -nt $CAP/_all_ops_np.pkl ]; then rm -rf $CAP $CAP.*; ( henv; export CUDA_VISIBLE_DEVICES=$GPU; "$HPY" -u $CV/lib/gpu_emu.py capture $HAR $EV $CAP ) || exit 1; fi
( cd $QAT/hailoqat && CUDA_VISIBLE_DEVICES=$GPU env -u LD_PRELOAD LD_LIBRARY_PATH=$G/lib OMP_NUM_THREADS=4 $G/bin/python graph26.py $CAP.hn $CAP $CAP.input_nhwc.npy cuda --cont --json=$CAP.cov.json ) > $CAP.check.txt 2>&1
python3 - $CAP.cov.json $CAP.check.txt <<'P' || { echo "GPU_EMU: executor not exact enough on the capture row, refusing"; exit 3; }
import json, re, sys
L = json.load(open(sys.argv[1]))["layers"]; txt = open(sys.argv[2]).read()
mx = {m[0]: int(m[1]) for m in re.findall(r"MISMATCH (\S+) \(\w+\): \d+/\d+ max (\d+)", txt)}
bad = [l for l in L if l["status"] != "exact"]; worst = [l for l in bad if not (l["status"] == "mismatch" and mx.get(l["layer"], 9) <= 1 and l["bad"] < 1e-3 * l["numel"])]
print(f"GPU_EMU coverage: {len(L) - len(bad)}/{len(L)} layers exact; {len(bad) - len(worst)} within 1 LSB on <0.1% elements: " + ", ".join(f"{l['layer']} {l['bad']}/{l['numel']}" for l in bad))
sys.exit(1 if worst else 0)
P
cd $QAT && CUDA_VISIBLE_DEVICES=$GPU env -u LD_PRELOAD LD_LIBRARY_PATH=$G/lib OMP_NUM_THREADS=4 $G/bin/python -u $CV/lib/gpu_emu.py run $CAP $EV $PRE ${GPU_EMU_BS:-8}
