#!/bin/bash
set -euo pipefail
D=/mnt/storage/lib_project/models/moe/runs/codex_fusion_20261119/hailo_current_20261119
source /mnt/storage/lib_project/hailo8-exact-qat/tools/env.sh
export CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2 OPT_LEVEL=0
export LD_PRELOAD="/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"
cd "$D"
"$HPY" -u "$D/aux_models/layout_resize_candidate.py" "$D/r5_nofuse_parsed.har" "$D/r5_resize_candidate.har" /mnt/storage/lib_project/models/Hailo_model/preproc_hailo/calib/layout_1024.npy mix16
"$HPY" -u /mnt/storage/lib_project/models/Hailo_model/hailo_re/compile_hef.py "$D/r5_resize_candidate.har" "$D/r5_resize_candidate.hef"
"$HPY" -u /mnt/storage/lib_project/hailo8-exact-qat/tools/fixes/hailo8_guard.py "$D/r5_resize_candidate.hef" "$D/r5_resize_candidate.har"
echo 'Candidate only: heldout emulation and chip tests are still required.'
