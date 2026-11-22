#!/bin/bash
# parallel per-context compiles of PP-DocLayoutV3 (turbo/hcc.py), HCC_PAR capped (VM-safe), one partition at a time
source /mnt/storage/lib_project/models/Hailo_model/convert/lib/env.sh; henv; export CUDA_VISIBLE_DEVICES=99 HCC_PAR=4 OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4
cd /mnt/storage/lib_project/models/Hailo_model/hailo_re/turbo
for k in k5_m0.5 k8_m0.5 k10_m0.5; do
  echo "== $k $(date +%T)"; timeout 14400 "$HPY" -u hcc.py /mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g1/q.har /mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g1/part/$k /mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g1/hcc_$k --cap 900 --topk 1 --batches 1 > /mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g1/hcc_$k.log 2>&1; echo "   exit $? $(date +%T)"
  grep -E "ctx|context|FAIL|fail|error|winner|fps|FPS" /mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g1/hcc_$k.log | tail -12
  ls /mnt/storage/lib_project/models/Hailo_model/convert/runs/layout_g1/hcc_$k/*.hef 2>/dev/null && break
done
